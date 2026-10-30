"""KV-cached speculative SAMPLING for the sequential MTP heads (Leviathan/Chen).

The MTP sequential is the DRAFT (depth j samples from its head distribution p_j); the trunk + NTP head
is the TARGET (q):

  * draft d_j ~ p_j, conditioned on the previously-sampled drafts;
  * verify the block [n0, d_1..d_k] in ONE cached trunk forward -> q_0..q_k;
  * accept d_j w.p. min(1, q(d_j)/p(d_j)); on reject resample from normalize((q - p)+); if all k
    are accepted, sample a bonus token from q.

Every committed token is distributed exactly as a sample from the target head, so this is lossless
for ANY draft distribution — and unlike greedy top-1 acceptance it rewards matching the verifier's
full distribution. The trunk keeps a `DynamicCache`; the sequential depths re-draft cache-free.
"""

import torch
import torch.nn.functional as F

from .acceptance import aggregate_acceptance  # noqa: F401 (re-export for callers)


def _filter(logits, top_k, top_p):
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and 0.0 < top_p < 1.0:
        s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(s_logits, dim=-1)
        remove = torch.cumsum(probs, dim=-1) - probs > top_p          # keep the first token
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, s_idx, remove)
        logits = logits.masked_fill(mask, float("-inf"))
    return logits


def norm_logits(logits, temperature, top_k=0, top_p=0.0):
    """1-D `[V]` logits -> `[V]` probabilities (temperature, then top-k / top-p)."""
    return F.softmax(_filter(logits / max(temperature, 1e-6), top_k, top_p), dim=-1)


def _sample(probs):
    return int(torch.multinomial(probs, 1))


def _residual(q, p):
    """normalize((q - p)+) — the speculative-sampling rejection-resample distribution."""
    r = torch.clamp(q - p, min=0.0)
    s = r.sum()
    return r / s if s > 0 else q


@torch.no_grad()
def sample_self_speculative_sequential_cached(model, prompt, max_new_tokens,
                                           temperature=1.0, top_k=0, top_p=0.0,
                                           cache_sequential=False, n_draft=None):
    """Lossless (in distribution) KV-cached sequential speculative SAMPLING. Returns (ids[1,n], stats).

    `n_draft` (gamma) = drafts proposed per round; None = `config.n_mtp`. Fewer than n_mtp drafts the
    first gamma depths only. MORE than n_mtp is allowed only for a weight-SHARED sequential
    (`mtp_share_weights`), where every depth re-enters the same module, so any depth is defined --
    that is what lets one weight-shared checkpoint be evaluated at every gamma. `stats["n_draft"]`
    records it; aggregate with `aggregate_acceptance(stats, n_draft)`.

    `cache_sequential` also caches the sequential depths. ⚠ DEFAULT OFF because it measures ~3x SLOWER:
    the drafter is not the bottleneck (2 layers over ~1024 positions is one efficient matmul), so
    caching it saves nothing while adding an O(L) `torch.cat` per call. Correct but not useful; do
    not enable without a fresh measurement.

    The cache, when on, is append-only. Depth d at position t consumes x_{t+d}, so with x_0..x_L
    confirmed, depth d is settled only for t <= L-d and its cache LAGS by d. Each round splits into
    at most three slices: already-cached (read-only), newly confirmed (write), and the tail that
    consumes DRAFTS (read-only — a rejected draft left in the KV would corrupt later drafts).
    """
    from transformers import DynamicCache
    model.eval()
    dev = prompt.device
    n_mtp = int(model.config.n_mtp)
    k = n_mtp if n_draft is None else int(n_draft)
    if k < 0 or (k > n_mtp and not getattr(model.config, "mtp_share_weights", False)):
        raise ValueError(f"n_draft={k} needs 0 <= n_draft <= n_mtp={n_mtp} for an unshared sequential "
                         "(a weight-shared sequential may unroll past n_mtp)")
    cache = DynamicCache()
    dcache = [DynamicCache() for _ in range(k)] if cache_sequential else None
    n_forward = 0
    proposed, accepted, round_accepts = [0] * k, [0] * k, []

    def trunk_step(tokens, start):
        nonlocal n_forward
        n_forward += 1
        cp = torch.arange(start, start + tokens.size(1), device=dev)
        H = model.trunk_forward(tokens, past_key_values=cache, use_cache=True, cache_position=cp)
        return H, model.logits(H)[0]                          # ([1,t,d], [t,V] logits)

    def draft(H_all, seq, n0):
        """Sequential rollout from the anchor: sample k drafts; return (drafts, ps) with ps[i] the draft
        distribution that produced drafts[i] (same conditioning as the target dist that verifies it)."""
        hk, gen, drafts, ps = H_all, [n0], [], []
        for depth in range(1, k + 1):
            fut = torch.tensor([gen[:depth]], device=dev)
            token_ids = torch.cat([seq[:, depth:], fut], dim=1)
            hk = model.dsmtp_layer(hk, token_ids, depth)
            p = norm_logits(model.logits(hk[:, -1:], depth)[0, -1], temperature, top_k, top_p)  # per-head final norm when set
            d = _sample(p)
            ps.append(p); drafts.append(d); gen.append(d)
        return drafts, ps

    def draft_cached(H_all, seq, n0):
        """Same rollout, extending each depth's own append-only cache by ~(m + k) positions instead of
        re-running it over all L. Slice semantics are documented on the function above."""
        L = seq.size(1)
        have = [dc.get_seq_length() for dc in dcache]
        start = max(0, min(min(have), L))
        h_in = H_all[:, start:L]                       # h^{depth-1} over [start, L); h^0 to begin with
        drafts, ps = [], []
        for depth in range(1, k + 1):
            dc = dcache[depth - 1]
            xs = torch.cat([seq, torch.tensor([[n0] + drafts], device=dev)], dim=1)   # x_0..x_{L+|drafts|}
            tok = xs[:, start + depth: L + depth]       # position t -> token x_{t+depth}
            n_conf = max(0, min(L, L - depth + 1) - start)   # [start, L-depth] are confirmed
            n_over = max(0, min(have[depth - 1] - start, n_conf))
            outs, pos = [], start

            def run(n, readonly):
                nonlocal pos
                if n <= 0:
                    return
                o = pos - start
                outs.append(model.dsmtp_layer(
                    h_in[:, o:o + n], tok[:, o:o + n], depth,
                    past_key_values=dc, cache_position=torch.arange(pos, pos + n, device=dev),
                    cache_readonly=readonly))
                pos += n

            run(n_over, True)                 # already cached -> recompute the hidden, do not re-write
            run(n_conf - n_over, False)       # newly confirmed -> append-only write
            run(L - pos, True)                # draft-dependent tail -> never stored
            h_in = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
            p = norm_logits(model.logits(h_in[:, -1:], depth)[0, -1], temperature, top_k, top_p)
            d = _sample(p)
            ps.append(p); drafts.append(d)
        return drafts, ps

    if cache_sequential:
        draft = draft_cached

    L = prompt.size(1)
    H, ntp = trunk_step(prompt, 0)                            # prefill
    H_buf, seq, confirmed = H, prompt.clone(), L
    n0 = _sample(norm_logits(ntp[-1], temperature, top_k, top_p))   # target-distributed first token
    drafts, ps = draft(H_buf, seq, n0)
    out = []

    while len(out) < max_new_tokens:
        out.append(n0)                                        # n0 is always target-distributed -> commit
        if len(out) >= max_new_tokens:
            break
        if k == 0:
            H, ntp = trunk_step(torch.tensor([[n0]], device=dev), confirmed)
            H_buf = torch.cat([H_buf, H], 1); seq = torch.cat([seq, torch.tensor([[n0]], device=dev)], 1)
            confirmed += 1; n0 = _sample(norm_logits(ntp[-1], temperature, top_k, top_p)); continue

        block = torch.tensor([[n0] + drafts], device=dev)
        H, ntp = trunk_step(block, confirmed)                 # ntp[i] verifies drafts[i] (same conditioning)
        m, correction = 0, None
        for i in range(k):
            proposed[i] += 1
            q, p, d = norm_logits(ntp[i], temperature, top_k, top_p), ps[i], drafts[i]
            if torch.rand((), device=dev) < min(1.0, (q[d] / p[d]).item()):   # accept w.p. min(1, q/p)
                accepted[i] += 1; out.append(d); m += 1
                if len(out) >= max_new_tokens:
                    break
            else:
                correction = _sample(_residual(q, p))         # reject -> resample from (q - p)+
                break
        round_accepts.append(m)
        if len(out) >= max_new_tokens:
            break
        if correction is None:                                # all k accepted -> bonus from the target
            correction = _sample(norm_logits(ntp[k], temperature, top_k, top_p))

        cache.crop(confirmed + m + 1)                         # drop rejected drafts from the trunk cache
        H_buf = torch.cat([H_buf, H[:, : m + 1]], 1)
        seq = torch.cat([seq, block[:, : m + 1]], 1)
        confirmed += m + 1
        n0 = correction
        drafts, ps = draft(H_buf, seq, n0)

    gen = torch.tensor([out[:max_new_tokens]], device=dev)
    return gen, {"n_forward": n_forward, "proposed": proposed, "accepted": accepted,
                 "round_accepts": round_accepts, "n_generated": len(out[:max_new_tokens]),
                 "n_draft": k}
