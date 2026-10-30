"""KV-cached speculative SAMPLING for META parallel-head MTP (block drafting).

The meta counterpart of `sample_speculative_sequential_cached.py` — same Leviathan/Chen accept/residual
rule, same trunk `DynamicCache` bookkeeping, same `stats` fields (so `aggregate_acceptance` and
`evaluate_self_spec` work unchanged). The ONLY difference is the draft:

  * sequential draft: teacher-forced rollout — depth j conditions on the previously-sampled drafts,
    so p_j approximates the ONE-STEP conditional P(x_{t+1+j} | x_<=t, d_1..d_{j-1});
  * meta draft (here): head j reads ONLY the trunk hidden sequence (x_<=t) and proposes all k
    tokens in ONE shot — p_j is the j-step MARGINAL P(x_{t+1+j} | x_<=t), blind to n0 and to the
    other drafts: the intermediate path is integrated out, not conditioned on.

The mismatch does NOT break losslessness: the accept ratio only has to use the SAME p the draft was
sampled from, so the committed token is target-distributed for any draft distribution. A marginal
draft is simply rejected more often, which makes the meta-vs-sequential acceptance gap a direct
measurement of that path noise. Heads re-draft cache-free each round; the trunk stays cached.
"""

import torch

from .acceptance import aggregate_acceptance  # noqa: F401 (re-export for callers)
from .sequential import _residual, _sample, norm_logits


@torch.no_grad()
def sample_self_speculative_meta_cached(model, prompt, max_new_tokens,
                                        temperature=1.0, top_k=0, top_p=0.0):
    """Lossless (in distribution) KV-cached META block speculative SAMPLING. Returns (ids[1,n], stats)."""
    from transformers import DynamicCache
    model.eval()
    dev = prompt.device
    k = int(model.config.n_mtp)
    cache = DynamicCache()
    n_forward = 0
    proposed, accepted, round_accepts = [0] * k, [0] * k, []

    def trunk_step(tokens, start):
        nonlocal n_forward
        n_forward += 1
        cp = torch.arange(start, start + tokens.size(1), device=dev)
        H = model.trunk_forward(tokens, past_key_values=cache, use_cache=True, cache_position=cp)
        return H, model.logits(H)[0]                          # ([1,t,d], [t,V] logits)

    def draft(H_all):
        """One-shot parallel block draft at the anchor (the LAST confirmed position t): head j's
        output at t predicts x_{t+1+j}, so heads 1..k propose the k tokens AFTER n0 (= x_{t+1}) —
        the same block slots the sequential fills, but as marginals over the intermediate tokens."""
        drafts, ps = [], []
        for j in range(1, k + 1):
            hj = model.head_forward(H_all, j)                 # head j over the confirmed hidden seq
            p = norm_logits(model.logits(hj[:, -1:], j)[0, -1], temperature, top_k, top_p)
            d = _sample(p)
            ps.append(p); drafts.append(d)
        return drafts, ps

    L = prompt.size(1)
    H, ntp = trunk_step(prompt, 0)                            # prefill
    H_buf, confirmed = H, L
    n0 = _sample(norm_logits(ntp[-1], temperature, top_k, top_p))   # target-distributed first token
    drafts, ps = draft(H_buf)
    out = []

    while len(out) < max_new_tokens:
        out.append(n0)                                        # n0 is always target-distributed -> commit
        if len(out) >= max_new_tokens:
            break
        if k == 0:
            H, ntp = trunk_step(torch.tensor([[n0]], device=dev), confirmed)
            H_buf = torch.cat([H_buf, H], 1)
            confirmed += 1; n0 = _sample(norm_logits(ntp[-1], temperature, top_k, top_p)); continue

        block = torch.tensor([[n0] + drafts], device=dev)
        H, ntp = trunk_step(block, confirmed)                 # ntp[i] verifies drafts[i] (target conditioning)
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
        confirmed += m + 1
        n0 = correction
        drafts, ps = draft(H_buf)

    gen = torch.tensor([out[:max_new_tokens]], device=dev)
    return gen, {"n_forward": n_forward, "proposed": proposed, "accepted": accepted,
                 "round_accepts": round_accepts, "n_generated": len(out[:max_new_tokens])}
