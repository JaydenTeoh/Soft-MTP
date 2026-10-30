"""Shared MTP evaluation functions.

Used by `eval.callback.MTPEvalCallback` and by `eval_spec.py`.

- `evaluate_mtp`: per-head CE + verifier-agreement (KL(q||p), top-1/k, mass profile, sharpening).
- `evaluate_self_spec`: speculative-SAMPLING acceptance + wall-clock speedup vs the cached-AR baseline.

All three are DDP-sharded (`rank`/`world`): disjoint batch/prompt shards + raw-accumulator all-reduce,
bit-identical to the single-process eval.
"""

import math
from collections import defaultdict

import torch
import torch.nn.functional as F


@torch.no_grad()
def _reduce_eval_mtp(ce_tot, ce_cnt, agg, nf, device):
    """All-reduce (SUM) the raw evaluate_mtp accumulators across DDP ranks, so a sharded
    multi-GPU eval is bit-identical to the single-GPU eval over the same batches."""
    import torch.distributed as dist
    dist.all_reduce(ce_tot)
    dist.all_reduce(ce_cnt)
    keys = ("count", "kl_verifier_mtp", "tv_qp")
    buf = torch.zeros(nf, len(keys), device=device)
    for j in range(1, nf):
        for ki, key in enumerate(keys):
            buf[j, ki] = agg[j][key]
    dist.all_reduce(buf)
    for j in range(1, nf):
        for ki, key in enumerate(keys):
            agg[j][key] = buf[j, ki].item()


@torch.no_grad()                       # WITHOUT this the per-batch forward graphs accumulate across
                                       # max_batches (`ce_tot += CE(...)` keeps each graph alive) -> the
                                       # eval grows ~linearly in max_batches and OOMs (mbs-independent).
def evaluate_mtp(model, loader, device, max_batches, rank=0, world=1):
    """Per-head MTP CE + verifier-agreement metrics (heads 1..n_mtp), for wandb.

    One pass per batch: NTP verifier logits q (head 0) and each head's logits p ->
    per-head CE plus KL(verifier||MTP), top-1 agreement, top-k overlap. The KL/agree
    metrics are the ones the recovered-KL objective is meant to improve.

    DDP-sharded: with `world>1` each rank processes the disjoint batch subset
    `bi % world == rank` (union == the same `max_batches` batches), then the raw
    sum/count accumulators are all-reduced -> identical to a single-GPU eval.
    """
    from .distill import distill_metrics_for_head

    was_training = model.training      # restore (not force-train) at exit: offline callers stay in eval
    model.eval()
    nf = model.n_heads_total
    ce_tot = torch.zeros(nf, device=device)
    ce_cnt = torch.zeros(nf, device=device)
    agg = {j: defaultdict(float) for j in range(1, nf)}
    for bi, (input_ids, labels) in enumerate(loader):
        if bi >= max_batches:
            break
        if world > 1 and bi % world != rank:        # disjoint shard of the same max_batches
            continue
        input_ids, labels = input_ids.to(device), labels.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            H = model.trunk_forward(input_ids)
            hiddens = model.all_head_hiddens(H, input_ids)   # parallel (Meta) or sequential (DeepSeek)
            q_logits = model.logits(hiddens[0])           # verifier (NTP)
            for j in range(nf):
                tgt = labels[:, j:]                      # head j's target: one slice, T-j long
                # meta heads are full length; the sequential already returns exactly T-j -> crop to match
                p_logits = model.logits(hiddens[j], j)[:, :tgt.size(1)]
                ce_tot[j] += F.cross_entropy(
                    p_logits.reshape(-1, p_logits.size(-1)).float(),
                    tgt.reshape(-1), ignore_index=-100, reduction="sum")
                ce_cnt[j] += (tgt != -100).sum()
                if j >= 1:
                    m = distill_metrics_for_head(p_logits, q_logits, tgt, j)
                    if m is not None:
                        for key, val in m.items():
                            agg[j][key] += val
                del p_logits
    model.train(was_training)
    if world > 1:
        _reduce_eval_mtp(ce_tot, ce_cnt, agg, nf, device)
        import torch.distributed as dist
    ce = (ce_tot / ce_cnt.clamp(min=1)).tolist()
    # Head 0 is the NTP trunk head -> its CE IS the NTP objective; emit it under its canonical name
    # (val/ntp_loss), NOT as a draft head. Heads 1..n_mtp are the MTP draft heads (mtp/h{j}_ce).
    out = {"val/ntp_loss": ce[0]}
    out.update({f"mtp/h{j}_ce": ce[j] for j in range(1, nf)})
    for j, a in agg.items():
        c = max(1.0, a["count"])
        out[f"mtp/h{j}_kl_verifier"] = a["kl_verifier_mtp"] / c   # D_KL(verifier || draft)
        out[f"mtp/h{j}_tv_verifier"] = a["tv_qp"] / c             # TV(q,p); acceptance is 1 - TV
    return out


def summarize_val_losses(mtp_out):
    """ADD the derived val-set summary losses to `evaluate_mtp`'s output. `evaluate_mtp` already names the
    NTP head `val/ntp_loss` (head 0's CE) and the MTP draft heads `mtp/h{j>=1}_ce`; this derives:
      * `val/ppl` = exp(`val/ntp_loss`).
      * `val/mtp_loss` = mean MTP draft-head hard CE (token-weighted per head upstream).
      * `val/mtp_kl_verifier` = mean D_KL(verifier || draft).
      * `val/mtp_tv_verifier` = mean TV(verifier, draft). Speculative acceptance is exactly 1 - TV,
        so this is the teacher-forced predictor of the acceptance that decoding measures.

    Purely ADDITIVE (consumes no keys); each field is skipped if its inputs are absent (no NTP head on a
    curvature-only eval step; no MTP heads when `n_mtp == 0`)."""
    out = {}
    ntp = mtp_out.get("val/ntp_loss")
    if ntp is not None:
        out["val/ppl"] = math.exp(ntp)
    draft_ces = [v for k, v in mtp_out.items()
                 if k.startswith("mtp/h") and k.endswith("_ce")]
    if draft_ces:
        out["val/mtp_loss"] = sum(draft_ces) / len(draft_ces)
    kls = [v for k, v in mtp_out.items() if k.startswith("mtp/h") and k.endswith("_kl_verifier")]
    if kls:
        out["val/mtp_kl_verifier"] = sum(kls) / len(kls)
    tvs = [v for k, v in mtp_out.items() if k.startswith("mtp/h") and k.endswith("_tv_verifier")]
    if tvs:
        out["val/mtp_tv_verifier"] = sum(tvs) / len(tvs)
    return out


@torch.no_grad()
def evaluate_self_spec(model, loader, device, spec_fn, ar_fn, n_prompts=8, prompt_len=128,
                       new_tokens=64, rank=0, world=1):
    """M6 self-speculative decoding eval (greedy, lossless == plain AR), for wandb.

    `spec_fn(model, prompt, n) -> (ids, stats)` is the self-spec decoder; `ar_fn(model,
    prompt, n) -> ids` is its FAIR AR baseline (Meta: both KV-cached; sequential: both
    cache-free). Drafts -> NTP-verify -> accept longest prefix, timed against `ar_fn` on a
    few val prompts. Returns `spec/*` metrics: tokens/forward (the acceptance headline),
    per-head draft acceptance, alpha, and the measured wall-clock speedup.

    Run under bf16 autocast (the representative inference regime). NOTE: `spec/lossless`
    is the FRACTION of prompts whose spec output bit-matches AR; greedy spec is exactly
    lossless only in exact arithmetic (the fp32 unit test) — under bf16 a block forward
    and a single-token forward round differently, so near-tie argmaxes can flip. A
    confident (trained) model stays ~1.0; a near-uniform random model won't.
    """
    import time

    from ..decoding import aggregate_acceptance

    if model.config.n_mtp == 0:
        return {}
    prompts = []
    for input_ids, _ in loader:
        for row in input_ids:
            if row.size(0) >= prompt_len:
                prompts.append(row[:prompt_len][None].to(device))
            if len(prompts) >= n_prompts:
                break
        if len(prompts) >= n_prompts:
            break
    if not prompts:
        return {}
    # DDP-sharded: each rank decodes a disjoint slice of the SAME prompt set; stats are gathered
    # and the aggregate is computed over the union -> identical to a single-GPU eval, ~world x faster.
    my_prompts = prompts[rank::world] if world > 1 else prompts

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    was_training = model.training
    model.eval()
    stats, spec_s, ar_s, n_match = [], 0.0, 0.0, 0
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        spec_fn(model, prompts[0], 8)   # warm FLA/triton kernels
        ar_fn(model, prompts[0], 8)
        sync()
        for pr in my_prompts:
            sync(); t = time.perf_counter()
            g_spec, s = spec_fn(model, pr, new_tokens)
            sync(); spec_s += time.perf_counter() - t
            sync(); t = time.perf_counter()
            g_ar = ar_fn(model, pr, new_tokens)
            sync(); ar_s += time.perf_counter() - t
            stats.append(s)
            n_match += int(bool(torch.equal(g_spec, g_ar)))
    model.train(was_training)

    n_eval = len(my_prompts)
    if world > 1:
        import torch.distributed as dist
        gathered = [None] * world
        dist.all_gather_object(gathered, stats)     # combine raw per-prompt stats across ranks
        stats = [s for sub in gathered for s in sub]
        red = torch.tensor([spec_s, ar_s, float(n_match), float(n_eval)], device=device)
        dist.all_reduce(red)                         # sum decode times + match count + prompt count
        spec_s, ar_s = red[0].item(), red[1].item()
        n_match, n_eval = int(round(red[2].item())), int(round(red[3].item()))

    agg = aggregate_acceptance(stats, model.config.n_mtp)
    tot = n_eval * new_tokens
    ar_tps, spec_tps = tot / max(ar_s, 1e-8), tot / max(spec_s, 1e-8)
    out = {
        "spec/tokens_per_forward": agg["tokens_per_forward"],
        "spec/mean_drafts_accepted": agg["mean_drafts_accepted"],
        "spec/alpha": agg.get("alpha", 0.0),
        "spec/speedup": spec_tps / max(ar_tps, 1e-8),
        "spec/spec_tok_s": spec_tps,
        "spec/ar_tok_s": ar_tps,
        "spec/lossless": n_match / max(1, n_eval),  # bf16 greedy-match fraction (see docstring)
    }
    for j in range(model.config.n_mtp):
        out[f"spec/accept_h{j + 1}"] = agg[f"accept_h{j + 1}"]
    return out
