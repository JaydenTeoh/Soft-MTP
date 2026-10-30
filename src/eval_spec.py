"""KV-cached self-speculative-decoding evaluation on trained checkpoints.

    python -m src.eval_spec --ckpts output/167m_fineweb_soft_mtp/checkpoint-60000 \
                            --val data/fineweb-edu/val.bin --n-prompts 128 --fp32

Loads each trained `HFMTPForCausalLM` checkpoint (its custom SDPA Block has an inference-only KV
cache) and times cached self-speculative SAMPLING against cached plain-greedy AR, over `--n-prompts`
prompts of `--prompt-len` tokens generating `--new-tokens` each. Reports per-head acceptance,
acceptance length, tokens/forward and wall-clock speedup -- one row per checkpoint and prompt set.

Defaults are 1024 prompt length / 512 generated tokens / 128 prompts:

`val/mtp_kl_verifier` from `evaluate_mtp` measures the same quantity (alpha = 1 - TV(p, q))
teacher-forced on real held-out text with no generation at all; report the two together.
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

from .decoding import aggregate_acceptance
from .decoding.sequential import sample_self_speculative_sequential_cached
from .decoding.meta import sample_self_speculative_meta_cached
from .decoding.plain import plain_greedy_sequential_cached   # AR timing baseline


def spec_decoder_for(model):
    """Arch dispatch: DeepSeek -> sequential rollout drafter; META -> parallel-head block drafter
    (marginal proposals; same lossless accept/residual rule, lower acceptance = §10 path noise)."""
    return (sample_self_speculative_sequential_cached if model.config.mtp_arch == "deepseek"
            else sample_self_speculative_meta_cached)


def load_prompts(val_bin, n, plen, newt, seed=0):
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    block = plen + newt
    nb = len(data) // block
    idx = np.random.default_rng(seed).choice(nb, size=min(n, nb), replace=False)
    return [torch.tensor(data[int(i) * block: int(i) * block + plen].astype(np.int64))[None] for i in idx]


_WARMED = False   # process-level: see SPEC_WARM_SCOPE in eval_model


def eval_model(model, prompts, new_tokens, device, temperature=1.0, bf16=True, ar_prompts=None):
    if ar_prompts is None:
        ar_prompts = int(os.environ.get('SPEC_AR_PROMPTS', len(prompts)))
    """Speculative SAMPLING acceptance + speedup (the distribution-sensitive metric)."""
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    ctx = torch.autocast(device, dtype=torch.bfloat16) if (bf16 and device == "cuda") else nullcontext()
    spec_dec = spec_decoder_for(model)
    stats, spec_s, ar_s, spec_s_paired = [], 0.0, 0.0, 0.0
    # inference_mode, not no_grad: self-spec decoding is LAUNCH-BOUND (hundreds of rounds, each a
    # handful of tiny ops plus per-draft .item() syncs), so the per-op bookkeeping no_grad still pays
    # -- version counters and view tracking -- is a measurable share of the step. inference_mode drops
    # both. Safe here: nothing produced under it is ever used in autograd; the decoders' own
    # @torch.no_grad() decorators nest fine inside it.
    with torch.inference_mode(), ctx:
        # WARM UP AT THE TIMED SHAPES. A short warm-up does NOT cover the one-time init -- lazy
        # kernel selection, autotuning, allocator growth -- that the first FULL-length generation
        # triggers. With one, all of it lands inside the first timed set, and that set alone reads
        # as a slowdown while acceptance is unchanged: a pure timing artifact. 8 prompts fully
        # warm, 4 do not.
        # SCOPE: the warm-up warms PROCESS state, and every prompt set here runs the same shapes,
        # so warming once per process is enough and warming per set is mostly redundant.
        #   SPEC_WARM_SCOPE=process  warm once per process (default)
        #   SPEC_WARM_SCOPE=set      warm before every set
        # Both arms of a comparison MUST use the same scope -- never change it mid-run.
        # The model is reloaded per checkpoint but the CUDA context is not, so `process` scope
        # stays valid across checkpoints too.
        global _WARMED
        scope = os.environ.get("SPEC_WARM_SCOPE", "process")
        if scope == "set" or not _WARMED:
            n_warm = min(int(os.environ.get("SPEC_WARM_PROMPTS", 8)), len(prompts))
            for wp in prompts[:n_warm]:
                wp = wp.to(device)
                spec_dec(model, wp, new_tokens, temperature=temperature)
                plain_greedy_sequential_cached(model, wp, new_tokens)
            _WARMED = True
        sync()
        for i, p in enumerate(prompts):
            p = p.to(device)
            sync(); t = time.perf_counter()
            _, s = spec_dec(model, p, new_tokens, temperature=temperature)
            sync(); dt_spec = time.perf_counter() - t; spec_s += dt_spec
            # AR baseline: only the first `ar_prompts` (SPEC_AR_PROMPTS, default all). It feeds
            # ONLY `speedup`, and costs roughly two thirds of the runtime, so subsampling it buys
            # back wall-clock without touching the acceptance metrics -- those come from exact
            # integer counters in the spec pass alone.
            if i < ar_prompts:
                sync(); t = time.perf_counter()
                plain_greedy_sequential_cached(model, p, new_tokens)   # timing baseline (forward cost == sampling AR)
                sync(); dt_ar = time.perf_counter() - t; ar_s += dt_ar
                spec_s_paired += dt_spec
            else:
                dt_ar = None
            # PER-PROMPT timings, not just running totals. Both intervals are already measured,
            # and storing them makes per-prompt speedup, acceptance length and tok/forward
            # derivable offline -- which is how you check whether the prompt count is large enough.
            s["spec_s"], s["ar_s"] = dt_spec, dt_ar
            stats.append(s)
    agg = aggregate_acceptance(stats, model.config.n_mtp)
    n_ar = min(ar_prompts, len(prompts))
    tot = len(prompts) * new_tokens
    spec_tps = tot / max(spec_s, 1e-9)
    ar_tps = (n_ar * new_tokens) / max(ar_s, 1e-9)
    # paired: both numerator and denominator over the SAME first n_ar prompts
    agg.update(speedup=(ar_s / max(spec_s_paired, 1e-9)) if n_ar else float("nan"),
               spec_tok_s=spec_tps, ar_tok_s=ar_tps, n_ar_prompts=n_ar)
    # RAW is what makes sharding exact: aggregate_acceptance sums proposed/accepted/n_forward/
    # n_generated and concatenates round_accepts, so a union over shards == one unsharded run.
    # Timings sum too (each shard owns a GPU, so no contention distorts the spec-vs-AR ratio).
    raw = {"stats": stats, "spec_s": spec_s, "ar_s": ar_s, "n_ar_prompts": n_ar,
           "n_prompts": len(prompts), "new_tokens": new_tokens, "n_mtp": model.config.n_mtp}
    return agg, raw


def load_domain_prompts(spec, limit=None):
    """'name=path.npy' -> (name, [1,plen] tensor list). The .npy is a uint16 [n, plen] array of
    token ids -- build your own from any corpus. It must be a FIXED file, identical for every
    arm/checkpoint, or the comparison is not paired.

    `limit` takes the FIRST n rows, so the subset is deterministic and identical across arms, shards
    and reruns (a random subsample would silently break paired comparison). Previously --n-prompts
    applied only to the fineweb set while the fixtures were always consumed whole, so asking for
    fewer prompts cut the cost of one set in five."""
    name, path = spec.split("=", 1)
    arr = np.load(path)
    assert arr.ndim == 2, f"{path}: expected [n, plen], got {arr.shape}"
    if limit is not None:
        arr = arr[:limit]
    return name, [torch.tensor(row.astype(np.int64))[None] for row in arr]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--val", default=None, help="FineWeb val.bin (evaluated as domain 'fineweb')")
    p.add_argument("--n-prompts", type=int, default=128)
    p.add_argument("--prompt-len", type=int, default=1024)
    p.add_argument("--new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)   # speculative SAMPLING temperature
    p.add_argument("--fp32", action="store_true",
                   help="disable bf16 autocast (weights are fp32 master weights either way). "
                        "Acceptance should be ~unchanged if bf16 noise is common-mode between "
                        "draft p and target q; speedup should RISE, since fp32 doubles weight "
                        "traffic and spec decoding wins by being less memory-bound than AR.")
    p.add_argument("--domain-npy", nargs="+", default=None, metavar="name=path.npy",
                   help="fixed domain prompt sets: a uint16 [n, prompt_len] .npy of token ids per "
                        "name. Result keys become '<ckpt>::<name>'; without this flag they stay "
                        "plain '<ckpt>'.")
    p.add_argument("--shard", default=None, metavar="i/N",
                   help="evaluate only prompts[i::N] (per domain). load_prompts is seeded (seed=0) and "
                        "the .npy sets are fixed files, so every shard draws the SAME prompts and the "
                        "slices are disjoint. Pool the --out-json files by SUMMING the integer "
                        "counters and concatenating round_accepts -- do NOT read a shard's own "
                        "table as the result.")
    p.add_argument("--out-json", default=None, help="write raw per-prompt stats here (required for sharding)")
    a = p.parse_args()
    from .models.model import HFMTPForCausalLM

    if not a.val and not a.domain_npy:
        p.error("need --val and/or --domain-npy")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    domains = []                                # [(name, prompts)]; name=None = legacy single-set mode
    if a.domain_npy:
        if a.val:
            domains.append(("fineweb", load_prompts(a.val, a.n_prompts, a.prompt_len, a.new_tokens)))
        domains += [load_domain_prompts(s, limit=a.n_prompts) for s in a.domain_npy]
    else:
        domains.append((None, load_prompts(a.val, a.n_prompts, a.prompt_len, a.new_tokens)))
    shard_tag = ""
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        assert 0 <= i < n, f"--shard i/N needs 0<=i<N, got {a.shard}"
        domains = [(name, pr[i::n]) for name, pr in domains]
        shard_tag = f" [shard {i}/{n}]"
    n_tot = sum(len(pr) for _, pr in domains)
    print(f"# self-spec SAMPLING eval (T={a.temperature}): {n_tot} prompts over {len(domains)} set(s)"
          f"{shard_tag}, new {a.new_tokens}, device {device}\n")
    out = {}
    hdr = (f"{'ckpt':38s} {'domain':>8s} {'tok/fwd':>8s} {'accept_h*':>16s} {'alpha':>6s} "
           f"{'speedup':>8s} {'tok/s':>8s}")
    print(hdr); print("-" * len(hdr))
    for c in a.ckpts:
        model = HFMTPForCausalLM.from_pretrained(c).to(device).eval()
        for name, prompts in domains:
            r, raw = eval_model(model, prompts, a.new_tokens, device, temperature=a.temperature,
                                bf16=not a.fp32)
            out[c if name is None else f"{c}::{name}"] = raw
            accs = " ".join(f"{r.get(f'accept_h{i+1}', 0):.2f}" for i in range(model.config.n_mtp))
            print(f"{c[-38:]:38s} {(name or 'val'):>8s} {r['tokens_per_forward']:8.3f} {accs:>16s} "
                  f"{r.get('alpha',0):6.3f} {r['speedup']:7.2f}x {r['spec_tok_s']:8.0f}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    if a.out_json:
        import json, os
        os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
        json.dump(out, open(a.out_json, "w"))
        print(f"\nwrote raw stats -> {a.out_json}")


if __name__ == "__main__":
    main()
