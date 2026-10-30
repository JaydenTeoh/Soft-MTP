"""Periodic MTP + self-speculative eval for the HF `Trainer`.

A `TrainerCallback` that logs per-head verifier-agreement (`evaluate_mtp`) and the headline
SELF-SPECULATIVE metric. The self-spec metric is **speculative SAMPLING only** (accept a draft with
prob min(1, q/p), residual-resample on reject — `sample_speculative_sequential_cached`): it depends on
the full verifier distribution `q`, which is where the KL objective should win. Greedy (top-1)
speculative acceptance is NOT measured — it is blind to the distribution match. Requires a
KV-cache-capable trunk (`DynamicCache`) — the custom SDPA Block gained
an inference-only KV cache); `plain_greedy_sequential_cached` is only the wall-clock AR timing baseline.
Logged as `spec/*` (e.g. `spec/accept_h*`, `spec/alpha`,
`spec/tokens_per_forward`, `spec/speedup`, `spec/temperature`).

Runs on all ranks (DDP-sharded eval / collectives); only rank 0 logs.
"""

import torch
from torch.utils.data import DataLoader

from transformers import TrainerCallback

from ..data import BinDataset, mtp_collate
from .evaluation import evaluate_mtp, evaluate_self_spec, summarize_val_losses


def eval_batches_from_cfg(t, default=200):
    """Number of val batches for `evaluate_mtp` — `train.eval_batches`, with back-compat for the
    legacy key `eval_mtp_batches` (21 shipped configs still use it; kept readable so completed-run
    configs replay unchanged).

    200 x 12 x 1024 = 2.46M tokens matches nanoGPT, but FIXED (`shuffle=False`) rather than
    resampled — the same held-out sample at every eval and across arms, which is what makes small
    cross-arm differences readable. A smaller sample costs ~0.004 nats of absolute accuracy at 1.5B
    while moving the between-arm gap by only ~0.0004, since the error is common-mode.

    This governs the ENTIRE val/ panel, not just the MTP heads: `val/ppl`, `val/mtp_loss` and
    `val/mtp_kl_verifier` are all derived from this one call.
    """
    if "eval_batches" in t:
        return int(t["eval_batches"])
    return int(t.get("eval_mtp_batches", default))


class MTPEvalCallback(TrainerCallback):
    def __init__(self, val_bin, seq_len, n_future, eval_every, eval_batches=200,
                 spec_n_prompts=16, spec_prompt_len=256, spec_new_tokens=128, batch_size=12,
                 spec_every=None, spec_temperature=1.0):
        ds = BinDataset(val_bin, seq_len, n_future)
        self.loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            collate_fn=lambda b: mtp_collate(b, seq_len, n_future),
        )
        self.eval_every = int(eval_every)
        # Self-spec decoding (128 prompts x autoregressive generation) dominates eval wall-clock; run it
        # LESS often than the cheap mtp/grad/NTP metrics. spec_every must be a multiple of eval_every.
        self.spec_every = int(spec_every) if spec_every else int(eval_every)
        self.spec_temperature = float(spec_temperature)   # for the speculative-SAMPLING pass
        self.eval_batches = int(eval_batches)
        self.spec_kw = dict(n_prompts=int(spec_n_prompts), prompt_len=int(spec_prompt_len),
                            new_tokens=int(spec_new_tokens))

    def _run(self, model, rank=0, world=1, do_mtp=True, do_spec=True):
        device = "cuda" if next(model.parameters()).device.type == "cuda" else "cpu"
        if device == "cuda":
            torch.cuda.empty_cache()    # release the training step's cached allocator blocks (~74GB
                                        # reserved) so the eval forward allocates against a clean pool
        is_sequential = model.config.mtp_arch == "deepseek"
        free = (lambda: torch.cuda.empty_cache()) if device == "cuda" else (lambda: None)
        metrics = {}
        if do_mtp:
            metrics.update(evaluate_mtp(model, self.loader, device, self.eval_batches,
                                        rank=rank, world=world))
            free()   # drop the mtp-eval logit buffers before the (cached-decode) spec pass
        # Speculative SAMPLING ONLY (accept w.p. min(1, q/p), residual-resample on reject) — the
        # distribution-sensitive metric; greedy top-1 acceptance is blind to the verifier-distribution
        # match (where KL should win), so it is NOT measured. Needs the KV-cached trunk;
        # plain cached-greedy AR is only the wall-clock timing baseline. `spec/lossless` is N/A.
        # ARCH DISPATCH: DeepSeek -> sequential rollout drafter; META -> the
        # parallel-head BLOCK drafter (marginal proposals; lossless by the same accept/residual rule —
        # its LOWER acceptance vs a matched sequential measures that path noise).
        if do_spec and getattr(model.config, "model_type", "") == "hf_mtp":
            from ..decoding.sequential import sample_self_speculative_sequential_cached
            from ..decoding.meta import sample_self_speculative_meta_cached
            from ..decoding.plain import plain_greedy_sequential_cached
            spec_dec = (sample_self_speculative_sequential_cached if is_sequential
                        else sample_self_speculative_meta_cached)
            T = self.spec_temperature
            sm = evaluate_self_spec(
                model, self.loader, device,
                lambda mdl, pr, n: spec_dec(mdl, pr, n, temperature=T),
                plain_greedy_sequential_cached, rank=rank, world=world, **self.spec_kw)
            sm.pop("spec/lossless", None)
            sm["spec/temperature"] = T
            metrics.update(sm)
            free()   # drop the self-spec KV caches before the grad-diagnostic fwd+bwd (the eval mem peak)
        # Derive val/ppl, val/mtp_loss, val/mtp_kl_verifier and val/mtp_tv_verifier from
        # evaluate_mtp's output, which already names the NTP head val/ntp_loss.
        metrics.update(summarize_val_losses(metrics))
        return metrics

    def _eval_and_log(self, args, state, kwargs, do_mtp, do_spec, tag="eval"):
        model = kwargs["model"]                # peel torch.compile (_orig_mod) and DDP (module), any order
        for _ in range(3):
            if hasattr(model, "_orig_mod"):
                model = model._orig_mod
            elif hasattr(model, "module"):
                model = model.module
            else:
                break
        # ALL ranks run the eval: it is DDP-sharded and uses all_reduce/all_gather collectives
        # that every rank must join. Rank 0 receives the combined metrics and is the only logger.
        metrics = self._run(model, rank=args.process_index, world=args.world_size,
                            do_mtp=do_mtp, do_spec=do_spec)
        if not state.is_world_process_zero or not metrics:
            return
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=state.global_step)
        except Exception:
            pass
        print(f"[{tag} step {state.global_step}] "
              + " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()) if isinstance(v, float)),
              flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        # Anchor eval at the START of training: for a FRESH run (global_step 0) this is the init baseline
        # (val/ntp_loss ~ ln(vocab), the sanity anchor); for a RESUME (global_step = the resumed step) it
        # fills the blind spot before the first eval_every multiple. Curvature/self-spec are meaningless at
        # random init, so on a fresh run we log only the cheap do_mtp metrics; on resume the model
        # is loaded and the self-spec pass is meaningful too.
        resuming = state.global_step > 0
        self._eval_and_log(args, state, kwargs, do_mtp=True, do_spec=resuming, tag="eval-anchor")

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step == 0:
            return
        do_mtp = self.eval_every > 0 and step % self.eval_every == 0
        do_spec = self.spec_every > 0 and step % self.spec_every == 0
        if not (do_mtp or do_spec):      # step decisions are identical across ranks
            return
        self._eval_and_log(args, state, kwargs, do_mtp=do_mtp, do_spec=do_spec)
