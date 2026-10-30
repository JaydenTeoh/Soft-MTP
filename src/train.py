"""HF `Trainer` entrypoint (single combined backward).

    torchrun --standalone --nproc_per_node=N -m src.train \
      --config configs/167m_fineweb.yaml --objective soft_mtp

`grad_accum` is derived here from `train.global_batch_seqs` and the world size, so the token budget
per step holds whatever GPU count you launch on. Two arms at different token budgets are not a
comparison, and only this process knows how many GPUs it got.

The model returns its combined loss (model-returns-loss pattern, like aynetdia/mtp_curriculum),
so the stock `Trainer` drives it with one `.backward()` per step -> DDP-native. We set
`model_accepts_loss_kwargs=False` because the loss is a summed multi-head loss we normalize
ourselves (Trainer then applies the correct `/gradient_accumulation_steps`).
"""

import argparse
import os
import time
from pathlib import Path

import torch
import yaml
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed


class CheckpointRotationCallback(TrainerCallback):
    """Disk-bounded resumable checkpointing. FORCE-saves at every step in `must_steps` (kept FOREVER —
    e.g. the WSD end-of-stable step right before the LR decay, for branch-from-stable), and keeps only the
    `keep_latest` most-recent ROLLING checkpoints (older rolling ones deleted, protected ones never). With
    save_only_model=False these are FULL checkpoints (optimizer+scheduler+RNG) -> resumable, so disk stays
    at ~ (len(must_steps)+keep_latest) full checkpoints instead of ALL of them. Set save_total_limit=None
    so HF's own rotation doesn't fight this. Rank-0 owns deletion; the force-save is set on all ranks."""

    def __init__(self, must_steps=(), keep_latest=1, full_steps=None, rolling_full_every=0):
        self.must = set(int(s) for s in must_steps)
        self.keep_latest = max(1, int(keep_latest))
        # `full_steps`: steps saved with FULL state (optimizer/scheduler/RNG -> resumable, ~5GB @350M) and kept
        # FOREVER. When set (or rolling_full_every>0), EVERY OTHER checkpoint is model-only (~1.8GB) — so we can
        # keep MANY curvature time-points cheaply (weights are all the offline per-layer Hessian diagnostic needs)
        # while the branch/crash-resume anchors stay resumable. When both empty, train.save_only_model applies.
        self.full = set(int(s) for s in (full_steps or ()))
        # `rolling_full_every`: keep ONE rolling FULL checkpoint (optimizer/scheduler/RNG) at the latest step that
        # is a multiple of this cadence, for crash-resume WITHOUT paying full-size disk at every kept step. When a
        # newer rolling-full is written, the previous one is DEMOTED to model-only (optimizer stripped) if it is a
        # protected must-step, else deleted by rotation. Permanent `full_steps` are never demoted.
        self.rolling_full_every = int(rolling_full_every)

    def _is_rolling(self, step):
        return self.rolling_full_every > 0 and step % self.rolling_full_every == 0 and step not in self.full

    @staticmethod
    def _strip_optimizer(path):
        """Demote a FULL checkpoint to model-only: remove optimizer/scheduler/RNG/scaler, keep the weights."""
        import glob
        removed = False
        for pat in ("optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state*.pth"):
            for f in glob.glob(os.path.join(path, pat)):
                try:
                    os.remove(f); removed = True
                except OSError:
                    pass
        return removed

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step in self.must or self._is_rolling(step):   # force a checkpoint at protected / rolling-full steps
            control.should_save = True
        if control.should_save and (self.full or self.rolling_full_every):
            # FULL only at permanent full_steps and at the CURRENT rolling-full step; every other save is model-only.
            args.save_only_model = not (step in self.full or self._is_rolling(step))
        return control

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:       # only rank 0 deletes / demotes
            return control
        import re
        import shutil
        ckpts = []
        for name in os.listdir(args.output_dir):
            m = re.fullmatch(r"checkpoint-(\d+)", name)
            if m and os.path.isdir(os.path.join(args.output_dir, name)):
                ckpts.append((int(m.group(1)), os.path.join(args.output_dir, name)))
        ckpts.sort()
        keep = {p for _, p in ckpts[-self.keep_latest:]}          # latest rolling checkpoint(s)
        keep |= {p for s, p in ckpts if s in self.must}           # protected must-steps (never deleted)
        keep |= {p for s, p in ckpts if s in self.full}           # permanent full checkpoints (never deleted)
        # the current rolling-full = the newest kept checkpoint whose step is a rolling multiple (non-permanent)
        rolling = [p for s, p in ckpts if self._is_rolling(s) and p in keep]
        latest_rolling = rolling[-1] if rolling else None
        for s, p in ckpts:
            if p not in keep:
                shutil.rmtree(p, ignore_errors=True)
                print(f"[ckpt-rotate] removed {os.path.basename(p)}", flush=True)
            elif self._is_rolling(s) and p != latest_rolling:     # protected but stale rolling-full -> demote
                if self._strip_optimizer(p):
                    print(f"[ckpt-rotate] demoted {os.path.basename(p)} to model-only", flush=True)
        return control


def _must_steps(t):
    """Config `train.save_must_steps`: an explicit list of step NUMBERS (e.g. [54000]) whose checkpoints
    are ALWAYS kept (protected from rolling-checkpoint rotation). Explicit numbers -> list as many as you
    want (e.g. the whole cooldown [54000, 57000, 60000]) without needing a keyword per step."""
    raw = t.get("save_must_steps", [])
    if isinstance(raw, int):
        raw = [raw]
    return sorted({int(s) for s in raw})


def _full_steps(t):
    """Config `train.save_full_steps`: steps saved with FULL state (optimizer/scheduler/RNG). When set, all
    OTHER checkpoints are model-only -> keep many cheap weight-only curvature checkpoints + resumable anchors."""
    raw = t.get("save_full_steps", [])
    if isinstance(raw, int):
        raw = [raw]
    return sorted({int(s) for s in raw})


class StartStepLogCallback(TrainerCallback):
    """Force a train-metric log on the FIRST optimizer step of the run — fresh -> step 1, RESUME -> the resume
    step — so the loss trajectory has a point at the very start / branch point, not only at the first
    `logging_steps` multiple (up to log_every-1 steps late; e.g. a cooldown-only resume from 18000 otherwise
    first logs at 18020). One extra Trainer.log at the first step; `logging_first_step` covers only fresh runs
    (global_step==1), so this also anchors resumes at their branch point."""

    def __init__(self):
        self._done = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self._done:
            control.should_log = True
            self._done = True
        return control


class PerfClockCallback(TrainerCallback):
    """Start the throughput clock at the step training actually begins from. `MTPTrainer.__init__`
    sets `_step_last = 0`, so on a RESUME the first log would divide the whole startup interval
    (compile + dataloader skip + the anchor eval) by the resumed step count — reading out as an
    impossible MFU. A no-op on a fresh run, where that step is 0."""

    def __init__(self, trainer):
        self._trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self._trainer._t_last = time.perf_counter()
        self._trainer._step_last = state.global_step
        return control


class YamlConfigToWandbCallback(TrainerCallback):
    """Mirror the resolved YAML into the W&B run config as `cfg/<section>/<key>`.

    WHY: W&B only ever receives HF `TrainingArguments`, so none of the knobs that actually define
    a run are visible on the run page — `objective`, `mtp_weight`, `eval_batches`, `save_must_steps`,
    `decay_fraction` and the rest. Worse than absent: the panel shows HF's unused
    `per_device_eval_batch_size` while `eval_strategy` is `no` (evaluation runs from
    MTPEvalCallback, not HF's eval loop), so reading it gives the WRONG eval batch size. Mirroring
    the YAML makes a run self-documenting and cross-run diffable.

    Runs on rank 0 at train begin, AFTER HF's WandbCallback has created `wandb.run` (callbacks fire in
    registration order and HF's is registered at Trainer construction). Values are JSON-scalars or short
    reprs; `allow_val_change` keeps a resume from erroring on an already-set key. Purely metadata: it
    logs nothing to the step stream and cannot affect training or eval numerics.
    """

    def __init__(self, cfg):
        self._cfg = cfg

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        try:
            import wandb
            if wandb.run is None:
                return
            flat = {}
            for section, body in (self._cfg or {}).items():
                if isinstance(body, dict):
                    for k, v in body.items():
                        flat[f"cfg/{section}/{k}"] = v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)
                else:
                    flat[f"cfg/{section}"] = body if isinstance(body, (int, float, str, bool, type(None))) else repr(body)
            wandb.config.update(flat, allow_val_change=True)
            print(f"[train] mirrored {len(flat)} yaml keys into the W&B run config", flush=True)
        except Exception as e:                       # never let metadata logging break a run
            print(f"[train] wandb yaml-config mirror skipped: {e}", flush=True)


from .eval.efficiency import count_params, flops_per_token
from .models.config import OBJECTIVES, HFMTPConfig
from .models.model import HFMTPForCausalLM, MTPCausalLMOutput


class MTPTrainer(Trainer):
    def __init__(self, *args, seq_len=1024, flops_per_tok=0.0, gpu_peak_flops=0.0, wsd=None,
                 muon=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Our model returns a self-normalized summed multi-head loss (it does NOT consume
        # num_items_in_batch); opt out of the 4.46 auto loss-kwargs path so grad-accum scaling
        # stays correct (HF docs).
        self.model_accepts_loss_kwargs = False
        self._wsd = wsd                    # dict(min_lr_ratio, decay_fraction, decay_shape) or None (cosine)
        self._muon = muon                  # dict(momentum, nesterov, ns_steps, adjust_lr_fn) or None (=AdamW)
        self._brk_sum = None   # {field: running-sum tensor} of the model's named loss breakdown
        self._brk_n = 0
        self._seq_len = seq_len
        self._fpt = flops_per_tok          # FLOPs/token for MFU
        self._gpu_peak_flops = gpu_peak_flops
        self._t_last = time.perf_counter()
        self._step_last = 0

    def create_optimizer(self):
        """AdamW (HF default) unless `muon` is set, in which case build Muon (2D hidden matrices) +
        AdamW (embed/lm_head/norms) as one MuonWrapper. Shares lr/wd/betas from TrainingArguments;
        `match_rms_adamw` lets Muon reuse the AdamW-tuned lr. The WSD scheduler (create_scheduler) then
        scales BOTH sub-optimizers' groups via the concatenated param_groups."""
        if self._muon is None:
            opt = super().create_optimizer()
            if self.args.process_index == 0:
                print(f"[optim] AdamW (HF default) | lr={self.args.learning_rate:g} "
                      f"wd={self.args.weight_decay}", flush=True)
            return opt
        if self.optimizer is None:
            from .training.optim import build_muon_adamw
            self.optimizer = build_muon_adamw(
                self.model, lr=self.args.learning_rate, weight_decay=self.args.weight_decay,
                betas=(self.args.adam_beta1, self.args.adam_beta2), adam_eps=self.args.adam_epsilon,
                **self._muon)
            # SELF-DOCUMENT the optimizer. Without this a Muon run and an AdamW run produce
            # byte-identical logs, so a config edit that dropped `optim.optimizer: muon` would be
            # invisible in the artifact -- you would only find out by reading the yaml months later.
            if self.args.process_index == 0:
                mu, aw = self.optimizer.optimizers
                nm = sum(p.numel() for g in mu.param_groups for p in g["params"])
                na = sum(p.numel() for g in aw.param_groups for p in g["params"])
                d = mu.param_groups[0]
                print(f"[optim] MUON on {sum(len(g['params']) for g in mu.param_groups)} 2D matrices "
                      f"({nm/1e6:.1f}M params) + AdamW on {na/1e6:.1f}M (embed/lm_head/norms) | "
                      f"lr={self.args.learning_rate:g} wd={d['weight_decay']} momentum={d['momentum']} "
                      f"nesterov={d['nesterov']} ns_steps={d['ns_steps']} "
                      f"adjust_lr_fn={d['adjust_lr_fn']}", flush=True)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        """WSD: build a LambdaLR from the shared `lr_lambda_factory` (HF has no 1-sqrt cooldown). For
        any other schedule, defer to HF's built-in (`lr_scheduler_type`, e.g. cosine_with_min_lr)."""
        if self._wsd is None:
            return super().create_scheduler(num_training_steps, optimizer)
        if self.lr_scheduler is None:
            from torch.optim.lr_scheduler import LambdaLR
            from .training.optim import lr_lambda_factory
            opt = self.optimizer if optimizer is None else optimizer
            fn = lr_lambda_factory(self.args.get_warmup_steps(num_training_steps), num_training_steps,
                                   schedule="wsd", min_lr_ratio=self._wsd["min_lr_ratio"],
                                   decay_fraction=self._wsd["decay_fraction"],
                                   decay_shape=self._wsd["decay_shape"])
            self.lr_scheduler = LambdaLR(opt, fn)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Same loss as the model returns, but also accumulate the NAMED detached breakdown fields
        from `MTPCausalLMOutput` (`breakdown_fields()` — derived from the dataclass) so `log()` can
        emit them per step (FLA-stack parity). Tensors are summed on-device; the single CPU sync
        happens once per log_every in `log()`."""
        outputs = model(**inputs)
        loss = outputs["loss"]
        vals = {k: v for k in MTPCausalLMOutput.breakdown_fields()
                if (v := getattr(outputs, k, None)) is not None}
        if vals:
            if self._brk_sum is None or self._brk_sum.keys() != vals.keys():
                self._brk_sum = {k: torch.zeros_like(v) for k, v in vals.items()}
                self._brk_n = 0
            for k, v in vals.items():
                self._brk_sum[k] += v.detach()
            self._brk_n += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        # Per-step train-loss breakdown (FLA parity). Bare names (no "train/" prefix): HF
        # WandbCallback.rewrite_logs prepends "train/" itself, so "train/*" here would double-prefix.
        # Values are logged at FULL precision: W&B stores raw floats (its UI rounds for display),
        # and the late-phase KL column decays toward ~1e-3 where pre-rounding would cost real
        # significant figures. (HF rounds its own 'loss' key to 4dp; that's theirs.)
        if self._brk_n > 0:
            for field, s in self._brk_sum.items():
                avg = s / self._brk_n
                if avg.dim() == 0:                       # scalar field (ntp_loss)
                    logs[field] = avg.item()
                else:                                    # per-head vector (mtp_loss / kl_loss):
                    logs[field] = avg.mean().item()                    # mean under the field name
                    for j, v in enumerate(avg.tolist(), start=1):      # + <x>_h{j}_loss per head
                        logs[field.replace("_loss", f"_h{j}_loss")] = v
            self._brk_sum, self._brk_n = None, 0
        # perf: throughput / step time / MFU / peak mem over the interval since the last train log
        if "loss" in logs:
            now = time.perf_counter()
            steps = self.state.global_step - self._step_last
            dt = now - self._t_last
            if steps > 0 and dt > 0:
                world = max(1, self.args.world_size)
                tok_per_step = (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
                                * world * self._seq_len)
                tps = steps * tok_per_step / dt                 # global tokens/sec
                logs["perf/step_time_s"] = round(dt / steps, 4)
                logs["perf/tokens_per_sec"] = round(tps, 1)
                if self._fpt and self._gpu_peak_flops:
                    logs["perf/mfu"] = round(100.0 * self._fpt * (tps / world) / self._gpu_peak_flops, 2)
                if torch.cuda.is_available():
                    logs["perf/peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                    torch.cuda.reset_peak_memory_stats()        # track the next interval, not the all-time max
            self._t_last, self._step_last = now, self.state.global_step
        super().log(logs, *args, **kwargs)


def _build_model(model_cfg):
    """Build `HFMTPForCausalLM` from the config's `model:` section."""
    m = dict(model_cfg)
    init_from = m.pop("init_from", None)
    model = HFMTPForCausalLM(HFMTPConfig(**m))
    c = model.config
    if c.mtp_arch == "deepseek" and c.n_mtp:
        # Self-document the sequential layout so a log identifies the architecture it came from.
        mtp_p = sum(p.numel() for n, p in model.named_parameters()
                    if n.startswith(("proj.", "norms_h.", "norms_e.", "mtp_blocks.", "mtp_final_norms.")))
        print(f"[mtp] deepseek sequential n_mtp={c.n_mtp} x {c.mtp_head_layers} block(s)"
              f" | per_head_norm={c.mtp_per_head_norm} | mtp params={mtp_p / 1e6:.1f}M", flush=True)
    if init_from:
        _init_from_checkpoint(model, init_from)
    return model


def _init_from_checkpoint(model, path):
    """WEIGHTS-ONLY init from a pretrained checkpoint — NOT a resume.

    `resume_from_checkpoint` restores optimizer moments, the LR schedule and the global step, which
    is what you want after a crash and exactly what you do NOT want for continued training on a new
    corpus: that needs a FRESH optimizer and a fresh cosine schedule over the new step budget.

    Loads with strict=False so a pretrained MTP checkpoint can initialise an `n_mtp: 0` model — the
    MTP modules (mtp_blocks / mtp_final_norms / norms_e / norms_h / proj) are simply dropped, which
    IS the intended "discard the MTP objective" semantics and matches deployment (DeepSeek-V3
    discards the heads at inference). The NTP path prefixes (embed / trunk / norm / lm_head) are
    disjoint from those, so the split is exact.

    ⚠ MISSING KEYS ARE FATAL. strict=False silently leaves anything unmatched at its RANDOM init, so
    a renamed prefix or a shape change would hand back a partly-untrained model that trains without
    complaint and quietly produces garbage results. Unexpected keys are fine (that is the discard);
    missing ones never are.
    """
    import glob as _glob, json as _json, os as _os
    from safetensors.torch import load_file as _load_st

    if _os.path.isdir(path):
        idx = _os.path.join(path, "model.safetensors.index.json")
        if _os.path.exists(idx):                      # sharded (1.5B+)
            shards = sorted({v for v in _json.load(open(idx))["weight_map"].values()})
            sd = {}
            for s in shards:
                sd.update(_load_st(_os.path.join(path, s)))
        else:
            hits = _glob.glob(_os.path.join(path, "model*.safetensors"))
            if not hits:
                raise FileNotFoundError(f"init_from: no safetensors under {path}")
            sd = {}
            for h in sorted(hits):
                sd.update(_load_st(h))
    else:
        sd = _load_st(path)

    res = model.load_state_dict(sd, strict=False)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    if missing:
        raise RuntimeError(
            f"init_from {path}: {len(missing)} parameter(s) NOT found in the checkpoint and would "
            f"stay randomly initialised — refusing. First few: {missing[:8]}")
    n_drop = sum(sd[k].numel() for k in unexpected)
    print(f"[init_from] loaded {path}: {len(sd) - len(unexpected)} tensors into the model; "
          f"dropped {len(unexpected)} checkpoint tensors ({n_drop/1e6:.1f}M params) — "
          f"{sorted({k.split('.')[0] for k in unexpected})}", flush=True)
    return model


MTP_MODULE_PREFIXES = ("proj.", "norms_h.", "norms_e.", "mtp_blocks.", "mtp_final_norms.", "mtp_heads.")


def freeze_backbone(model):
    """Train ONLY the MTP module(s) (FastMTP / Nemotron-style MTP boosting): embed, trunk, final
    norm and lm_head get requires_grad=False. Both optimizer paths (HF AdamW, MuonWrapper) and DDP skip
    frozen parameters, so nothing else needs to know. The trunk then builds no autograd graph at all,
    which is where the memory saving comes from. Returns the trainable parameter count."""
    n_train = n_frozen = 0
    for name, p in model.named_parameters():
        keep = name.startswith(MTP_MODULE_PREFIXES)
        p.requires_grad_(keep)
        if keep:
            n_train += p.numel()
        else:
            n_frozen += p.numel()
    if n_train == 0:
        raise ValueError("freeze_backbone: no MTP module parameters to train (n_mtp == 0?)")
    print(f"[freeze] backbone FROZEN ({n_frozen / 1e6:.1f}M params) | training only the MTP module(s): "
          f"{n_train / 1e6:.2f}M params ({100 * n_train / (n_train + n_frozen):.2f}% of the model)",
          flush=True)
    return n_train


def resolve_warmup_steps(t):
    """Warmup length from `train.warmup_steps` OR `train.warmup_ratio` (fraction of total_steps,
    rounded), never both. The ratio form follows a --total-steps override automatically."""
    if t.get("warmup_steps") is not None and t.get("warmup_ratio") is not None:
        raise ValueError("set train.warmup_steps OR train.warmup_ratio, not both")
    if t.get("warmup_ratio") is not None:
        r = float(t["warmup_ratio"])
        if not 0.0 <= r < 1.0:
            raise ValueError(f"train.warmup_ratio {r} must be in [0, 1)")
        return int(round(r * int(t["total_steps"])))
    return int(t.get("warmup_steps", 0))


def resolve_lr_floor(t, opt):
    """`(peak_lr, min_lr_ratio, min_lr)` from a config's `train`/`optim` dicts.

    Accepts EITHER spelling of the LR floor, never both (see build_trainer for why HF is always
    handed the ratio). Legacy configs that set neither keep the historical 0.1x default."""
    peak_lr = float(opt.get("lr", 6e-4))
    if t.get("min_lr") is not None and t.get("min_lr_ratio") is not None:
        raise ValueError("set train.min_lr OR train.min_lr_ratio, not both "
                         f"(got min_lr={t['min_lr']}, min_lr_ratio={t['min_lr_ratio']})")
    if t.get("min_lr") is not None:
        min_lr = float(t["min_lr"])
        if not 0.0 <= min_lr <= peak_lr:
            raise ValueError(f"train.min_lr {min_lr:g} must be in [0, peak lr {peak_lr:g}]")
        return peak_lr, min_lr / peak_lr, min_lr
    min_lr_ratio = float(t.get("min_lr_ratio", 0.1))
    return peak_lr, min_lr_ratio, min_lr_ratio * peak_lr


def build_trainer(cfg, overrides=None):
    # ⚠ SEED BEFORE THE MODEL IS BUILT. HF calls set_seed() inside Trainer.__init__, which runs
    # AFTER _build_model() — without this line initialisation is entropy-seeded, so two runs at the
    # same seed start from DIFFERENT weights. Check: two objectives at one seed must report an
    # identical loss at the step-0 eval anchor.
    set_seed(int(cfg.get("seed", 0)))
    model = _build_model(cfg["model"])
    if cfg["train"].get("freeze_backbone", False):
        freeze_backbone(model)
    # compile_mode: "whole" (HF torch_compile -> torch.compile(model); fuses the loss, benchmarked ~2x
    # throughput) | "none". Whole-model compile keeps EVAL cheap because the sequential self-spec /
    # evaluate_mtp call trunk_forward/dsmtp_layer/logits DIRECTLY -> those dispatch to the original
    # eager methods on the compiled module (no variable-shape recompiles).
    compile_mode = cfg["train"].get("compile_mode", "none")
    seq_len, n_future = cfg["data"]["seq_len"], model.n_heads_total
    # Data is provided by a registered SOURCE (default "bin" = the pretraining memmap path); the source
    # returns the train dataset, collator, and any eval callbacks — so train.py needs no per-task code.
    from .data.sources import get_data_source
    bundle = get_data_source(cfg["data"].get("source", "fineweb_edu"))(cfg, model, seq_len, n_future)
    train_ds, collate = bundle.train_dataset, bundle.data_collator

    t, opt = cfg["train"], cfg.get("optim", {})
    betas = opt.get("betas", [0.9, 0.95])
    # lr_schedule: "cosine" (HF cosine_with_min_lr) | "wsd" (our LambdaLR; 1-sqrt cooldown) | "constant"
    # (HF constant_with_warmup) | any HF `lr_scheduler_type`. wsd is handled by `create_scheduler`.
    lr_schedule = t.get("lr_schedule", "cosine")
    # HF `lr_scheduler_type` must be a valid SchedulerType at TrainingArguments construction. For "wsd"
    # the real schedule is built later in `create_scheduler` (our LambdaLR), so pass a valid PLACEHOLDER
    # here (constant_with_warmup) — it is never used because create_scheduler short-circuits when wsd is set.
    _HF_SCHED = {"cosine": "cosine_with_min_lr", "constant": "constant_with_warmup",
                 "wsd": "constant_with_warmup"}
    # LR floor: `train.min_lr` (absolute, preferred) or `train.min_lr_ratio` (fraction of peak).
    # ⚠ We hand HF the RATIO even for an absolute floor. HF's own absolute path divides by
    # `optimizer.defaults["lr"]`, which under the Muon wrapper is the MUON sub-optimizer's lr and can
    # move off the peak -> a silently wrong floor. Dividing by the config peak has no such dependency.
    peak_lr, min_lr_ratio, min_lr = resolve_lr_floor(t, opt)
    wsd = None
    if lr_schedule == "wsd":
        wsd = dict(min_lr_ratio=min_lr_ratio,
                   decay_fraction=t.get("decay_fraction", 0.2),
                   decay_shape=t.get("decay_shape", "1-sqrt"))
    # optim.optimizer: "adamw" (default; HF built-in) | "muon" (Muon on 2D hidden matrices + AdamW on
    # embed/lm_head/norms). Opt-in -> existing configs are byte-unchanged.
    muon = None
    if opt.get("optimizer", "adamw").lower() == "muon":
        muon = dict(momentum=opt.get("muon_momentum", 0.95),
                    nesterov=opt.get("muon_nesterov", True),
                    ns_steps=opt.get("muon_ns_steps", 5),
                    adjust_lr_fn=opt.get("muon_adjust_lr_fn", "match_rms_adamw"))
    wb = cfg.get("wandb", {})
    out_dir = str(Path(t.get("out_dir", "output")) / wb.get("name", "hf_run"))
    args_kw = dict(
        output_dir=out_dir,
        per_device_train_batch_size=t["micro_batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        max_steps=t["total_steps"],                          # required for map dataset + scheduler
        warmup_steps=resolve_warmup_steps(t),
        learning_rate=float(opt.get("lr", 6e-4)),
        weight_decay=opt.get("weight_decay", 0.1),
        adam_beta1=betas[0],
        adam_beta2=betas[1],
        lr_scheduler_type=_HF_SCHED.get(lr_schedule, lr_schedule),
        # Floor resolved above (train.min_lr | train.min_lr_ratio) -> HF wants it as `min_lr_rate`.
        # cosine-only kwarg; constant/other HF schedules reject it.
        lr_scheduler_kwargs=({"min_lr_rate": min_lr_ratio} if lr_schedule == "cosine" else {}),
        max_grad_norm=t.get("grad_clip", 1.0),
        logging_steps=t.get("log_every", 20),
        save_steps=t.get("save_every", 1000),
        save_strategy="steps",
        save_total_limit=None,                               # rotation handled by CheckpointRotationCallback
        save_only_model=t.get("save_only_model", False),     # FULL ckpt (optimizer/scheduler/RNG) -> RESUMABLE
        bf16=t.get("bf16", True),
        ddp_find_unused_parameters=False,                    # all params used for MTP objectives
        torch_compile=(compile_mode == "whole"),             # HF compiles the whole model when "whole"

        report_to=("wandb" if wb.get("enabled") else "none"),
        run_name=wb.get("name"),
        seed=cfg.get("seed", 0),
        dataloader_num_workers=t.get("num_workers", 2),
        dataloader_pin_memory=True,
        # ALWAYS fast-forward the dataloader to the resumed global_step on resume, so the run consumes the
        # SAME continuation data the source stream would serve next (faithful continuation / controlled A/B).
        # ignore_data_skip=True would restart the shuffle from the top -> re-expose already-seen data AND
        # diverge from the source run; no valid use of that in this project, so it is hardcoded off (the
        # config knob was removed). Cost = an O(step) dataloader skip on resume; worth it.
        ignore_data_skip=False,
    )
    args_kw.update(overrides or {})
    m = cfg["model"]
    n_layers_total = m["num_hidden_layers"] + m.get("n_mtp", 0) * m.get("mtp_head_layers", 1)
    fpt = flops_per_token(count_params(model), n_layers_total, m["hidden_size"], seq_len)
    trainer = MTPTrainer(model=model, args=TrainingArguments(**args_kw),
                         train_dataset=train_ds, data_collator=collate,
                         seq_len=seq_len, flops_per_tok=fpt,
                         gpu_peak_flops=float(cfg.get("gpu_peak_flops", 9.89e14)), wsd=wsd, muon=muon)
    # Periodic eval -> W&B (rank 0 only). The DATA SOURCE decides what eval to run (the pretraining "bin"
    # source returns MTPEvalCallback; a task source returns its own), so train.py stays task-agnostic.
    for cb in bundle.eval_callbacks:
        trainer.add_callback(cb)
    # Disk-bounded resumable checkpointing: protect the must-save steps (e.g. pre-decay branch point),
    # keep only the latest rolling checkpoint(s). keep_latest falls back to the old save_total_limit.
    trainer.add_callback(CheckpointRotationCallback(
        must_steps=_must_steps(t),
        keep_latest=t.get("save_keep_latest", t.get("save_total_limit", 1)),
        full_steps=_full_steps(t),
        rolling_full_every=t.get("save_rolling_full_every", 0)))
    trainer.add_callback(PerfClockCallback(trainer))   # perf metrics measure real elapsed steps on a resume
    trainer.add_callback(StartStepLogCallback())         # a train-metric point at step 1 / the resume branch step
    trainer.add_callback(YamlConfigToWandbCallback(cfg))  # make the run self-documenting (see the class)
    return trainer


def train(cfg):
    wb = cfg.get("wandb", {})
    if wb.get("enabled") and wb.get("project"):
        os.environ["WANDB_PROJECT"] = wb["project"]   # else HF defaults the project to "huggingface"
    trainer = build_trainer(cfg)
    # Resume: config train.resume_from_checkpoint = <path> | "auto"/"latest"/true (auto-find latest in
    # output_dir). Native HF resume restores model + optimizer + scheduler(LR) + RNG + global_step and
    # ROLLS THE DATA LOADER FORWARD (skips the already-seen batches) — bit-exact ONLY if the checkpoint has
    # optimizer.pt/scheduler.pt (i.e. was saved with save_only_model=False).
    resume = cfg["train"].get("resume_from_checkpoint", None)
    rk = True if resume in (True, "auto", "latest") else (str(resume) if resume else None)
    if isinstance(rk, str) and not os.path.isfile(os.path.join(rk, "optimizer.pt")):
        print(f"[resume] WARNING: '{rk}' has no optimizer.pt/scheduler.pt (a save_only_model=True / model-"
              f"only checkpoint) -> WARM-START only: fresh Adam moments and the LR schedule may restart. "
              f"This will NOT reproduce the original trajectory. Save with save_only_model=False to resume "
              f"exactly.", flush=True)
    trainer.train(resume_from_checkpoint=rk)
    trainer.save_model()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--objective", choices=OBJECTIVES)
    p.add_argument("--name", help="run name; also the output subdirectory under train.out_dir")
    p.add_argument("--wandb", action="store_true",
                   help="log to Weights & Biases (off by default; needs `wandb login` or WANDB_API_KEY)")
    p.add_argument("--wandb-project", dest="wandb_project", help="override wandb.project")
    p.add_argument("--total-steps", type=int)
    p.add_argument("--grad-accum", type=int,
                   help="set accumulation explicitly, bypassing the global_batch_seqs derivation "
                        "(this is how you deliberately change the token budget)")
    p.add_argument("--micro-batch-size", dest="micro_batch_size", type=int,
                   help="override train.micro_batch_size (per-GPU); accumulation re-derives around it, "
                        "so the token budget is unchanged")
    p.add_argument("--n-mtp", type=int, help="override model.n_mtp (e.g. 0 for the pure-NTP baseline)")
    p.add_argument("--mtp-per-head-norm", dest="mtp_per_head_norm", action="store_true", default=None,
                   help="DeepSeek-faithful per-head final RMSNorm (override model.mtp_per_head_norm) — for the shared-vs-per-head A/B on one config")
    p.add_argument("--init-from", dest="init_from", default=None,
                   help="WEIGHTS-ONLY init from a pretrained checkpoint (fresh optimizer + schedule). "
                        "Use for continued training; --resume-from-checkpoint is for crash recovery.")
    p.add_argument("--resume-from-checkpoint", dest="resume", default=None,
                   help="path to a checkpoint dir, or 'auto' to resume the latest in output_dir")
    p.add_argument("--mtp-weight", dest="mtp_weight", type=float, default=None,
                   help="override model.mtp_weight (the auxiliary MTP loss weight lambda_MTP axis)")
    p.add_argument("--seed", type=int, default=None,
                   help="override the top-level `seed` (init + data order + RNG). Use for SEED REPLICATES at a "
                        "fixed config (e.g. is an off-trend lambda point real or run-to-run noise?) — give the "
                        "replicate a distinct --name so it does not collide with the seed-0 run.")
    p.add_argument("--set", dest="overrides", nargs="+", default=[], metavar="dotted.key=value",
                   help="generic config override(s), e.g. --set optim.lr=1e-3 data.train_bin=x.bin; the "
                        "value is parsed as YAML. Applied BEFORE the named flags above. The key must "
                        "already exist unless its parent section does (a typo must not pass silently).")
    a = p.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    for ov in a.overrides:
        key, sep, val = ov.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects dotted.key=value, got {ov!r}")
        *path, leaf = key.split(".")
        node = cfg
        for part in path:
            if not isinstance(node.get(part), dict):
                raise SystemExit(f"--set {key}: no config section {part!r}")
            node = node[part]
        v = yaml.safe_load(val)
        if isinstance(v, str):               # PyYAML reads "1e-3" (no dot) as a STRING — coerce numerics
            try:
                v = float(v)
            except ValueError:
                pass
        node[leaf] = v
    if a.objective:
        cfg["model"]["objective"] = a.objective
    if a.total_steps:
        cfg["train"]["total_steps"] = a.total_steps
    if a.micro_batch_size:
        cfg["train"]["micro_batch_size"] = a.micro_batch_size
    if a.n_mtp is not None:
        cfg["model"]["n_mtp"] = a.n_mtp
    elif cfg["model"]["objective"] == "ntp":
        cfg["model"]["n_mtp"] = 0        # the pure next-token floor: no MTP heads to build or train
    if a.mtp_per_head_norm:
        cfg["model"]["mtp_per_head_norm"] = True
    if a.mtp_weight is not None:
        cfg["model"]["mtp_weight"] = a.mtp_weight
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.name:
        cfg.setdefault("wandb", {})["name"] = a.name
    if a.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    if a.wandb_project:
        cfg.setdefault("wandb", {})["project"] = a.wandb_project
    if a.resume:
        cfg["train"]["resume_from_checkpoint"] = a.resume
    if getattr(a, "init_from", None):
        cfg["model"]["init_from"] = a.init_from
    if not a.name:                       # config name + objective, so an A/B lands in its own dir
        cfg.setdefault("wandb", {}).setdefault(
            "name", f"{Path(a.config).stem}_{cfg['model']['objective']}")

    # Accumulation is DERIVED, never configured: the invariant that has to hold across arms is
    # tokens per step, and the world size is only knowable here. A config carrying `grad_accum`
    # instead (a custom one, say) is taken literally.
    t, world = cfg["train"], int(os.environ.get("WORLD_SIZE", "1"))
    if a.grad_accum:
        t["grad_accum"] = a.grad_accum
    elif "global_batch_seqs" in t:
        per_step = t["micro_batch_size"] * world
        if t["global_batch_seqs"] % per_step:
            raise SystemExit(
                f"global_batch_seqs ({t['global_batch_seqs']}) is not divisible by "
                f"micro_batch_size ({t['micro_batch_size']}) x world_size ({world}). Launch on a GPU "
                f"count that divides {t['global_batch_seqs'] // t['micro_batch_size']}, or change "
                f"micro_batch_size — silently training at a different token budget is worse.")
        t["grad_accum"] = t["global_batch_seqs"] // per_step
    if int(os.environ.get("RANK", "0")) == 0:
        seqs = t["micro_batch_size"] * t["grad_accum"] * world
        print(f"[train] {a.config}  objective={cfg['model']['objective']}  "
              f"{world} GPU x mbs {t['micro_batch_size']} x ga {t['grad_accum']} = {seqs} seqs x "
              f"{cfg['data']['seq_len']} = {seqs * cfg['data']['seq_len']:,} tokens/step", flush=True)

    train(cfg)


if __name__ == "__main__":
    main()
