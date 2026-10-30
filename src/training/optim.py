"""LR schedule helpers (cosine OR warmup-stable-decay).

`lr_lambda_factory` is the SINGLE source of truth for the LR multiplier: the HF Trainer
(`MTPTrainer.create_scheduler`) builds a `LambdaLR` from it when `lr_schedule="wsd"`, and
`wsd_decay_start` marks the end-of-stable branch step.

- **cosine** (the default): linear warmup -> cosine decay over `[warmup, total]` to `min_lr_ratio`.
- **wsd** (warmup-stable-decay): linear warmup -> CONSTANT peak (stable) -> a terminal cooldown over the
  last `decay_fraction*total` steps to `min_lr_ratio`, with `decay_shape`:
    - `1-sqrt`  (default; Nemotron/MiniCPM): `1 - sqrt(p)` — drops fast then dwells at low LR, with
      no high-LR plateau at the start of the cooldown.
    - `cosine`  (DeepSeek-V4 cooldown): `0.5(1+cos(pi p))` — flat start, steep middle, flat end.
    - `linear`: `1 - p`.
  The stable phase makes a single end-of-stable checkpoint a clean branch point for decay A/Bs.
"""

import math
from collections import OrderedDict

import torch


# --- Muon (+ auxiliary AdamW) optimizer ---------------------------------------
# torch.optim.Muon takes ONLY the 2D hidden matrices; the embedding, lm_head and all 1D params
# (RMSNorm gains, biases) go to an auxiliary AdamW. `adjust_lr_fn="match_rms_adamw"` scales Muon's
# update by 0.2*sqrt(max(m,n)) so it reuses AdamW's lr and wd (Moonshot arXiv:2502.16982). Weight
# decay applies to the Muon matrices only: not to the vocab-coupled embedding/unembedding, and not
# to norm gains (flip `vector` below to decay them).
_MUON_EXCLUDE_TAGS = ("embed", "lm_head")   # 2D params kept on AdamW (input embedding + output head)


def split_muon_adamw_params(model):
    """Partition trainable params (dion / Moonshot / NextLat convention) into three lists:
      muon        — 2D hidden matrices (attention / MLP / sequential-proj weights)  -> Muon,
      embed_head  — the embedding + unembedding (lm_head)                        -> AdamW (wd 0),
      vector      — all 1D params (RMSNorm gains, biases)                        -> AdamW (wd 0)."""
    muon, embed_head, vector = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and any(tag in name for tag in _MUON_EXCLUDE_TAGS):
            embed_head.append(p)     # embed / lm_head (2D) -> AdamW, NO weight decay (dion)
        elif p.ndim == 2:
            muon.append(p)           # hidden matrix -> Muon
        else:
            vector.append(p)         # 1D norms / biases -> AdamW, NO weight decay
    return muon, embed_head, vector


class MuonWrapper(torch.optim.Optimizer):
    """Wrap Muon (2D hidden matrices) + an auxiliary AdamW (embed/lm_head/norms) as ONE
    `torch.optim.Optimizer` so the HF Trainer, accelerate, and the LR scheduler treat them uniformly.
    `param_groups` is the CONCATENATION of the
    sub-optimizers' groups (the SAME dict objects), so a `LambdaLR` that writes `group['lr']` reaches
    every sub-optimizer. `state_dict`/`load_state_dict` roundtrip each sub-optimizer -> resumable."""

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        self.defaults = dict(self.optimizers[0].defaults)
        # replicate the attrs torch.optim.Optimizer.__init__ would set (we bypass super().__init__
        # because there is no single (params, defaults) to give it) so hook-registering callers are safe.
        for hook in ("_optimizer_step_pre_hooks", "_optimizer_step_post_hooks",
                     "_optimizer_state_dict_pre_hooks", "_optimizer_state_dict_post_hooks",
                     "_optimizer_load_state_dict_pre_hooks", "_optimizer_load_state_dict_post_hooks"):
            setattr(self, hook, OrderedDict())
        self.state = {}
        # expose the real group dicts (mutated in place by the scheduler)
        self.param_groups = [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for o in self.optimizers:
            o.step()
        return loss

    def state_dict(self):
        return {"combined": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state_dict):
        for o, sd in zip(self.optimizers, state_dict["combined"]):
            o.load_state_dict(sd)
        self.param_groups = [g for o in self.optimizers for g in o.param_groups]

    def add_param_group(self, param_group):
        raise NotImplementedError("MuonWrapper has fixed Muon/AdamW groups")


def build_muon_adamw(model, *, lr, weight_decay, betas, adam_eps=1e-8,
                     momentum=0.95, nesterov=True, ns_steps=5, adjust_lr_fn="match_rms_adamw"):
    """Muon (2D hidden matrices, wd) + auxiliary AdamW (embed/lm_head/norms, wd 0) as one
    `MuonWrapper`. `lr` is shared — valid because `match_rms_adamw` matches Muon's update RMS to
    AdamW's; weight decay lands only on the Muon matrices (dion practice; embed/lm_head/norms get 0)."""
    muon_p, embed_head_p, vector_p = split_muon_adamw_params(model)
    muon = torch.optim.Muon(muon_p, lr=lr, weight_decay=weight_decay, momentum=momentum,
                            nesterov=nesterov, ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn)
    adamw = torch.optim.AdamW(
        [{"params": embed_head_p, "weight_decay": 0.0},   # embed + lm_head: NO wd (dion)
         {"params": vector_p, "weight_decay": 0.0}],      # norms + biases:  NO wd
        lr=lr, betas=tuple(betas), eps=adam_eps)
    return MuonWrapper([muon, adamw])


def decay_curve(p, shape):
    """Cooldown multiplier f(p): 1 -> 0 as p goes 0 -> 1 (p clamped to [0,1])."""
    p = min(max(p, 0.0), 1.0)
    if shape == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * p))
    if shape == "linear":
        return 1.0 - p
    if shape in ("1-sqrt", "1_minus_sqrt", "inv_sqrt", "sqrt"):
        return 1.0 - math.sqrt(p)
    raise ValueError(f"unknown decay_shape: {shape!r} (use 1-sqrt | cosine | linear)")


def wsd_decay_start(warmup_steps, total_steps, decay_fraction):
    """Step at which the WSD terminal cooldown begins (== end of the stable phase)."""
    decay_steps = max(1, round(decay_fraction * total_steps))
    return max(int(warmup_steps), int(total_steps) - decay_steps)


def lr_lambda_factory(warmup_steps, total_steps, *, schedule="cosine", min_lr_ratio=0.1,
                      decay_fraction=0.2, decay_shape="1-sqrt"):
    """Return `step -> LR multiplier` in `[min_lr_ratio, 1]`. See module docstring for the shapes."""
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    if schedule not in ("cosine", "wsd"):
        raise ValueError(f"unknown schedule: {schedule!r} (use cosine | wsd)")
    decay_start = wsd_decay_start(warmup_steps, total_steps, decay_fraction)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if step >= total_steps:
            return min_lr_ratio
        if schedule == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * decay_curve(progress, "cosine")
        # wsd: constant peak through the stable phase, then the terminal cooldown
        if step < decay_start:
            return 1.0
        p = (step - decay_start) / max(1, total_steps - decay_start)
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay_curve(p, decay_shape)

    return lr_lambda

