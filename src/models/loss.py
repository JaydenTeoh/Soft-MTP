"""MTP losses: ntp | hard_mtp | soft_mtp | direct_kl.

soft_mtp trains each MTP head against the model's own next-token distribution instead of the
realized token, using a straight-through KL-recovery coupling: adding `s_k * (sg(z_k) - z_k)` into the
aligned slice of the NTP hidden leaves the forward value exactly unchanged, while the backward
routes the NTP CE's own hidden gradient into head k. The head nets `coeff_k * (p_k - q)W`, the KL
gradient, in one backward — no teacher cache, no second softmax.

`labels` is PRE-SHIFTED and the same length as `input_ids`: `labels[:, t]` is the target of
position t. Head k is supervised on its first `T - k` positions with target `labels[:, k:]`; NTP is
head 0 with target `labels`. This is NOT HF's convention (`ForCausalLMLoss` shifts internally); we
shift in the collator, so every input position is supervised.

⚠ Teacher forcing and supervision read DIFFERENT sequences: the sequential forces depth k from
`input_ids[:, k:]`, supervision comes from `labels`. They coincide in pretraining, which is the only
reason one tensor ever sufficed — under SFT `labels` is IGNORE_INDEX over the prompt and embedding
-100 would crash.
"""

import importlib

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def direct_kl_loss_sum(student_logits, teacher_logits_aligned, labels, lam=1.0, ignore_index=IGNORE_INDEX):
    """Forward KL D(q‖p) against the detached verifier, summed over non-ignored positions."""
    # torch's KL has no ignore_index (only cross_entropy does), so the mask is hand-rolled.
    mask = (labels != ignore_index).unsqueeze(-1)
    hi = lambda t: t.float() if t.dtype in (torch.float16, torch.bfloat16) else t   # never DOWNcast fp64
    logq = F.log_softmax(hi(teacher_logits_aligned), dim=-1).detach()
    q = logq.exp()
    logp = F.log_softmax(hi(student_logits), dim=-1)
    kl = q * (logq - logp)                      # [..., V]
    return lam * (kl * mask).sum()


def _ce_mean(logits, target, ignore_index=IGNORE_INDEX):
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=ignore_index)


_LIGER_FLCE = None


@torch.compiler.disable
def _liger_flce(W, z, y, ignore_index):
    # The disable is required: dynamo cannot trace Liger's autograd Function under bf16 autocast
    # (its fused addmm(out_dtype=fp32) fails fake-tensor propagation). Costs a graph break per CE
    # call, not the kernel.
    global _LIGER_FLCE
    if _LIGER_FLCE is None:
        # importlib, not a literal `from liger_kernel...` line: liger is OPTIONAL, but HF's
        # trust_remote_code loader statically regexes every import in a vendored modelling file
        # and refuses to load unless each one is installed. A real import statement here would
        # make an optional dependency mandatory for anyone loading this model from the Hub.
        mod = importlib.import_module("liger_kernel.transformers.fused_linear_cross_entropy")
        _LIGER_FLCE = mod.LigerFusedLinearCrossEntropyLoss(ignore_index=ignore_index, reduction="mean")
    return _LIGER_FLCE(W, z.reshape(-1, z.size(-1)), y.reshape(-1))


def ce_loss(z, W, y, ce_kernel, ignore_index=IGNORE_INDEX):
    """Mean per-token CE against `W`, for any head. `naive` materializes `[*, V]` logits; `liger`
    streams over the vocabulary and never does, which is the dominant memory term at large V."""
    if ce_kernel == "liger":
        return _liger_flce(W, z, y, ignore_index)
    return _ce_mean(F.linear(z, W), y, ignore_index)


def _direct_kl_loss(model, hiddens, h_ntp, labels, W, mtp_weight, ce_kernel):
    """Full-KD ablation: the KL gradient reaches the trunk AND the shared `W`, where soft_mtp
    recovers it into the trunk only and leaves `W` the hard push. Materializes `[B, T, V]` softmaxes.

    Logging units stay cross-arm comparable: `mtp_heads` is the counterfactual hard CE (what the
    other objectives report), `mtp_heads_kl` is the KL that actually enters `total`.
    """
    ntp_hidden = h_ntp
    ntp_loss = ce_loss(ntp_hidden, W, labels, ce_kernel)
    parts = {"ntp": ntp_loss.detach(), "mtp_heads": [], "mtp_heads_kl": []}
    hi = lambda t: t.float() if t.dtype in (torch.float16, torch.bfloat16) else t
    verifier_logits = (hi(ntp_hidden) @ hi(W).t()).detach()
    head_terms = []
    for head in range(1, model.n_heads_total):
        head_targets = labels[:, head:]
        head_span = head_targets.size(1)
        head_hidden = model.final_norm(hiddens[head], head)[:, :head_span]
        student_logits = hi(head_hidden) @ hi(W).t()
        # The teacher is a plain crop: head `head` at t and NTP at t+head are supervised under the
        # SAME condition, so every student position has a real teacher and there is no tail to mask.
        teacher_logits = verifier_logits[:, head:head + head_span]
        n_head_supervised = (head_targets != IGNORE_INDEX).sum().clamp(min=1)
        head_terms.append(direct_kl_loss_sum(student_logits, teacher_logits, head_targets)
                          / n_head_supervised)
        parts["mtp_heads"].append(_ce_mean(student_logits.detach(), head_targets).detach())
        parts["mtp_heads_kl"].append(head_terms[-1].detach())
    mtp_mean = torch.stack(head_terms).mean()
    total = ntp_loss + mtp_weight * mtp_mean
    parts["mtp"] = torch.stack(parts["mtp_heads"]).mean()
    parts["total"] = total.detach()
    return total, parts


def combined_mtp_loss(model, H, input_ids, labels, objective=None, mtp_weight=None):
    """One scalar loss for ntp | hard_mtp | soft_mtp | direct_kl -> (total, parts).

    hard_mtp and soft_mtp share one head loop: identical per-head CE against the same undetached
    `W`, differing only in that soft_mtp also accumulates the KL-recovery term and runs the NTP CE on
    the coupled hidden. The loss VALUE is the hard CE either way — the KL enters through the gradient.

    `parts` (detached, for logging): {'ntp', 'mtp', 'mtp_heads', 'total'}; direct_kl adds
    'mtp_heads_kl', the terms that actually enter `total`.
    """
    cfg = model.config
    objective = cfg.objective if objective is None else objective
    mtp_weight = cfg.mtp_weight if mtp_weight is None else mtp_weight
    ce_kernel = getattr(cfg, "ce_kernel", "naive")
    W = model.lm_head.weight

    hiddens = model.all_head_hiddens(H, input_ids)       # [h0, ..., h_n_mtp] pre-norm
    h_ntp = model.final_norm(hiddens[0], 0)

    assert labels.shape[:2] == input_ids.shape[:2], (
        f"labels must be PRE-SHIFTED and the same shape as input_ids (labels[:, t] = target of "
        f"position t); got {tuple(labels.shape)} vs {tuple(input_ids.shape)}. HF's convention is the "
        f"other one (unshifted labels, shifted inside the loss) -- collate with mtp_collate_dict."
    )
    # Guard, not a branch: with an empty head slice `mtp_mean` would average over fewer heads than
    # `kl_recovery_scale` divides by, silently turning soft_mtp into a partial-removal arm.
    assert labels.size(1) > model.n_heads_total - 1, (
        f"seq_len ({labels.size(1)}) must exceed n_mtp ({model.n_heads_total - 1}): the deepest head "
        f"would have no target."
    )

    if objective == "ntp" or cfg.n_mtp == 0:
        ntp_loss = ce_loss(h_ntp, W, labels, ce_kernel)
        return ntp_loss, {"ntp": ntp_loss.detach(), "mtp_heads": [],
                          "mtp": torch.zeros((), device=H.device), "total": ntp_loss.detach()}

    if objective == "direct_kl":
        return _direct_kl_loss(model, hiddens, h_ntp, labels, W, mtp_weight, ce_kernel)

    # ---- shared head loop: hard_mtp | soft_mtp ---------------------------------------------------
    is_soft = objective == "soft_mtp"
    ntp_targets = labels
    ntp_hidden = h_ntp
    n_heads = model.n_heads_total - 1

    # float64 BEFORE dividing: an int64/int64 tensor divide is float32 in torch, and a trailing
    # .to() only re-rounds what was already lost.
    n_ntp_supervised = ((ntp_targets != IGNORE_INDEX).sum().clamp(min=1).to(torch.float64)
                        if is_soft else None)

    head_losses = []
    for head in range(1, model.n_heads_total):
        # Head k predicts the token k ahead, so head has seq_len-k supervised positions.
        head_targets = labels[:, head:]
        head_hidden = model.final_norm(hiddens[head], head)[:, :head_targets.size(1)]
        head_losses.append(ce_loss(head_hidden, W, head_targets, ce_kernel))
        if not is_soft:
            continue

        # straight-through gradient routing
        # s_k undoes the NTP CE's 1/T mean and re-applies k-th MTP head's own
        # 1/(T-k) mean and the 1/K head average, so the routed gradient
        # lands at exactly the scale the MTP head's own CE gradient uses.
        kl_recovery_scale = (mtp_weight * n_ntp_supervised
                       / (n_heads * (head_targets != IGNORE_INDEX).sum().clamp(min=1).to(torch.float64))
                       ).to(head_hidden.dtype)

        # ntp_hidden is numerically unchanged; only its backward changes
        ntp_hidden[:, head:] = ntp_hidden[:, head:] + kl_recovery_scale * (head_hidden.detach() - head_hidden)

    ntp_loss = ce_loss(ntp_hidden, W, ntp_targets, ce_kernel)
    mtp_mean = torch.stack(head_losses).mean()
    total = ntp_loss + mtp_weight * mtp_mean
    return total, {"ntp": ntp_loss.detach(), "mtp_heads": [m.detach() for m in head_losses],
                   "mtp": mtp_mean.detach(), "total": total.detach()}
