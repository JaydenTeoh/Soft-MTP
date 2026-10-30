"""The KL-recovery identity: `soft_mtp` IS direct KL minimisation, one backward pass cheaper.

Soft-MTP trains each MTP head against the model's own next-token distribution `q` instead of the
realized token `e_y`. The obvious way to do that is `direct_kl`: materialise `q`, take a softmax
over the vocabulary per head, and minimise `KL(q || p)` explicitly. `soft_mtp` instead routes the
next-token head's own gradient into the MTP hiddens with a straight-through coupling, which costs
no extra softmax and no teacher cache.

These tests are the gate on that being the same thing. They check, in float64:

  1. the head-hidden and trunk gradients of `soft_mtp` equal those of `direct_kl` to machine
     precision -- the identity the method rests on;
  2. the two differ at the shared unembedding `W`, and ONLY there, which is a design decision
     rather than an error: the recovery is trunk-side, so `soft_mtp` keeps the hard
     cross-entropy push on `W` while `direct_kl` replaces it with the KL push;
  3. the head gradient equals the closed form `coeff_k * (p_k - q)^T W` written out from scratch,
     so neither implementation is taken on trust -- a shared misconception cannot satisfy both;
  4. the forward VALUE is untouched: `soft_mtp` and `hard_mtp` report the same loss, because the
     coupling is `x.detach() - x`, exactly zero in any dtype. The KL enters only in the backward.

Run: `pytest tests/ -q` from the repository root.
"""
import pytest
import torch

from src.models.config import HFMTPConfig
from src.models.loss import IGNORE_INDEX, combined_mtp_loss
from src.models.model import HFMTPForCausalLM

B, T, V, MTP_WEIGHT = 3, 11, 61, 0.7


def _model(objective, n_mtp=2, **kw):
    torch.manual_seed(0)
    return HFMTPForCausalLM(HFMTPConfig(
        vocab_size=V, hidden_size=24, num_hidden_layers=2, num_heads=4, intermediate_size=48,
        max_position_embeddings=32, n_mtp=n_mtp, mtp_arch="deepseek", objective=objective,
        mtp_weight=MTP_WEIGHT, **kw)).double()


def _batch(mask=None):
    torch.manual_seed(5)
    ids = torch.randint(0, V, (B, T))
    labels = torch.randint(0, V, (B, T))
    if mask == "prompt":                      # SFT-shaped: the first 4 positions are context
        labels[:, :4] = IGNORE_INDEX
    elif mask == "ragged":                    # a different ignored span per row
        for b in range(B):
            labels[b, : 2 + b] = IGNORE_INDEX
    return ids, labels


def _run(objective, mask=None, n_mtp=2, scale_mul=1.0, **kw):
    """Gradients at every locus the comparison cares about, for one objective."""
    m = _model(objective, n_mtp=n_mtp, **kw)
    ids, labels = _batch(mask)
    H = m.trunk_forward(ids).detach().requires_grad_(True)

    # Tap the POST-norm head hidden `z_k` -- the tensor the closed form is about, and the one the
    # CE and the coupling both consume. The PRE-norm hidden is the wrong place to look: the
    # sequential heads attend, so position t there also receives gradient from LATER positions.
    heads, orig_norm = [], m.final_norm
    def spy(hidden, head_idx=0):
        z = orig_norm(hidden, head_idx)
        if head_idx > 0:
            z.retain_grad()
            heads.append(z)
        return z
    m.final_norm = spy

    total, parts = combined_mtp_loss(m, H, ids, labels, objective=objective,
                                     mtp_weight=MTP_WEIGHT * scale_mul)
    total.backward()
    return {"heads": [h.grad.clone() for h in heads], "trunk": H.grad.clone(),
            "W": m.lm_head.weight.grad.clone(), "loss": total.item(), "parts": parts,
            "hiddens": [h.detach() for h in heads], "model": m, "labels": labels}


# --- 1. the identity -------------------------------------------------------------------------
@pytest.mark.parametrize("mask", [None, "prompt", "ragged"])
@pytest.mark.parametrize("n_mtp", [1, 2, 3])
def test_recovered_gradient_equals_direct_kl_minimisation(mask, n_mtp):
    """soft_mtp's gradient into the MTP heads and the trunk IS direct KL's, to fp64 precision."""
    s, d = _run("soft_mtp", mask, n_mtp), _run("direct_kl", mask, n_mtp)
    for k, (a, b) in enumerate(zip(s["heads"], d["heads"]), 1):
        assert torch.allclose(a, b, atol=1e-14, rtol=0), \
            f"head {k} differs by {(a - b).abs().max().item():.3e}"
    assert torch.allclose(s["trunk"], d["trunk"], atol=1e-14, rtol=0), \
        f"trunk differs by {(s['trunk'] - d['trunk']).abs().max().item():.3e}"


def test_the_two_objectives_differ_at_the_unembedding_and_only_there():
    """Not a bug: the recovery is trunk-side, so soft_mtp keeps the hard CE push on the shared W."""
    s, d = _run("soft_mtp"), _run("direct_kl")
    assert (s["W"] - d["W"]).abs().max() > 1e-3, "W gradients should DIFFER between the two"
    hard = _run("hard_mtp")
    assert torch.allclose(s["W"], hard["W"], atol=1e-14, rtol=0), \
        "soft_mtp's push on W should be identical to hard_mtp's"


# --- 2. against the closed form, trusting neither implementation ------------------------------
@pytest.mark.parametrize("mask", [None, "prompt"])
def test_head_gradient_matches_the_closed_form(mask):
    """dL/d z_k[t] == coeff_k * (p_k[t] - q[t+k])^T W, written out from scratch.

    Neither `soft_mtp` nor `direct_kl` is trusted here: both are compared against the algebra.
    """
    r = _run("soft_mtp", mask)
    m, labels = r["model"], r["labels"]
    W = m.lm_head.weight.detach()
    ids, _ = _batch(mask)
    H = m.trunk_forward(ids).detach()
    q = torch.softmax(m.final_norm(H, 0) @ W.t(), dim=-1)          # the next-token verifier
    n_heads = m.n_heads_total - 1
    for k, (g, z) in enumerate(zip(r["heads"], r["hiddens"]), 1):
        y = labels[:, k:]
        sup = (y != IGNORE_INDEX)
        p = torch.softmax(z @ W.t(), dim=-1)
        coeff = MTP_WEIGHT / (n_heads * sup.sum().clamp(min=1).double())
        want = coeff * ((p - q[:, k : k + y.size(1)]) * sup.unsqueeze(-1)) @ W
        assert torch.allclose(g, want, atol=1e-14, rtol=0), \
            f"head {k}: max|diff| {(g - want).abs().max().item():.3e}"


@pytest.mark.parametrize("mask", ["prompt", "ragged"])
def test_masked_positions_receive_exactly_zero(mask):
    r = _run("soft_mtp", mask)
    for k, g in enumerate(r["heads"], 1):
        y = r["labels"][:, k:]
        ignored = (y == IGNORE_INDEX)
        if ignored.any():
            assert g[ignored].abs().max() == 0.0, \
                f"head {k} leaks {g[ignored].abs().max().item():.3e} onto ignored positions"


# --- 3. the forward value is untouched ---------------------------------------------------------
def test_soft_and_hard_report_the_same_loss():
    """`x.detach() - x` is exactly 0, so the coupling cannot move the reported loss."""
    s, h = _run("soft_mtp"), _run("hard_mtp")
    assert s["loss"] == h["loss"], f"{s['loss']!r} != {h['loss']!r}"


# --- 4. mutation check: the gate must be able to fail ------------------------------------------
@pytest.mark.parametrize("mul", [0.5, 2.0])
def test_perturbing_the_recovery_scale_breaks_the_identity(mul):
    """If the identity held for a wrong mtp_weight the test above would be vacuous."""
    s = _run("soft_mtp", "prompt", scale_mul=mul)
    d = _run("direct_kl", "prompt")
    assert not torch.allclose(s["heads"][0], d["heads"][0], atol=1e-14, rtol=0)
