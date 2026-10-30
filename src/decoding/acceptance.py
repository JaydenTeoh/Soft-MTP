"""Acceptance-statistics aggregation for the self-speculative decoders.

Consumed by the cached sequential decoders, `eval.evaluation.
evaluate_self_spec`, and `eval_spec.py`.
"""


def aggregate_acceptance(stats_list, n_mtp):
    """Combine per-prompt stats into headline metrics.

    Standard SD metrics (as in the speculative-decoding literature):
      * `accept_h{i}` = CONDITIONAL acceptance rate AR_i = P(accept draft i | drafts <i all accepted).
        Conditional by construction: the decoders increment `proposed[i]` inside the verify loop,
        which `break`s on the first rejection, so position i is only ever proposed when every
        earlier draft was accepted.
      * `acceptance_length` = AL = expected tokens COMMITTED per verification step, INCLUDING the
        one free token every round yields — the residual correction on a reject, or the bonus
        sampled from the target when all k drafts are accepted. Each round commits `m + 1` tokens
        (`confirmed += m + 1` in the decoders), so AL = mean(m) + 1 = mean_drafts_accepted + 1.
        Plain autoregressive decoding is AL = 1.
      * `mean_drafts_accepted` = mean(m), the DRAFTS only — AL without the free token. Kept because
        it is the raw quantity the decoders report and what alpha is derived from.
      * `tokens_per_forward` = generated / trunk-forwards. Equals AL up to the one-off prefill
        forward amortised over the generation (<1% at 512 new tokens), and does NOT count the k
        cache-free sequential drafting passes -- hence a speedup UPPER BOUND, not wall-clock.
      * `tau_iid` = the Leviathan block estimate (1-α^{γ+1})/(1-α) using α = mean(m)/γ; the i.i.d.
        idealisation of AL, for comparison against the measured value.
    """
    prop = [sum(s["proposed"][i] for s in stats_list) for i in range(n_mtp)]
    acc = [sum(s["accepted"][i] for s in stats_list) for i in range(n_mtp)]
    n_fwd = sum(s["n_forward"] for s in stats_list)
    n_gen = sum(s["n_generated"] for s in stats_list)
    rounds = [m for s in stats_list for m in s["round_accepts"]]
    per_head = [acc[i] / max(1, prop[i]) for i in range(n_mtp)]
    mean_accept = sum(rounds) / max(1, len(rounds))         # mean drafts accepted / round
    out = {
        "tokens_per_forward": n_gen / max(1, n_fwd),        # ≈ speedup (cache-free upper bound)
        "mean_drafts_accepted": mean_accept,                # drafts only (AL without the free token)
        "acceptance_length": mean_accept + 1.0,             # AL: + the free verification token
        "n_forward": n_fwd, "n_generated": n_gen,
    }
    for i in range(n_mtp):
        out[f"accept_h{i + 1}"] = per_head[i]               # head i+1 (horizon +{i+2})
    if n_mtp > 0:
        alpha = mean_accept / n_mtp                          # overall per-draft acceptance
        out["alpha"] = alpha
        out["tau_iid"] = ((1 - alpha ** (n_mtp + 1)) / (1 - alpha)) if alpha < 1 else (n_mtp + 1)
    return out
