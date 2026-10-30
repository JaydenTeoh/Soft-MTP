"""Verifier-agreement metrics for the MTP heads.

The *verifier* is the next-token head: at position t it defines q_t = softmax(W h_t), the
distribution the MTP heads should match for self-speculative decoding. Head j at position t
predicts x_{t+1+j}, so its aligned teacher is the verifier at position t+j -- the same offset
as training.

Per horizon, averaged over supervised positions:
  * kl_verifier_mtp : D_KL(q || p), the headline draft-quality metric
  * tv_qp           : TV(q, p). Speculative acceptance is exactly 1 - TV, so this is the
                      teacher-forced predictor of the acceptance rate measured by decoding.

LOWER on both means the head matches the verifier better, hence higher expected acceptance.
That is what soft_mtp targets; hard-label CE is not (it is reported as mtp/h*_ce for contrast).
"""

from collections import defaultdict

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


@torch.no_grad()
def distill_metrics_for_head(p_logits, q_logits, labels_j, j, ignore_index=IGNORE_INDEX):
    """Metrics for one MTP head.

    p_logits [B, L_j, V] and labels_j [B, L_j] are ALREADY at head j's own span L_j (= T-j): the
    sequential returns h^j cropped, and the target is the slice labels[:, j:]. q_logits [B, T, V] is the
    FULL-length verifier, and the +j alignment happens HERE -- q_full[t+j] pairs with p[t], since both
    predict x_{t+1+j}. (An earlier caller passed full-length p/labels and this function did
    all three crops; doing it twice is what broke it.)

    Returns a dict of *summed* metrics plus `count` (valid positions), ready to aggregate across
    batches; None if no valid positions.
    """
    Lj = p_logits.size(1)
    if Lj == 0 or j + Lj > q_logits.size(1):
        return None
    p = p_logits.float()
    q = q_logits[:, j:j + Lj].float()      # q_aligned[t] = q_full[t + j]
    y = labels_j
    mask = y != ignore_index
    if int(mask.sum()) == 0:
        return None
    pm, qm = p[mask], q[mask]              # [n, V], [n, V]

    logp = F.log_softmax(pm, dim=-1)
    logq = F.log_softmax(qm, dim=-1)
    prob_p, prob_q = logp.exp(), logq.exp()
    kl = (prob_q * (logq - logp)).sum(-1)          # KL(q || p) per position

    tv = 0.5 * (prob_q - prob_p).abs().sum(-1)     # total variation TV(q,p) per position

    s = lambda t: float(t.sum())
    return dict(
        count=int(mask.sum()),
        kl_verifier_mtp=s(kl),
        tv_qp=s(tv),
    )


def evaluate_distill(model, loader, device, n_batches=20, k=10, autocast=True):
    """Aggregate per-head verifier-agreement metrics over `n_batches` val batches.

    Returns {j: {metric: value, 'count': int}} for each MTP head j=1..n_mtp.
    """
    model.eval()
    n_future = model.n_heads_total
    acc = {j: defaultdict(float) for j in range(1, n_future)}
    for bi, (input_ids, labels) in enumerate(loader):
        if bi >= n_batches:
            break
        input_ids, labels = input_ids.to(device), labels.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
            H = model.trunk_forward(input_ids)
            q_logits = model.logits(model.head_forward(H, 0))   # NTP verifier logits
            for j in range(1, n_future):
                p_logits = model.logits(model.head_forward(H, j), j)   # per-head final norm when mtp_per_head_norm
                tgt = labels[:, j:]
                m = distill_metrics_for_head(p_logits[:, :tgt.size(1)], q_logits, tgt, j, k)
                del p_logits
                if m is None:
                    continue
                for key, val in m.items():
                    acc[j][key] += val
    model.train()
    out = {}
    for j, a in acc.items():
        c = max(1.0, a["count"])
        out[j] = {key: a[key] / c for key in a if key != "count"}
        out[j]["count"] = int(a["count"])
    return out
