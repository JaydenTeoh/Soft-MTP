"""Training-efficiency metrics: tokens/sec, step time, peak memory, MFU.

MFU uses the standard 6N FLOPs/token estimate plus the attention term, summed
over the trunk and all MTP-head layers.
"""

import time

import torch


def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def flops_per_token(n_params, n_layers_total, hidden, seq_len):
    """Approx fwd+bwd FLOPs/token: 6*N (matmuls) + 12*L*hidden*seq_len (attention)."""
    return 6 * n_params + 12 * n_layers_total * hidden * seq_len


class StepTimer:
    """Rolling throughput / step-time tracker; call .tick(tokens) once per step."""

    def __init__(self):
        self._t = time.perf_counter()

    def tick(self, tokens):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        dt = now - self._t
        self._t = now
        return {"step_time_s": dt, "tokens_per_sec": tokens / dt if dt > 0 else 0.0}

    @staticmethod
    def peak_memory_gb():
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1e9

    @staticmethod
    def reset_peak():
        # Clear the CUDA peak-allocation high-water mark so the next peak_memory_gb()
        # reflects only the upcoming window (training) and isn't stuck at an earlier
        # transient (e.g. the full-logit eval).
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def mfu(tokens_per_sec, flops_per_tok, gpu_peak_flops):
        if gpu_peak_flops <= 0:
            return 0.0
        return 100.0 * flops_per_tok * tokens_per_sec / gpu_peak_flops
