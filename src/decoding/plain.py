"""Cached plain-greedy AR baseline for the DeepSeek sequential (HF-Llama, `DynamicCache`).

The TRUNK keeps a `DynamicCache` over the sequence (one forward per token here). This is the FAIR
wall-clock timing baseline for the self-spec speedup AND the exact (lossless) reference for the trunk
KV-cache correctness — `plain_greedy_sequential_cached` must match the cache-free `plain_greedy_sequential`
bit-for-bit. The speculative-SAMPLING decoder
(`sample_speculative_sequential_cached.py`) reuses the same trunk cache for its verify step. We do NOT
do greedy speculative decoding — only sampling spec is measured (top-1 acceptance is blind to the
verifier-distribution match). Needs a cache-capable `trunk_forward(..., past_key_values, use_cache,
cache_position)`. Batch size 1.
"""

import torch

from .acceptance import aggregate_acceptance  # noqa: F401 (re-export for callers)


@torch.no_grad()
def plain_greedy_sequential_cached(model, prompt, max_new_tokens):
    """Cached greedy NTP autoregression (the FAIR speedup baseline). One trunk forward per token."""
    from transformers import DynamicCache
    model.eval()
    dev = prompt.device
    cache = DynamicCache()
    L = prompt.size(1)
    H = model.trunk_forward(prompt, past_key_values=cache, use_cache=True,
                            cache_position=torch.arange(L, device=dev))
    out = [int(model.logits(H[:, -1:])[0, -1].argmax())]
    pos = L
    while len(out) < max_new_tokens:
        H = model.trunk_forward(torch.tensor([[out[-1]]], device=dev), past_key_values=cache,
                                use_cache=True, cache_position=torch.arange(pos, pos + 1, device=dev))
        pos += 1
        out.append(int(model.logits(H[:, -1:])[0, -1].argmax()))
    return torch.tensor([out[:max_new_tokens]], device=dev)
