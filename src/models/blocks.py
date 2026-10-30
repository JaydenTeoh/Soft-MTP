"""Pure-PyTorch Llama-style transformer block (no FLA/triton; runs on CPU + GPU).

RMSNorm (fp32-upcast, matches Llama) + RoPE (rotate_half convention) + GQA attention via
`F.scaled_dot_product_attention` (uses FlashAttention on CUDA) + SwiGLU. Verified equal to
HF `LlamaForCausalLM` to 1e-4. One `Block` is reused
for trunk layers and MTP heads, and `torch.compile`s cleanly (regional compile).
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        in_dtype = x.dtype  # upcast LOW precision to fp32 (matches Llama; stable under bf16),
        h = x.float() if in_dtype in (torch.float16, torch.bfloat16) else x  # but never downcast fp64
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * h).to(in_dtype)


def build_rope_cache(seq_len, head_dim, theta, device, dtype=torch.float32):
    """cos/sin table [T, head_dim]. Trig computed in fp32, cast to `dtype` last."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    # Disable any outside autocast context if any, to really force fp32 (autocast has no policy for
    # `meta`, which the param-count probes build on, so the context is guarded rather than assumed).
    ctx = (torch.autocast(device_type=dev.type, enabled=False)
           if dev.type in ("cpu", "cuda", "xpu") else nullcontext())
    with ctx:
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, head_dim]
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """cos/sin are [T, D] (batch-shared positions) or [B, T, D] (per-row `position_ids`, the HF
    convention — `generate()` derives them as `attention_mask.cumsum(-1) - 1`, so left-padded rows
    get 0..n-1 on their REAL tokens). Both broadcast to q/k's [B, H, T, D]."""
    if cos.dim() == 2:
        cos, sin = cos[None, None], sin[None, None]  # [1, 1, T, D]
    else:
        cos, sin = cos[:, None], sin[:, None]        # [B, 1, T, D]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.n_kv = cfg.num_heads, cfg.num_kv_heads
        self.hd = cfg.hidden_size // cfg.num_heads
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, past_key_values=None, layer_idx=0, cache_position=None,
                cache_readonly=False, attention_mask=None):
        """`attention_mask` is the pre-built ADDITIVE 4D mask [B, 1, T, k_len] (causal + padding), or
        None for the training/cache-free path which uses the compile-clean `is_causal` kernel."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if past_key_values is not None:
            if cache_readonly:
                # READ-ONLY: attend against the cache without writing to it. transformers'
                # `update()` is unconditionally append-in-place, so this is hand-rolled. The sequential
                # drafter needs it — its last positions consume unconfirmed DRAFT tokens, which must
                # attend but must never enter the cache, or a rejected draft poisons later drafts.
                layer = past_key_values.layers[layer_idx]
                pk, pv = getattr(layer, "keys", None), getattr(layer, "values", None)
                if pk is not None and pk.numel():
                    k = torch.cat([pk, k], dim=-2)
                    v = torch.cat([pv, v], dim=-2)
            else:                                        # inference KV cache (eager): append + read full K/V
                k, v = past_key_values.update(k, v, layer_idx)
        if self.n_kv != self.n_heads:  # GQA: expand kv heads
            rep = self.n_heads // self.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        if attention_mask is None and past_key_values is not None and cache_position is not None:
            # SAFETY NET: a cached call whose caller forgot the pre-built mask would otherwise run
            # BIDIRECTIONAL attention with no error — exactly the f0816ed regression that left
            # `_dsmtp_layer` non-causal on multi-position chunks while every decode gate kept
            # passing (the chunk's LAST position is causal by accident, and that is the only one
            # sampled from). Model-level callers still pass the shared mask, so this never fires on
            # the trunk/sequential paths and their build-once economics are unchanged.
            attention_mask = build_causal_mask(cache_position, k.size(2), None, x.device, q.dtype)
        if past_key_values is None and attention_mask is None:
            # training / cache-free without padding: clean compiled is_causal kernel. This mirrors
            # HF's contract exactly — `create_causal_mask` returns None when the sdpa backend can
            # use `is_causal` (no padding, no cache), and the attention interface falls back to it.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:                                            # cached, and/or padded cache-free: additive mask
            # `attention_mask` here is the FINAL additive mask, already combining causal + padding,
            # built ONCE per forward by `build_causal_mask` and shared by every layer — the same split
            # HF uses (DeepseekV3Model calls `create_causal_mask` and DeepseekV3Attention only passes
            # the result to the attention interface). Building it here instead would reallocate the
            # same [B,1,T,k_len] tensor once per layer.
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


def build_causal_mask(cache_position, k_len, attention_mask, device, dtype):
    """The additive [B, 1, T, k_len] mask for the CACHED path: causal AND padding, built once per
    forward and shared by every layer (HF's `create_causal_mask` plays this role for DeepSeek-V3 /
    Llama; their attention modules never construct a mask themselves).

    causal : key position <= query's absolute `cache_position`.
    padding: `attention_mask` is the HF convention [B, k_len], 1 = real token, 0 = pad. Without this
             term a query attends PAD keys, which silently degrades ANY batched generation with mixed
             prompt lengths — lm-eval left-pads its generate_until batches.

    Returns a 2D [T, k_len] mask when there is no padding to apply, so the common single-sequence case
    does not pay for a [B,1,T,k_len] broadcast.
    """
    T = cache_position.numel()
    kpos = torch.arange(k_len, device=device)
    allow = kpos[None, :] <= cache_position[:, None]
    mask = torch.zeros(T, k_len, device=device, dtype=dtype).masked_fill(~allow, float("-inf"))
    if attention_mask is None:
        return mask
    am = attention_mask[:, :k_len]
    mask = mask[None, None].expand(am.size(0), 1, T, k_len).clone()
    return mask.masked_fill_(am[:, None, None, :] == 0, float("-inf"))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-norm transformer block; reused for trunk layers and MTP heads."""

    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, past_key_values=None, layer_idx=0, cache_position=None,
                cache_readonly=False, attention_mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, past_key_values, layer_idx, cache_position,
                          cache_readonly, attention_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x
