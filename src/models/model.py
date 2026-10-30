"""MTP model — parallel (Meta-style) and sequential (DeepSeek-style) heads in one class.

HF `PreTrainedModel` over the verified pure-PyTorch `Block` (no FLA/triton), so it runs under
the HF `Trainer` with a single combined backward and also imports/runs locally. `forward`
returns the combined MTP loss (model-returns-loss pattern) so `Trainer` picks it up directly.

  trunk: num_hidden_layers Blocks -> H (pre-norm); NTP reads H directly.
  meta:     head j  = mtp_head_layers Block(s) on H (parallel)        -> predicts x_{t+1+j}
  deepseek: depth k = block_k(proj_k[RMSNorm(h^{k-1}); RMSNorm(emb(x_{t+k}))])  (sequential)
  shared final RMSNorm + lm_head (so every head goes through the same W -> recovery identity).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .blocks import RMSNorm, Block, build_causal_mask, build_rope_cache
from .config import HFMTPConfig
from .loss import combined_mtp_loss


@dataclass
class MTPCausalLMOutput(CausalLMOutputWithPast):
    """`CausalLMOutputWithPast` + the NAMED per-step train-loss breakdown (detached tensors), so the
    Trainer logs by field name instead of unpacking a positional tensor. All fields are plain tensors
    (pytree-safe under whole-model torch.compile / DDP). Units are cross-arm-consistent: `mtp_loss`
    is hard CE on EVERY arm (for direct_kl the counterfactual CE); `kl_loss` is the KL training loss,
    present only on direct_kl (there `loss = ntp_loss + w*mean(kl_loss)`, else `+ w*mean(mtp_loss)`).

    Logging convention (implemented in `MTPTrainer.log`): a scalar field logs under its own name; a
    per-head vector field `<x>_loss` ([n_mtp]) logs its MEAN under `<x>_loss` plus `<x>_h{j}_loss`
    per head (1-based)."""
    ntp_loss: Optional[torch.FloatTensor] = None  # scalar: NTP CE
    mtp_loss: Optional[torch.FloatTensor] = None  # [n_mtp]: per-head hard CE (MTP arms)
    kl_loss: Optional[torch.FloatTensor] = None   # [n_mtp]: per-head KL (direct_kl only)

    @classmethod
    def breakdown_fields(cls):
        """The breakdown = exactly the fields this class ADDS over `CausalLMOutputWithPast` —
        derived, not hand-listed, so adding a field here is the only change ever needed."""
        from dataclasses import fields
        base = {f.name for f in fields(CausalLMOutputWithPast)}
        return tuple(f.name for f in fields(cls) if f.name not in base)


class HFMTPPreTrainedModel(PreTrainedModel):
    config_class = HFMTPConfig
    _no_split_modules = ["Block"]

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=std)


class HFMTPForCausalLM(HFMTPPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: HFMTPConfig):
        super().__init__(config)
        Blk = Block
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.trunk = nn.ModuleList(Blk(config) for _ in range(config.num_hidden_layers))
        D, base = config.n_mtp, config.num_hidden_layers
        if config.mtp_arch == "meta":
            self.mtp_heads = nn.ModuleList(
                nn.ModuleList(Blk(config) for _ in range(config.mtp_head_layers)) for _ in range(D)
            )
        else:  # deepseek sequential
            self.proj = nn.ModuleList(nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False) for _ in range(D))
            self.norms_h = nn.ModuleList(RMSNorm(config.hidden_size, config.norm_eps) for _ in range(D))
            self.norms_e = nn.ModuleList(RMSNorm(config.hidden_size, config.norm_eps) for _ in range(D))
            self.mtp_blocks = nn.ModuleList(
                nn.ModuleList(Blk(config) for _ in range(config.mtp_head_layers)) for _ in range(D)
            )
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)   # NTP (head 0) final norm; also the
        # shared final norm for the MTP heads UNLESS mtp_per_head_norm (then each head has its own, below —
        # the DeepSeek-faithful `shared_head.norm`; the lm_head weight stays shared either way).
        if getattr(config, "mtp_per_head_norm", False):
            self.mtp_final_norms = nn.ModuleList(
                RMSNorm(config.hidden_size, config.norm_eps) for _ in range(D))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._rope = None
        self.post_init()

    # --- HF plumbing ---
    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    # --- core ---
    @property
    def n_heads_total(self):
        return 1 + self.config.n_mtp

    def _rope_cache(self, T, device, dtype):
        """Cache the fp32 table and cast on return, as HF does (it buffers only the dtype-stable
        `inv_freq` and ends with `cos.to(dtype=x.dtype)`). Caching the CAST result went stale across a
        dtype change -- the check below sees length and device but not dtype -- so an fp32 forward then
        `.to(bfloat16)` fed fp32 cos/sin to a bf16 model and SDPA raised on mixed q/k/v dtypes."""
        if self._rope is None or self._rope[0].size(0) < T or self._rope[0].device != device:
            hd = self.config.hidden_size // self.config.num_heads
            self._rope = build_rope_cache(T, hd, self.config.rope_theta, device, torch.float32)
        cos, sin = self._rope
        return cos[:T].to(dtype), sin[:T].to(dtype)

    def trunk_forward(self, input_ids, past_key_values=None, use_cache=False, cache_position=None,
                      attention_mask=None, position_ids=None, inputs_embeds=None):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        x = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        T = x.size(1)
        if past_key_values is None and cache_position is None:
            if attention_mask is not None and not bool(attention_mask.all()):
                # padded cache-free batch (e.g. lm-eval generate_until with use_cache=False): the
                # additive causal+padding mask replaces `is_causal`. The `.all()` skip mirrors HF's
                # `allow_is_causal_skip` — an all-ones mask keeps the fast kernel. Eval-only branch
                # (training never passes a mask), so the CPU sync is fine and compile never sees it.
                cp = torch.arange(T, device=x.device)
                attn_mask = build_causal_mask(cp, T, attention_mask, x.device, x.dtype)
                if position_ids is None:
                    cos, sin = self._rope_cache(T, x.device, x.dtype)
                else:
                    cos_all, sin_all = self._rope_cache(int(position_ids.max()) + 1, x.device, x.dtype)
                    cos, sin = cos_all[position_ids], sin_all[position_ids]
                for block in self.trunk:
                    x = block(x, cos, sin, attention_mask=attn_mask)
                return x
            # training / cache-free fast path (compile-clean: contiguous [0:T] rope + is_causal SDPA)
            cos, sin = self._rope_cache(T, x.device, x.dtype)
            for block in self.trunk:
                x = block(x, cos, sin)
            return x  # pre-norm trunk hidden H (NTP reads this)
        # inference with a KV cache (eager): used by the cached self-spec decoders + self-spec eval.
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if cache_position is None:
            past = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past, past + T, device=x.device)
        # RoPE: per-row `position_ids` when given (HF convention — generate() derives them from the
        # attention mask, so left-padded rows count 0..n-1 on real tokens), else the batch-shared
        # absolute `cache_position` (equivalent up to a uniform shift, which relative RoPE cancels).
        if position_ids is not None:
            cos_all, sin_all = self._rope_cache(int(position_ids.max()) + 1, x.device, x.dtype)
            cos, sin = cos_all[position_ids], sin_all[position_ids]
        else:
            cos_all, sin_all = self._rope_cache(int(cache_position.max()) + 1, x.device, x.dtype)
            cos, sin = cos_all[cache_position], sin_all[cache_position]
        # Build the additive causal+padding mask ONCE and share it across layers — HF's split
        # (DeepseekV3Model calls create_causal_mask; the attention modules only consume it). Per-layer
        # construction would reallocate the same [B,1,T,k_len] tensor num_hidden_layers times.
        k_len = int(cache_position.max()) + 1
        attn_mask = build_causal_mask(cache_position, k_len, attention_mask, x.device, x.dtype)
        for i, block in enumerate(self.trunk):
            x = block(x, cos, sin, past_key_values, i, cache_position, attention_mask=attn_mask)
        return x

    def _meta_head(self, H, j):
        if j == 0:
            return H
        cos, sin = self._rope_cache(H.size(1), H.device, H.dtype)
        x = H
        for block in self.mtp_heads[j - 1]:
            x = block(x, cos, sin)
        return x

    def _dsmtp_layer(self, h_prev, token_ids, depth,
                     past_key_values=None, cache_position=None, cache_readonly=False):
        """Sequential depth `depth`. `h_prev` is h^{depth-1}; `token_ids[t]` is the teacher-forced token
        x_{t+depth}.

        INFERENCE KV CACHE (optional; the training path is untouched and bit-identical). With
        `past_key_values`/`cache_position` the depth's blocks attend over the cached prefix, so a
        drafter can extend the sequential by a few positions per round instead of re-running it over the
        whole sequence. Mirrors `trunk_forward`'s cached branch: rope at ABSOLUTE positions, one
        `layer_idx` per block within THIS depth's own cache.

        `cache_readonly=True` attends against the cache without writing. The caller owns the split:
        depth d at position t consumes x_{t+d}, so positions with t+d <= (last confirmed index) are
        CONFIRMED and may be written (append-only, monotone, no crop -- the cache stays contiguous),
        while the trailing positions consume unconfirmed DRAFTS and must be read-only or a rejected
        draft poisons later draft distributions. See
        `decoding/sequential.py`.
        """
        i = depth - 1
        e = self.embed(token_ids.clamp(min=0))
        h = self.proj[i](torch.cat((self.norms_h[i](h_prev), self.norms_e[i](e)), dim=-1))
        # Carrier-precision parity: under bf16 autocast the proj emits bf16 while the trunk carrier
        # h_prev is fp32, an asymmetry no production stack has. Match the carrier rather than forcing
        # .float() — under a weight-cast bf16 model a forced fp32 would crash the block GEMMs.
        h = h.to(h_prev.dtype)
        if past_key_values is None and cache_position is None:
            # training / cache-free fast path — UNCHANGED (contiguous [0:T] rope + is_causal SDPA)
            cos, sin = self._rope_cache(h.size(1), h.device, h.dtype)
            for block in self.mtp_blocks[i]:
                h = block(h, cos, sin)
        else:
            past = past_key_values.get_seq_length() if past_key_values is not None else 0
            if cache_position is None:
                cache_position = torch.arange(past, past + h.size(1), device=h.device)
            cos_all, sin_all = self._rope_cache(int(cache_position.max()) + 1, h.device, h.dtype)
            cos, sin = cos_all[cache_position], sin_all[cache_position]   # rope at ABSOLUTE positions
            # Causal mask built once and shared by this depth's blocks. k_len is the post-update key
            # width `past + T`, NOT cache_position.max()+1 — the latter under-counts for a read-only
            # overlap chunk, whose new keys duplicate already-cached positions.
            attn_mask = build_causal_mask(cache_position, past + h.size(1), None, h.device, h.dtype)
            for bi, block in enumerate(self.mtp_blocks[i]):
                h = block(h, cos, sin, past_key_values, bi, cache_position, cache_readonly, attn_mask)
        return h

    # Public alias (the cache-free sequential self-spec decoder calls `model.dsmtp_layer(...)`, matching
    # the FLA `dsmtp_transformer` interface).
    def dsmtp_layer(self, h_prev, token_ids, depth,
                    past_key_values=None, cache_position=None, cache_readonly=False):
        return self._dsmtp_layer(h_prev, token_ids, depth,
                                 past_key_values=past_key_values, cache_position=cache_position,
                                 cache_readonly=cache_readonly)

    def head_forward(self, H, j):
        """Pre-norm hidden of head j (0 = NTP trunk). Matches the FLA model interface so
        the shared `metrics.distill.evaluate_distill` / `trunk_grad_agreement` run on HF too.
        DeepSeek depth j needs the sequential context -> use `all_head_hiddens` instead there."""
        if self.config.mtp_arch == "meta":
            return self._meta_head(H, j)
        raise NotImplementedError("DeepSeek head_forward: use all_head_hiddens(H, labels)")

    def all_head_hiddens(self, H, input_ids=None):
        """Pre-norm hidden per head [h^0, ..., h^{n_mtp}]. Meta: parallel (input_ids unused).

        DeepSeek: teacher-forced sequential. Depth k consumes the INPUT TOKEN x_{t+k}, i.e.
        `input_ids[:, k:]`, so it is defined for t in [0, T-k) and h^k is CROPPED to T-k positions --
        exactly the span on which head k is supervised (it predicts x_{t+1+k}).

        ⚠ TEACHER FORCING READS INPUTS, NOT LABELS. These are the same sequence only in
        pretraining. Under supervised fine-tuning `labels` is IGNORE_INDEX over the prompt, and
        embedding -100 would crash -- so the sequential must never be fed labels. Nothing is read past the
        input, so there is no padded target matrix and no mask.

        The crop is per-depth and the loop is unrolled at trace time, so every slice length is a
        compile-time constant -- no dynamic shapes, no recompilation.
        """
        if self.config.mtp_arch == "meta":
            return [self._meta_head(H, j) for j in range(self.n_heads_total)]
        T = input_ids.size(1)
        outs, h_prev = [H], H
        for k in range(1, self.n_heads_total):
            L = T - k                                  # positions whose teacher token is in-window
            if L <= 0:
                break
            h_prev = self._dsmtp_layer(h_prev[:, :L], input_ids[:, k:k + L], k)
            outs.append(h_prev)
        return outs

    def final_norm(self, hidden, head_idx=0):
        """Final RMSNorm before the shared lm_head. head_idx 0 = NTP -> self.norm. With
        mtp_per_head_norm, head_idx k>=1 uses the head's OWN norm (DeepSeek `shared_head.norm`);
        otherwise every head reuses self.norm (a NO-OP threading when the flag is off -> callers can
        always pass the index and stay bit-identical)."""
        if head_idx == 0 or not getattr(self.config, "mtp_per_head_norm", False):
            return self.norm(hidden)
        return self.mtp_final_norms[head_idx - 1](hidden)

    def logits(self, hidden, head_idx=0):
        return F.linear(self.final_norm(hidden, head_idx), self.lm_head.weight)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: "int | torch.Tensor" = 0,
        return_all_heads: Optional[bool] = None,
        **kwargs,
    ) -> MTPCausalLMOutput:
        """Signature follows `DeepseekV3ForCausalLM.forward` (HF `modeling_deepseek_v3.py`): HF
        argument order and names (`position_ids`, `inputs_embeds`, int-or-tensor `logits_to_keep`),
        and `past_key_values` returned on the output.

        ⚠ `position_ids` in this signature is BEHAVIOUR, not decoration.
        `GenerationMixin.prepare_inputs_for_generation` inspects the signature and, finding the name,
        derives per-row positions as `attention_mask.cumsum(-1) - 1`, so a left-padded row gets RoPE
        0..n-1 on its real tokens. That makes batched and unbatched generation exactly equal.

        ⚠ The cache args are load-bearing. If they fall into `**kwargs` and get dropped while
        `trunk_forward` runs cache-free, `generate()` (which defaults to `use_cache=True`) feeds only
        the newest token from step 2 on, believing the prefix is cached — the model then sees one
        token with no context and emits degenerate repetition after a plausible first token.

        The cache is created here, not in `trunk_forward`, only so we hold a reference to return;
        `Attention.forward` is what appends to it, in place, so returning the same object is correct.

        Training is untouched: with no cache or positions, `trunk_forward` takes its compile-clean
        `is_causal` fast path.
        """
        if (labels is not None or return_all_heads) and input_ids is None:
            raise ValueError("MTP paths teacher-force from token ids: `labels`/`return_all_heads` "
                             "require `input_ids`, not `inputs_embeds`")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        H = self.trunk_forward(input_ids, past_key_values=past_key_values, use_cache=use_cache,
                               cache_position=cache_position, attention_mask=attention_mask,
                               position_ids=position_ids, inputs_embeds=inputs_embeds)
        if return_all_heads:  # eval/distill: teacher-forced all-head logits [B, T-n_mtp, 1+n_mtp, V]
            # Head j is supervised on T-j positions, so the sequential returns RAGGED hiddens. Stacking
            # needs a rectangle -> take the common prefix (the deepest head's span, T-n_mtp). Nothing
            # meaningful is dropped: the tail positions this omits are exactly the ones no deep head
            # can predict, and every consumer compares heads AT THE SAME position.
            hs = self.all_head_hiddens(H, input_ids)
            n_pos = min(h.size(1) for h in hs)
            stacked = torch.stack([self.logits(h[:, :n_pos], j) for j, h in enumerate(hs)], dim=2)
            return MTPCausalLMOutput(logits=stacked)
        if labels is not None:  # training: combined MTP loss, one backward (full logits omitted to save mem)
            total, parts = combined_mtp_loss(self, H, input_ids, labels)
            # NAMED detached breakdown on the output (see MTPCausalLMOutput): the Trainer logs by field
            # name — no positional packing. `logits` stays None during training (full logits are
            # deliberately not materialized to save memory; eval uses return_all_heads).
            heads = parts["mtp_heads"]
            kl_heads = parts.get("mtp_heads_kl")               # direct_kl only
            return MTPCausalLMOutput(
                loss=total,
                ntp_loss=parts["ntp"],
                mtp_loss=torch.stack(heads) if heads else None,
                kl_loss=torch.stack(kl_heads) if kl_heads else None,
            )
        # `logits_to_keep` (DeepSeek-V3 / Llama convention): generation needs only the last position,
        # and at V=50304 a 900-token prefill would otherwise materialise ~181 MB of unused fp32 logits.
        # HF's exact form: int -> trailing slice (0 -> slice(0, None) == everything); Tensor -> fancy
        # index of explicit positions (`DeepseekV3ForCausalLM.slice_indices`).
        sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return MTPCausalLMOutput(logits=self.logits(H[:, sl, :]), past_key_values=past_key_values)


try:
    AutoConfig.register(HFMTPConfig.model_type, HFMTPConfig)
    AutoModelForCausalLM.register(HFMTPConfig, HFMTPForCausalLM)
except ValueError:
    pass
