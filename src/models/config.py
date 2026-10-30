"""HF config for the MTP model.

Standalone `PretrainedConfig` (no FLA). `num_hidden_layers` is the NTP trunk depth; `n_mtp`
heads are added on top (Meta) or sequential (DeepSeek), per `mtp_arch`. `objective` lives on the
config so the model's `forward` can return the right loss for the HF `Trainer` (model-returns-
loss pattern); the training script sets it.
"""

from transformers import PretrainedConfig

# `soft_mtp` = KL / soft-target MTP (the mirror of `hard_mtp`'s one-hot targets). Do NOT confuse with
# metric names that contain `kl` (e.g. `mtp_kl_verifier` = KL-to-verifier) — those are unrelated.
OBJECTIVES = ("ntp", "hard_mtp", "soft_mtp", "direct_kl")


class HFMTPConfig(PretrainedConfig):
    model_type = "hf_mtp"

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=768,
        num_hidden_layers=12,        # NTP trunk depth
        num_heads=12,
        num_kv_heads=None,           # None -> = num_heads (no GQA)
        n_mtp=3,                     # number of MTP heads / sequential depths
        mtp_head_layers=1,           # transformer blocks per MTP head/depth
        mtp_arch="deepseek",         # "deepseek" (sequential, DeepSeek-V3 style) | "meta" (parallel)
                                     # The two differ in what a head READS, not just its shape: a
                                     # sequential depth consumes the teacher-forced token x_{t+k} and
                                     # is cropped to T-k, a meta head ignores input_ids entirely.
                                     # Code built from this default rather than from a shipped
                                     # config therefore gets a structurally different model.
        mtp_per_head_norm=False,     # DeepSeek-faithful: each MTP head gets its OWN final RMSNorm before
                                     # the SHARED lm_head (official `shared_head.norm` is per-module; only
                                     # the head weight is shared). False (default) = one shared final norm
                                     # for NTP + all heads.

        max_position_embeddings=1024,
        rope_theta=10000.0,
        norm_eps=1e-5,
        intermediate_size=None,      # None -> ~8/3 * hidden, rounded to 128
        tie_word_embeddings=False,   # untied by default
        objective="soft_mtp",        # ntp | hard_mtp | soft_mtp | direct_kl
        mtp_weight=1.0,              # auxiliary weight on the MTP loss term (the STABLE-phase value if scheduled)
        ce_kernel="naive",           # MTP-head CE: "naive" (materialize logits) | "liger" (fused linear CE, low mem)
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.n_mtp = n_mtp
        self.mtp_head_layers = mtp_head_layers
        self.mtp_arch = mtp_arch
        self.mtp_per_head_norm = mtp_per_head_norm
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        if intermediate_size is None:
            inter = int(8 * hidden_size / 3)
            intermediate_size = (inter + 127) // 128 * 128
        self.intermediate_size = intermediate_size
        self.objective = objective
        self.mtp_weight = mtp_weight
        self.ce_kernel = ce_kernel
        super().__init__(vocab_size=vocab_size, tie_word_embeddings=tie_word_embeddings, **kwargs)
        assert hidden_size % num_heads == 0
        assert num_heads % self.num_kv_heads == 0
        assert mtp_arch in ("meta", "deepseek")
        assert self.objective in OBJECTIVES, f"unknown objective {objective!r}"
