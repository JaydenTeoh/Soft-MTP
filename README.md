<div align="center">

# Soft Multi-Token Prediction

<img src="media/soft_mtp.png" alt="Soft Multi-Token Prediction" width="100%"/>

[![arXiv](https://img.shields.io/badge/arXiv-2610.38331-b31b1b.svg)](https://arxiv.org/abs/2610.38331) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-FFD21E.svg)](https://huggingface.co/collections/JaydenTeoh/soft-multi-token-prediction)

<h3>Official codebase for <a href="https://arxiv.org/abs/2610.38331">Multi-Token Prediction Done Right</a></h3>

Author: <a href="https://jaydenteoh.github.io/" style="color: #0891B2; text-decoration: none;">Jayden Teoh</a>


</div>


## Setup

```bash
git clone https://github.com/JaydenTeoh/Soft-MTP && cd Soft-MTP
pip install -r requirements.txt
python -m src.fetch_data --shards 5
```

This installs the necessary packages and fetches the pre-tokenized FineWeb-Edu tokens from
[JaydenTeoh/fineweb-edu-100bt-gpt2](https://huggingface.co/datasets/JaydenTeoh/fineweb-edu-100bt-gpt2). 

`--shards N` fetches N x 10B tokens into `data/fineweb-edu/` and resumes if interrupted. Every
config in `configs/` trains on 31.5B tokens, so 4 shards is the minimum; the default of 5 needs
~120 GB of disk while assembling.

PyTorch 2.9 or later is required to use the Muon optimizer.

## Pretrained checkpoints 🤗

The models from the paper for NTP, MTP and Soft-MTP objectives are available on the
[Hugging Face](https://huggingface.co/collections/JaydenTeoh/soft-multi-token-prediction). The MTP heads are included for self-speculative decoding; `generate()` ignores them.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("JaydenTeoh/soft-mtp-689m-model", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("JaydenTeoh/soft-mtp-689m-model")
```

## KL gradient–recovery trick

`combined_mtp_loss()` (in [src/models/loss.py](src/models/loss.py)) is a single function covering all objectives.

Standard MTP (`hard_mtp`) and Soft-MTP (`soft_mtp`) share this loss function and differ only by the `is_soft` condition. 

The KL gradient–recovery trick which is integral to Soft-MTP can be best understood from the following code chunk:

```python
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
```

Verify that the KL gradient–recovery trick produces the same gradient at the MTP heads as explicitly minimising `KL(q || p)`, to machine precision in float64:

```bash
pytest tests/ -q
```

## Training

```bash
torchrun --standalone --nproc_per_node=4 -m src.train \
  --config configs/167m_fineweb.yaml --objective soft_mtp
```

The objective can be toggled via the command-line argument:

```bash
--objective soft_mtp     # the method
--objective hard_mtp     # the standard MTP with cross-entropy
--objective direct_kl    # direct KL minimization
--objective ntp          # next-token prediction
```

**Note:** `direct_kl` materializes teacher and student softmaxes over the full vocabulary, so it needs noticeably more memory than the others.

We provide the configs for the 4 model sizes we use in the paper under the `configs/` folder:

<div align="center">

| **Config File** | **NTP parameters** | **+ K=3 MTP heads** | 
|---|---|---|
| [167m_fineweb.yaml](configs/167m_fineweb.yaml) | 167M | 178M | 
| [332m_fineweb.yaml](configs/332m_fineweb.yaml) | 332M | 357M | 
| [689m_fineweb.yaml](configs/689m_fineweb.yaml) | 689M | 745M | 
| [1p2b_fineweb.yaml](configs/1p2b_fineweb.yaml) | 1174M | 1273M |

</div>

## Evaluation

We use [Weights & Biases](https://wandb.ai) for metric tracking. Log in via:

```bash
wandb login <wandb_token>
```

Per-head cross-entropy, KL-to-verifier, and other metrics are logged throughout training in `eval_every=1000` intervals. 

For downstream benchmark evaluations, we use the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
python -m src.eval_lm --model hf --tasks wikitext,lambada_openai,piqa,hellaswag \
  --model_args pretrained=output/167m_fineweb_soft_mtp/checkpoint-60000,tokenizer=gpt2,backend=causal,dtype=float32 \
  --batch_size 32 --output_path results/lm_eval/167m_soft
```
Insert the prefix `accelerate launch --num_processes N` to evaluate data-parallel across GPUs.


To evaluate the speculative decoding performance of the models after training, use:

```bash
python -m src.eval_spec --ckpts output/167m_fineweb_soft_mtp/checkpoint-60000 \
                        --val data/fineweb-edu/val.bin --n-prompts 128 --fp32
```
Measures:
- `tok/fwd`: tokens generated per trunk forward pass
- `accept_h*`: per-head acceptance, conditional on all earlier drafts being accepted
- `alpha`: mean per-draft acceptance
- speedup: wall-clock against a KV-cached autoregressive baseline

Acc. Len. is `alpha * n_mtp`, excluding the always-accepted next token.

The paper evaluates five prompt sets — `fineweb`, `wiki`, `books`, `science`, `math` prepared under `src/data/prompts/` as fixed 256 x 512 token arrays, so every run sees the same prompts:

```bash
python -m src.eval_spec --ckpts output/167m_fineweb_soft_mtp/checkpoint-60000 --fp32 \
  --domain-npy fineweb=src/data/prompts/fineweb_p512.npy \
               wiki=src/data/prompts/wiki_p512.npy \
               books=src/data/prompts/books_p512.npy \
               science=src/data/prompts/science_p512.npy \
               math=src/data/prompts/math_p512.npy
```

Wall-clock speedup depends on the GPU, so run the arms you are comparing on the same machine.


## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{teoh2026multitokenpredictionright,
  title={Multi-Token Prediction Done Right},
  author={Jayden Teoh},
  journal={arXiv preprint arXiv:2610.38331},
  year={2026},
}
```

MIT. See [LICENSE](LICENSE).
