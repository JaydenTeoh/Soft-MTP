"""lm-evaluation-harness on trained checkpoints (a thin delegate to `lm_eval`'s own CLI).

    python -m src.eval_lm --model hf --tasks wikitext,lambada_openai,piqa,hellaswag \
      --model_args pretrained=output/167m_fineweb_soft_mtp/checkpoint-60000,tokenizer=gpt2,backend=causal,dtype=float32 \
      --batch_size 32 --output_path results/lm_eval/167m_soft

The ONLY thing this adds over `python -m lm_eval` is importing `src.models.model` first, which
registers `model_type: hf_mtp` with AutoConfig/AutoModelForCausalLM. After that the stock `hf`
backend loads our checkpoints like any other HF model, and every flag is lm_eval's, unchanged.

Three model_args are not optional:
  tokenizer=gpt2   checkpoints ship no tokenizer files. The padded vocab (50304 vs GPT-2's 50257)
                   only adds unused logit rows.
  backend=causal   lm_eval auto-detects the backend from a STATIC model_type -> class table that a
                   runtime AutoModelForCausalLM.register() does not extend, so without this the
                   load fails outright.
  dtype=float32    the checkpoint IS fp32 master weights. dtype=bfloat16 is a WEIGHT CAST -- a third
                   numeric regime matching neither training (bf16 autocast over fp32 weights) nor
                   the file.

Multi-GPU is DATA parallel -- each rank loads a full replica and lm_eval shards the requests:

    accelerate launch --num_processes 8 -m src.eval_lm --model hf ...

Do NOT pass parallelize=True: that shards one model across cards, for models too big for one GPU.

⚠ PIN --batch_size. With `auto`, lm_eval probes a size per machine, and batch composition changes
which items sit together under padding -- enough to move individual tasks by a few tenths of a
point. Fixed batch size and fixed world size are what make two runs comparable.

`max_length` needs no flag: HFLM reads config.max_position_embeddings (2048 for the shipped configs).
"""

import sys

import src.models.model  # noqa: F401  (registers hf_mtp with the Auto classes -- must precede lm_eval)

try:
    from lm_eval.__main__ import cli_evaluate
except ImportError:
    sys.exit("lm_eval is not installed (it is an optional extra):\n"
             "    pip install lm_eval\n"
             "then re-run.")

if __name__ == "__main__":
    cli_evaluate()
