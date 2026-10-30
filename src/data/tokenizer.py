"""GPT-2 BPE tokenizer (vocab 50257, EOT=50256); model vocab padded to 50304.

Tries `tiktoken`, falls back to HF `GPT2TokenizerFast`. The padded vocab (a
multiple of 128) is what the model config should use; the extra rows are unused
token ids and train to near-zero.
"""

EOT = 50256
VOCAB_SIZE = 50257
PADDED_VOCAB_SIZE = 50304


class _Tok:
    def __init__(self, encode_fn, encode_batch_fn=None):
        self._encode = encode_fn
        self._encode_batch = encode_batch_fn
        self.eot = EOT

    def encode(self, text):
        """Token ids for a document (no special tokens)."""
        return self._encode(text)

    def encode_batch(self, texts):
        """Token ids for a list of documents. Uses the tokenizer's internal multithreading
        (tiktoken `encode_ordinary_batch` releases the GIL across all cores) when available;
        falls back to a per-doc loop. ~cores× faster than `encode` per doc for big corpora."""
        if self._encode_batch is not None:
            return self._encode_batch(texts)
        return [self._encode(t) for t in texts]


def get_tokenizer():
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return _Tok(lambda t: enc.encode_ordinary(t),
                    lambda ts: enc.encode_ordinary_batch(ts))   # multithreaded (rayon)
    except Exception:
        from transformers import GPT2TokenizerFast

        hf = GPT2TokenizerFast.from_pretrained("gpt2")
        return _Tok(lambda t: hf.encode(t),
                    lambda ts: hf(ts, add_special_tokens=False)["input_ids"])
