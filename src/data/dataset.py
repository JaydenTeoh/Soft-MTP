"""Pre-tokenized binary datasets + MTP block packing.

Tokenize a text corpus once into a flat `uint16` `.bin` (documents separated by
EOT), then stream fixed-length blocks via a memmap. A block has length
`seq_len + n_future` so every head (NTP + n_mtp) has `seq_len` valid labels via
a padded per-head target matrix.
"""

import numpy as np
import torch
from torch.utils.data import Dataset



def prepare_bin(docs, tokenizer, out_path, text_key="text", report_every=100_000):
    """Tokenize an iterable of dicts/strings to a flat uint16 .bin (EOT-separated).

    Returns the total number of tokens written.
    """
    total = 0
    with open(out_path, "wb") as f:
        for i, doc in enumerate(docs):
            text = doc if isinstance(doc, str) else doc[text_key]
            ids = tokenizer.encode(text)
            ids.append(tokenizer.eot)
            np.asarray(ids, dtype=np.uint16).tofile(f)
            total += len(ids)
            if report_every and (i + 1) % report_every == 0:
                print(f"  {i + 1} docs, {total:,} tokens")
    return total


class BinDataset(Dataset):
    """`seq_len + 1` token blocks from a flat uint16 .bin via memmap, STRIDED BY `seq_len`.

    READ ONE MORE TOKEN THAN YOU FEED. The block is `seq_len + 1` long but only its first `seq_len`
    tokens are input; the extra token is the target of the LAST input position. Stride is `seq_len`, so
    consecutive blocks overlap by exactly that one token and the corpus is covered with no gap: every
    token is an input to exactly one block AND a target of exactly one block (modulo the two ends).
    Without the +1 the final input position would produce a trunk hidden with no target -- computed and
    discarded -- and NTP would be supervised on seq_len-1 of seq_len positions.

    `n_future` is accepted for signature compatibility and does NOT size the read. That matters:
    block length is `seq_len + 1` for EVERY objective, so `n` — and hence the seeded data order — is
    identical for a pure-NTP arm and an MTP arm it is the baseline for.

    ⚠ THE POOL SIZE IS PART OF THE EXPERIMENT. `n = filesize // block_len` and the order is
    `randperm(n)` under the run seed, so tokenizing a different amount of data changes the order
    every arm sees. Arms being compared must read the SAME bin; a different pool is equivalent to
    a different data seed.
    """

    def __init__(self, path, seq_len, n_future=None):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.block_len = seq_len + 1      # read one extra token: the last input position's target
        self.stride = seq_len             # ... but ADVANCE by seq_len, so blocks overlap by that token
        # a block starting at (n-1)*stride must have block_len tokens available
        self.n = max(0, (len(self.data) - 1) // seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.stride
        block = self.data[s : s + self.block_len].astype(np.int64)
        return torch.from_numpy(block)


def mtp_collate(blocks, seq_len, n_future=None):
    """[B] blocks of length seq_len+1 -> (input_ids[B, seq_len], labels[B, seq_len]).

    The nanoGPT x/y split: `labels` is the block shifted left by one, SAME length as the input, so
    labels[:, t] is the target of position t and every input position is supervised. Head j's target is
    then just `labels[:, j:]` (see `combined_mtp_loss`). `n_future` is unused, kept for call
    compatibility.
    """
    block = torch.stack(blocks)                    # [B, seq_len + 1]
    return block[:, :-1].contiguous(), block[:, 1:].contiguous()

# The HF Trainer path wants a dict (`model(**inputs)`); the eval path wants the tuple above.
# Both split one `seq_len + 1` block nanoGPT-style into x = block[:-1] / y = block[1:], so
# `labels[:, t]` is the target of position t and head j's target is the slice `labels[:, j:]`.


def mtp_collate_dict(blocks, seq_len, n_future=None):
    block = torch.stack(blocks)  # [B, seq_len + 1]
    return {
        "input_ids": block[:, :-1].contiguous(),
        "labels": block[:, 1:].contiguous(),   # PRE-SHIFTED: labels[:, t] is the target of position t
    }
