"""Fetch the PRE-TOKENIZED FineWeb-Edu bins instead of tokenizing from scratch.

    python -m src.fetch_data --shards 5          # 50B tokens, ~100 GB   (the default)
    python -m src.fetch_data --shards 1          # 10B tokens,  ~20 GB
    python -m src.fetch_data --shards 0          # val.bin only, 21 MB

`src/prepare_data.py` tokenizes FineWeb-Edu yourself and takes hours of CPU. The same corpus,
already tokenized with the same GPT-2 BPE by that same pipeline, is published as
`JaydenTeoh/fineweb-edu-100bt-gpt2`: ten 20 GB shards of exactly 10B tokens each, plus a
10,511,706-token val.bin. The repo is public, so no HF token is needed. Prefer this; reach for
prepare_data.py only if you want a different corpus, tokenizer or holdout.

train.bin is stored sharded because a single 200 GB file hangs the Hub's xet backend. The shards
tile the corpus contiguously, so `shards=N` is exactly the first N*10B tokens and concatenating
them reproduces one flat uint16 stream -- which is all `BinDataset` wants.

HOW MANY SHARDS. A config needs `total_steps * global_batch_seqs * seq_len` tokens; the four
shipped here are 60000 * 256 * 2048 = 31.5B, so 4 shards is the minimum and 5 is the default.

⚠ Shard count changes the data ORDER, not just the amount: `BinDataset` derives
`n = filesize // block_len` and walks `randperm(n)` under the run seed. Runs you intend to compare
must read the same train.bin. Fetching fewer tokens than the budget is worse than a reordering --
the run wraps and re-trains on data it has already seen.

Downloads resume. If one stalls (the Hub throttles a long single transfer now and then), interrupt
it and re-run: finished shards are skipped and a partial shard picks up where it stopped.
hf_transfer (in requirements.txt) does the transfer in parallel multipart; without it the
download falls back to a single slow stream.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

SHARD_BYTES = 20_000_000_000          # 20 GB == 10B uint16 tokens, per published shard
VAL_BYTES = 21_023_412                # 10,511,706 tokens
REPO = "JaydenTeoh/fineweb-edu-100bt-gpt2"


def _human(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _size(p):
    return p.stat().st_size if p.exists() else -1


def _download(repo, filename, cache_dir, token):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset",
                                local_dir=cache_dir, token=token))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/fineweb-edu",
                    help="output directory; the shipped configs read data/fineweb-edu (default)")
    ap.add_argument("--shards", type=int, default=5,
                    help="train shards to fetch, 0-10; each is exactly 10B tokens (default 5 = 50B)")
    ap.add_argument("--repo", default=REPO, help="override the source dataset repo")
    ap.add_argument("--keep-shards", action="store_true",
                    help="keep the downloaded shard files instead of deleting each after appending")
    a = ap.parse_args(argv)

    if not 0 <= a.shards <= 10:
        sys.exit(f"--shards must be 0..10, got {a.shards}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_shards"
    token = os.environ.get("HF_TOKEN")        # the repo is public; honoured if you set one anyway

    if not os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"):
        try:
            import hf_transfer  # noqa: F401
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        except ImportError:
            print("[fetch] hf_transfer not installed -- downloads will be single-stream and slow.\n"
                  "[fetch] `pip install hf_transfer` and re-run for parallel multipart.", flush=True)

    # ---- val.bin ---------------------------------------------------------------------------
    val = out / "val.bin"
    if _size(val) == VAL_BYTES:
        print(f"[fetch] val.bin already present ({VAL_BYTES:,} bytes) -- skipping")
    else:
        print(f"[fetch] val.bin ({_human(VAL_BYTES)}) ...", flush=True)
        shutil.move(str(_download(a.repo, "val.bin", tmp, token)), val)

    # ---- train.bin -------------------------------------------------------------------------
    want = a.shards * SHARD_BYTES
    train = out / "train.bin"
    if a.shards == 0:
        print("[fetch] --shards 0: val.bin only, train.bin not written")
    elif _size(train) == want:
        print(f"[fetch] train.bin already {a.shards} shard(s) ({want:,} bytes) -- skipping")
    else:
        if _size(train) > 0:
            print(f"[fetch] train.bin is {_size(train):,} bytes, want {want:,} -- rebuilding")
        names = [f"train_{i:02d}.bin" for i in range(a.shards)]
        print(f"[fetch] {a.shards} shard(s) = {want // 2:,} tokens, {_human(want)} "
              f"(peak disk ~{_human(want + SHARD_BYTES)} while assembling)", flush=True)
        for i, name in enumerate(names):          # download all first: a partial one resumes
            print(f"[fetch]   {name} ({i + 1}/{a.shards}) ...", flush=True)
            _download(a.repo, name, tmp, token)
        print("[fetch] concatenating -> train.bin", flush=True)
        with open(train, "wb") as dst:
            for name in names:
                src = tmp / name
                if _size(src) != SHARD_BYTES:
                    sys.exit(f"[fetch] {name} is {_size(src):,} bytes, expected {SHARD_BYTES:,}; "
                             f"delete {tmp} and re-run")
                with open(src, "rb") as f:
                    shutil.copyfileobj(f, dst, length=32 << 20)
                if not a.keep_shards:
                    src.unlink()                  # free disk as we go
                print(f"[fetch]   + {name}", flush=True)

    # Only reached once every shard has been appended, so what is left under _shards is the
    # hub's own .cache metadata. An interrupted run raises before here and keeps its partial
    # download, which is what makes a re-run resume rather than restart.
    if tmp.exists() and not a.keep_shards:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- verify ----------------------------------------------------------------------------
    ok = True
    if _size(val) != VAL_BYTES:
        print(f"[fetch] FAIL val.bin {_size(val):,} != {VAL_BYTES:,}"); ok = False
    if a.shards and _size(train) != want:
        print(f"[fetch] FAIL train.bin {_size(train):,} != {want:,}"); ok = False
    if not ok:
        sys.exit(1)
    budget = want // 2 // (256 * 2048)
    print(f"[fetch] OK  {out}/  val.bin {VAL_BYTES // 2:,} tokens"
          + (f" | train.bin {want // 2:,} tokens = {budget:,} steps at 256 x 2048" if a.shards else ""))


if __name__ == "__main__":
    main()
