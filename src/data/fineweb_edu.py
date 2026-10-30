"""Prepare FineWeb-Edu into train/val uint16 .bin via streaming tokenization.

The main pretraining corpus. Streams a sample subset (default `sample-10BT`) so it
needs little disk; holds out the first `val_docs` documents for validation. Usage:
    python -m src.prepare_data --out data/fineweb-edu --subset sample-10BT
"""

import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np

from .sources import DataBundle, register_data_source
from .tokenizer import get_tokenizer


# --- Training data source (the default pretraining path) -----------------------------------------
# The FineWeb-Edu training corpus lives in a flat uint16 .bin (produced by the tokenizer below) and is
# served by the generic `BinDataset` memmap + `seq_to_mtp` collation, with the full `MTPEvalCallback`
# when a val bin is set. Registered here so it's co-located with the data; `build_trainer` resolves it
# by name (default). Heavy imports are deferred into the builder (called once at trainer-build time).
@register_data_source("fineweb_edu")
def _fineweb_edu_source(cfg, model, seq_len, n_future):
    from . import BinDataset
    from .dataset import mtp_collate_dict
    t = cfg["train"]
    train_ds = BinDataset(cfg["data"]["train_bin"], seq_len, n_future)
    collate = partial(mtp_collate_dict, seq_len=seq_len, n_future=n_future)

    callbacks = []
    val_bin = cfg["data"].get("val_bin")
    if val_bin and t.get("eval_every", 0):
        from ..eval.callback import MTPEvalCallback, eval_batches_from_cfg
        callbacks.append(MTPEvalCallback(
            val_bin=val_bin, seq_len=seq_len, n_future=n_future,
            eval_every=t["eval_every"], eval_batches=eval_batches_from_cfg(t),
            batch_size=t.get("eval_batch_size", 8),
            spec_every=t.get("spec_every"),
            spec_n_prompts=t.get("spec_n_prompts", 16),
            spec_prompt_len=t.get("spec_prompt_len", 256),
            spec_new_tokens=t.get("spec_new_tokens", 128),
            spec_temperature=t.get("spec_temperature", 1.0)))
    return DataBundle(train_dataset=train_ds, data_collator=collate, eval_callbacks=callbacks)


# --- Tokenization Pool (Karpathy/NextLat pipeline) -----------------------------------------------
# The MAIN process owns the single streaming iterator; Pool workers ONLY tokenize. This overlaps
# network I/O (main) with CPU encode (workers) and, crucially, avoids the per-worker-`shard()`
# starvation of the old design (unbalanced streaming shards left most workers near-empty, so the
# even per-worker token quota could never sum to the target). Output is the same flat uint16
# train.bin/val.bin the trainer's BinDataset memmaps.
_WORKER_TOK = None


def _init_worker():
    global _WORKER_TOK
    _WORKER_TOK = get_tokenizer()


def _encode_batch_worker(texts):
    """Tokenize a batch of documents -> (flat uint16 token array, int64 per-doc lengths). Top-level
    so it's picklable for `mp.Pool` under the spawn start method. Each doc gets a trailing EOT."""
    tok = _WORKER_TOK
    arrs, lens = [], []
    for ids in tok.encode_batch(texts):
        ids.append(tok.eot)
        arrs.append(np.asarray(ids, dtype=np.uint16))
        lens.append(len(ids))
    if arrs:
        return np.concatenate(arrs), np.asarray(lens, dtype=np.int64)
    return np.empty(0, dtype=np.uint16), np.zeros(0, dtype=np.int64)


def _prepare_pipeline(out_dir, subset, val_docs, max_tokens, batch_docs, workers):
    """ONE streaming iterator in the main process feeds document batches to a Pool that only
    tokenizes (`pool.imap`, order-preserving), so download (main) overlaps encode (workers). Writes
    flat uint16 train.bin/val.bin (the first `val_docs` documents are held out to val) and stops at
    EXACTLY `max_tokens` train tokens. Replaces the per-worker-`shard()` design that starved most
    workers and could not reach the token target.

    Set RAYON_NUM_THREADS=1 in the environment so each single-purpose worker uses one core (else
    every worker's tokenizer spawns a full rayon pool and oversubscribes the CPU)."""
    import multiprocessing as mp
    from datasets import load_dataset

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _iter_batches():
        for attempt in range(6):                           # HF tree-listing / streaming 5xx are transient
            try:
                ds = load_dataset("HuggingFaceFW/fineweb-edu", name=subset,
                                  split="train", streaming=True)
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(10 * (attempt + 1))
        texts = []
        for doc in ds:
            texts.append(doc["text"])
            if len(texts) >= batch_docs:
                yield texts
                texts = []
        if texts:
            yield texts

    n_train = n_val = doc_count = 0
    t0 = time.time()
    done = False
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(workers, initializer=_init_worker)
    try:
        with open(out / "train.bin", "wb") as ftr, open(out / "val.bin", "wb") as fva:
            for toks, lens in pool.imap(_encode_batch_worker, _iter_batches(), chunksize=1):
                nb = len(lens)
                if doc_count + nb <= val_docs:                 # whole batch is val
                    toks.tofile(fva); n_val += int(toks.size)
                else:
                    if doc_count < val_docs:                   # batch straddles the val/train boundary
                        split = int(lens[: val_docs - doc_count].sum())
                        toks[:split].tofile(fva); n_val += split
                        toks = toks[split:]
                    if max_tokens is not None and n_train + toks.size >= max_tokens:
                        keep = max_tokens - n_train             # truncate to hit max_tokens exactly
                        toks[:keep].tofile(ftr); n_train += keep
                        done = True
                    else:
                        toks.tofile(ftr); n_train += int(toks.size)
                doc_count += nb
                if done or doc_count % (batch_docs * 20) == 0:
                    el = max(time.time() - t0, 1e-9)
                    print(f"  {doc_count:,} docs, {n_train:,} train / {n_val:,} val tokens, "
                          f"{n_train / el / 1e6:.2f}M tok/s", flush=True)
                if done:
                    break
    finally:
        pool.terminate()                                       # reap workers deterministically
        pool.join()
    print(f"Done (pipeline x{workers}): {n_train:,} train tokens, {n_val:,} val tokens -> {out}",
          flush=True)
    return n_train, n_val


def prepare(out_dir, subset="sample-10BT", max_docs=None, val_docs=10_000, max_tokens=None,
            batch_docs=1024, workers=1):
    """Stream `subset` of FineWeb-Edu -> uint16 train/val .bin. Stops at `max_docs` documents
    (if set) or once `max_tokens` TRAIN tokens are written (if set), whichever comes first.
    For >10B tokens pick a larger subset (`sample-100BT` / `sample-350BT`) and set `max_tokens`.
    `workers>1` runs the overlapped tokenization pipeline (one stream feeds a tokenizer Pool;
    ignores `max_docs`)."""
    if workers > 1:
        return _prepare_pipeline(out_dir, subset, val_docs, max_tokens, batch_docs, workers)
    from datasets import load_dataset

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer()
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=subset, split="train", streaming=True)

    n_train = n_val = 0
    with open(out / "train.bin", "wb") as ftr, open(out / "val.bin", "wb") as fva:
        texts, base = [], 0                                  # batch docs -> multithreaded encode_batch
        stop = False
        for i, doc in enumerate(ds):
            if max_docs is not None and i >= max_docs:
                break
            texts.append(doc["text"])
            if len(texts) < batch_docs:
                continue
            for k, ids in enumerate(tok.encode_batch(texts)):
                ids.append(tok.eot)
                arr = np.asarray(ids, dtype=np.uint16)
                if base + k < val_docs:
                    arr.tofile(fva); n_val += len(ids)
                else:
                    arr.tofile(ftr); n_train += len(ids)
            base += len(texts); texts = []
            print(f"  {base} docs, {n_train:,} train / {n_val:,} val tokens", flush=True)
            if max_tokens is not None and n_train >= max_tokens:   # token budget reached (batch granularity)
                stop = True
                break
        if texts and not stop:                               # flush the final partial batch
            for k, ids in enumerate(tok.encode_batch(texts)):
                ids.append(tok.eot)
                arr = np.asarray(ids, dtype=np.uint16)
                (fva if base + k < val_docs else ftr).write(arr.tobytes())
                n_val += len(ids) if base + k < val_docs else 0
                n_train += 0 if base + k < val_docs else len(ids)
    print(f"Done: {n_train:,} train tokens, {n_val:,} val tokens -> {out}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/fineweb-edu")
    p.add_argument("--subset", default="sample-10BT")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--val-docs", type=int, default=10_000)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="stop after this many TRAIN tokens (e.g. 50_000_000_000 for ~50B)")
    p.add_argument("--batch-docs", type=int, default=1024,
                   help="docs per multithreaded encode_batch call (throughput knob)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel tokenization processes (each streams a disjoint shard; ~Nx faster)")
    a = p.parse_args(argv)
    prepare(a.out, a.subset, a.max_docs, a.val_docs, a.max_tokens, a.batch_docs, a.workers)
    # Native extensions (pyarrow / aiohttp / tiktoken-rayon) can segfault during interpreter
    # finalization with a spawn Pool. The data is fully written and files are closed by now, so
    # exit hard to skip the problematic global teardown and return a clean exit code.
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


# ⚠ Entry points must IMPORT this module, never exec it under the name __main__ (e.g. via
# runpy.run_module(run_name="__main__")). The worker pool uses the "spawn" start method, which
# pickles `_init_worker` by module path -- so under runpy the child looks it up in ITS __main__,
# the launching script, and raises PicklingError for any --workers > 1.
if __name__ == "__main__":
    main()
