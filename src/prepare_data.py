"""Tokenize FineWeb-Edu into the flat uint16 train.bin / val.bin the trainer memory-maps.

    python -m src.prepare_data --out data/fineweb-edu --subset sample-10BT --workers 16

`--subset sample-10BT` (~10B tokens, ~20 GB) is enough to train the 167m_fineweb config end to
end; cap it with `--max-tokens` for a smaller run. The 31.5B-token recipe in configs/ wants
`--subset sample-100BT --max-tokens 35_000_000_000`.

`--workers` should match your CPU quota, not nproc: one streaming reader in the main process feeds a
pool of tokenize-only workers, so the job is CPU-bound and over-subscribing just thrashes.

A thin entry point; the pipeline is `src/data/fineweb_edu.py`. It IMPORTS that module rather
than exec'ing it -- see the note there on spawn and pickling.
"""
from .data.fineweb_edu import main

if __name__ == "__main__":
    main()
