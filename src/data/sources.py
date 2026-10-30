"""Data-source registry — the thin mechanism that lets `train.py` stay dataloader-agnostic.

A *data source* fully describes what to train/eval on: it returns the train dataset, the collator, and
any eval callbacks. Each data module OWNS its source (co-located with its data code + eval callback):
e.g. `fineweb_edu.py` registers "fineweb_edu". This file
holds ONLY the registry + `DataBundle` — no source bodies, no heavy imports.

    from .sources import register_data_source, DataBundle
    @register_data_source("my_task")
    def _build(cfg, model, seq_len, n_future) -> DataBundle: ...

`build_trainer` resolves `cfg["data"]["source"]` via `get_data_source`. A built-in source under
`data.<name>` is imported on demand (it self-registers), so nothing hardcodes the list.
"""

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_REGISTRY: dict = {}


@dataclass
class DataBundle:
    train_dataset: Any
    data_collator: Callable
    eval_callbacks: list = field(default_factory=list)   # TrainerCallbacks the source wants added
    eval_dataset: Optional[Any] = None


def register_data_source(name):
    def deco(fn):
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            raise ValueError(f"data source '{name}' already registered")
        _REGISTRY[name] = fn
        return fn
    return deco


def get_data_source(name):
    if name not in _REGISTRY:
        # convention: a built-in source lives at `data.<name>` and self-registers on import.
        # (Sources outside the package register themselves via their own entrypoint import.)
        try:
            importlib.import_module(f".{name}", package=__package__)
        except ImportError:
            pass
    if name not in _REGISTRY:
        raise KeyError(f"unknown data source '{name}'; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
