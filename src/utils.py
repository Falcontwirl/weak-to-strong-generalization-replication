"""Config loading, seeding, hashing, metadata, and small I/O helpers."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS: dict[str, Any] = {
    "experiment": "base_w2s",
    "name": None,
    "seeds": [0],
    "device": "auto",
    "output_root": "outputs",
    "dataset": {
        "path": "nyu-mll/glue",
        "name": "sst2",
        "text_field": "sentence",
        "label_field": "label",
        "label_names": ["negative", "positive"],
        "train_split": "train",
        # GLUE test labels are hidden, so the official labeled validation split is the final test set.
        "test_split": "validation",
        "test_limit": None,
    },
    "splits": {"split_seed": 1234, "sizes": {}},
    "models": [],
    "train": {
        "epochs": 2,
        "max_length": 128,
        "warmup_frac": 0.1,
        "weight_decay": 0.0,
        "evals_per_epoch": 2,
        "loss": "soft_ce",  # soft_ce | hard_ce
        # model selection on the val split: 'auto' = GT val for GT jobs, teacher-labeled val for weak jobs.
        "selection": "auto",  # auto | val_gt | val_teacher | final
        "eval_batch_size": 64,
        "grad_accum": 1,
        "save_checkpoints": False,
        "num_workers": 0,
    },
    "zero_shot": {
        "enabled": True,
        "prompt": "Review: {text}\nSentiment:",
        "verbalizers": [" negative", " positive"],
    },
    "analysis": {"n_bootstrap": 10000, "n_conf_bins": 5, "ece_bins": 15},
    "base_w2s": {"pairs": "all"},
    "daisy_chain": {},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, raw)
    if not cfg["name"]:
        cfg["name"] = path.stem
    names = [m["name"] for m in cfg["models"]]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate model names in ladder: {names}")
    return cfg


def stable_hash(obj: Any, n: int = 10) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def config_hash(cfg: dict) -> str:
    """Hash of the config excluding seeds and output location, so adding seeds reuses the same run dir."""
    c = {k: v for k, v in cfg.items() if k not in ("seeds", "output_root")}
    return stable_hash(c, 8)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str = "auto") -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_yaml(path: str | Path, obj: Any) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def run_metadata(cfg: dict, dataset_info: dict | None = None) -> dict:
    import importlib.metadata as md

    import torch

    pkgs = {}
    for p in ["torch", "transformers", "datasets", "peft", "numpy", "pandas", "accelerate"]:
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pkgs[p] = None
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": pkgs,
        "git": git_info(),
        "gpu": gpu,
        "seeds": cfg["seeds"],
        "models": {m["name"]: m["id"] for m in cfg["models"]},
        "dataset": dataset_info,
    }


def log(msg: str) -> None:
    print(msg, flush=True)
