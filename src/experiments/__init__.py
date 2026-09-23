"""Experiment definitions. Each module exposes REQUIRED_SPLITS, build_jobs(cfg, seed, first) and analyze(...)."""
import importlib


def get_experiment(name: str):
    if name not in ("base_w2s", "daisy_chain"):
        raise ValueError(f"unknown experiment '{name}'")
    return importlib.import_module(f"{__name__}.{name}")
