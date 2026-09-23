"""Resolve a job graph (dependencies via ``label_source``), hash jobs, execute with failure isolation."""
from __future__ import annotations

import gc
import traceback

import torch

from .data import TEST_ROLE
from .jobs import VAL_ROLE, Job, JobResult, RunContext, job_dir, run_job, selection_mode
from .utils import log, stable_hash, write_json

# Train-config keys that don't change results and therefore don't enter the cache key.
_NON_SEMANTIC_TRAIN_KEYS = {"save_checkpoints", "eval_batch_size", "num_workers"}


def prepare(jobs: list[Job], ctx: RunContext) -> list[Job]:
    """Validate, register label requests on teachers, topologically sort and hash jobs."""
    by = {}
    for j in jobs:
        if j.name in by:
            raise ValueError(f"duplicate job name {j.name}")
        by[j.name] = j
    for j in jobs:
        if j.train_split == TEST_ROLE:
            raise AssertionError(f"{j.name}: training on the test split is forbidden")
        if j.is_weak:
            if j.label_source not in by:
                raise ValueError(f"{j.name}: unknown label_source {j.label_source}")
            src = by[j.label_source]
            if src.kind != "train":
                raise ValueError(f"{j.name}: label source must be a trained job")
            needed = [j.train_split]
            if selection_mode(j, ctx.cfg["train"]) == "val_teacher":
                needed.append(VAL_ROLE)
            src.label_splits = tuple(dict.fromkeys([*src.label_splits, *needed]))

    ordered, state = [], {}

    def visit(j: Job):
        if state.get(j.name) == 2:
            return
        if state.get(j.name) == 1:
            raise ValueError(f"cycle at {j.name}")
        state[j.name] = 1
        if j.is_weak:
            visit(by[j.label_source])
        state[j.name] = 2
        ordered.append(j)

    for j in jobs:
        visit(j)
    for j in ordered:
        j.key = job_key(j, ctx, by)
    return ordered


def job_key(j: Job, ctx: RunContext, by: dict[str, Job]) -> str:
    task = ctx.task
    mcfg = {k: v for k, v in ctx.model_cfg(j.model).items() if k != "name"}
    payload = {
        "kind": j.kind, "model": mcfg, "seed": j.seed, "test": task.split_hash(TEST_ROLE),
        "dataset": {k: ctx.cfg["dataset"][k] for k in ("path", "name", "text_field", "label_field", "label_names")},
    }
    if j.kind == "zero_shot":
        payload["zero_shot"] = ctx.cfg["zero_shot"]
        payload["max_length"] = ctx.cfg["train"]["max_length"]
    else:
        payload["train"] = {k: v for k, v in ctx.cfg["train"].items() if k not in _NON_SEMANTIC_TRAIN_KEYS}
        payload["selection"] = selection_mode(j, ctx.cfg["train"])
        payload["train_split"] = task.split_hash(j.train_split)
        payload["val"] = task.split_hash(VAL_ROLE)
        payload["source"] = by[j.label_source].key if j.is_weak else "gt"
    return stable_hash(payload, 10)


def run_graph(jobs: list[Job], ctx: RunContext) -> dict[str, JobResult]:
    results: dict[str, JobResult] = {}
    for i, j in enumerate(jobs):
        log(f"\n=== job {i + 1}/{len(jobs)}: {j.name}")
        src = results.get(j.label_source) if j.is_weak else None
        if j.is_weak and (src is None or src.status != "done"):
            results[j.name] = JobResult(j, job_dir(ctx, j), "skipped", error=f"dependency {j.label_source} not done")
            log(f"[skip] {j.name}: dependency {j.label_source} failed")
            continue
        try:
            results[j.name] = run_job(j, ctx, src)
        except Exception as e:  # noqa: BLE001 - failures are logged and reported, never silently dropped
            tb = traceback.format_exc()
            log(f"[FAILED] {j.name}: {e}\n{tb}")
            results[j.name] = JobResult(j, job_dir(ctx, j), "failed", error=f"{type(e).__name__}: {e}")
            if isinstance(e, AssertionError) and "eakage" in str(e):
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    bad = [{"job": r.job.name, "status": r.status, "error": r.error} for r in results.values() if r.status != "done"]
    write_json(ctx.out_dir / "failures.json", bad)
    return results


# Rough A100 throughput assumptions for --dry-run estimates only.
_TOKENS_PER_EXAMPLE = 48
_EFFECTIVE_FLOPS = 60e12


def estimate(jobs: list[Job], ctx: RunContext) -> float:
    total = 0.0
    log(f"{'job':50s} {'model':>8s} {'n':>6s} {'est_s':>7s}")
    for j in jobs:
        mcfg = ctx.model_cfg(j.model)
        params = float(mcfg.get("params", 1e8))
        if j.kind == "zero_shot":
            n, flops = len(ctx.task.splits[TEST_ROLE]), 2 * params * _TOKENS_PER_EXAMPLE
            secs = n * flops / _EFFECTIVE_FLOPS + 15
        else:
            n = len(ctx.task.splits[j.train_split])
            epochs = mcfg.get("epochs", ctx.cfg["train"]["epochs"])
            n_eval = len(ctx.task.splits[VAL_ROLE]) * ctx.cfg["train"]["evals_per_epoch"] * epochs
            n_eval += len(ctx.task.splits[TEST_ROLE]) + sum(len(ctx.task.splits[s]) for s in j.label_splits)
            secs = (6 * params * _TOKENS_PER_EXAMPLE * n * epochs + 2 * params * _TOKENS_PER_EXAMPLE * n_eval) \
                / _EFFECTIVE_FLOPS + 20 + params / 1e8
        total += secs
        log(f"{j.name:50s} {j.model:>8s} {n:6d} {secs:7.0f}")
    log(f"Estimated total on A100: {total / 60:.1f} min for {len(jobs)} jobs (rough; tokens/example={_TOKENS_PER_EXAMPLE})")
    return total
