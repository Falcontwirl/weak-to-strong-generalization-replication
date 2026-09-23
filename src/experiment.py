"""Top-level CLI: raw dataset download -> splits -> job graph -> training -> metrics -> plots -> summary.

    python -m src.experiment --config configs/base_smoke.yaml [--force] [--dry-run] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import TEST_ROLE, load_task
from .experiments import get_experiment
from .jobs import RunContext
from .models import verify_models
from .runner import estimate, prepare, run_graph
from .utils import REPO_ROOT, config_hash, load_config, log, resolve_device, run_metadata, write_json, write_yaml


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="recompute jobs even if cached artifacts exist")
    ap.add_argument("--dry-run", action="store_true", help="print the job graph and a rough time estimate")
    ap.add_argument("--seeds", type=int, nargs="+", help="override config seeds")
    ap.add_argument("--output-dir", help="override the output directory")
    ap.add_argument("--analysis-only", action="store_true", help="only (re)build metrics/plots from cached jobs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.seeds:
        cfg["seeds"] = args.seeds
    exp = get_experiment(cfg["experiment"])
    missing = [s for s in exp.required_splits(cfg) if s not in cfg["splits"]["sizes"]]
    if missing:
        raise ValueError(f"experiment {cfg['experiment']} requires split sizes for {missing}")

    root = Path(cfg["output_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    out = Path(args.output_dir) if args.output_dir else root / f"{cfg['name']}-{config_hash(cfg)}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {out}")
    write_yaml(out / "resolved_config.yaml", cfg)

    task = load_task(cfg, out, force=args.force)
    log("Splits: " + ", ".join(f"{k}={len(v)}" for k, v in task.splits.items()))
    device = resolve_device(cfg["device"])
    if device == "cuda":
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ctx = RunContext(cfg=cfg, out_dir=out, task=task, device=device, force=args.force)
    write_json(out / "run_metadata.json", {**run_metadata(cfg, task.info), "device": device,
                                           "experiment": cfg["experiment"], "config_path": str(args.config)})

    jobs = []
    for i, seed in enumerate(cfg["seeds"]):
        jobs += exp.build_jobs(cfg, seed, first=(i == 0))
    jobs = prepare(jobs, ctx)
    assert all(j.train_split != TEST_ROLE for j in jobs)

    if args.dry_run:
        estimate(jobs, ctx)
        return None
    if not args.analysis_only:
        log("Verifying model ids...")
        verify_models(cfg["models"])
        results = run_graph(jobs, ctx)
    else:
        from .jobs import JobResult, job_dir
        from .utils import read_json

        results = {}
        for j in jobs:
            d = job_dir(ctx, j)
            status = "done" if (d / "done.json").exists() else "skipped"
            results[j.name] = JobResult(j, d, status, read_json(d / "done.json") if status == "done" else {},
                                        None if status == "done" else "not run")
    exp.analyze(ctx, results)
    log(f"\nWrote {out / 'summary.md'}")
    return out


if __name__ == "__main__":
    main()
