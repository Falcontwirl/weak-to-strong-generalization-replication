# Project context for Claude Code

Replication of Burns et al. 2023 "Weak-to-Strong Generalization" plus a daisy-chain extension
(GT -> M1 -> M2 -> M3 -> M4 vs direct M1 -> M4). Original spec: `docs/spec.docx`. User-facing docs: `README.md`.

## Decisions already made (don't re-litigate)
- Two stages, built in order: Stage 1 = modular base W2S experiment (`src/experiments/base_w2s.py`);
  Stage 2 = daisy-chain controls (`src/experiments/daisy_chain.py`) built on the same core without modifying it.
- Everything is a `Job(model, train_split, label_source)` (`src/jobs.py`); `src/runner.py` resolves the graph,
  registers label requests on teachers, content-hashes jobs (teacher hash is recursive), caches/resumes.
- Dataset: SST-2 (GLUE); official validation (872) = final test; all other splits carved from train with
  fixed `split_seed`. Ladder: Pythia 160M/410M/1B/2.8B, full fine-tuning, fp32 weights + bf16 autocast,
  zero-initialized last-token linear head. Soft-label CE is the default loss.
- Weak students select checkpoints on *teacher-labeled* val (no GT leakage); GT jobs on GT val.
- Compute: one A100 80GB, 1-2 h budget. Full configs ~60-70 min each for 3 seeds (`--dry-run` estimates).
- Runs on Brown's Oscar cluster via Slurm: `scripts/oscar/setup.sh` (login node, prefetches weights to
  `~/scratch/hf_cache`) and `scripts/oscar/run.sbatch` (`sbatch --export=ALL,CONFIG=configs/X.yaml ...`).
  Which Oscar partition has A100s is unverified; user checks with `sinfo -o "%P %G %f"`.

## Status (as of 2026-09-23)
- Done: Stage 1, Stage 2, README, Oscar scripts. `pytest -q` passes (30 tests; integration tests run tiny
  Pythia models on CPU, about 1 min). Tiny CPU runs of both experiments produce all artifacts/plots.
- Sanity check (Mac MPS, 2k examples, 1 epoch): 70M weak 73.5%, 160M W2S 78.8%, ceiling 83.0% -> PGR 0.55.
- NOT yet done: any GPU run. Next steps: base_smoke -> base_full -> chain_full (`--seeds 0` if short on time),
  then interpret `outputs/<config>-<hash>/summary.md`.

## Conventions
- Local env: `.venv` (uv, Python 3.12). Run `.venv/bin/python -m pytest -q`.
- CLI: `python -m src.experiment --config configs/<cfg>.yaml [--dry-run|--force|--seeds ...|--analysis-only]`.
- Plots: matplotlib PNGs in `src/plotting.py`, palette/roles at the top of that file.
- Not implemented (listed as extensions): the paper's auxiliary confidence loss.
