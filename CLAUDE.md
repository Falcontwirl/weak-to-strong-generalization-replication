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
- Compute: one A100 80GB, 1-2 h budget. Full configs ~60-70 min each for 3 seeds (`--dry-run` estimates,
  uncalibrated; see Status).
- Runs on Brown's Oscar cluster via Slurm: `scripts/oscar/setup.sh` (login node, prefetches weights to
  `~/scratch/hf_cache`) and `scripts/oscar/run.sbatch` (`sbatch --export=ALL,CONFIG=configs/X.yaml ...`).
  Oscar access (checked 2026-09-24): `gpu-he` is refused for this account; `-p gpu` has no H100/A100 gres.
  Largest cards on `-p gpu` are 48 GB class (`l40s`, `nvidia_rtx_a6000`, `nvidia_a40`); a bare
  `--gres=gpu:1` can land on a 24 GB RTX 3090. Always pin the type, e.g. `--gres=gpu:l40s:1`.

## Status (as of 2026-09-24)
- Done: Stage 1, Stage 2, README, Oscar scripts. `pytest -q` passes (30 tests; integration tests run tiny
  Pythia models on CPU, about 1 min). Tiny CPU runs of both experiments produce all artifacts/plots.
- Sanity check (Mac MPS, 2k examples, 1 epoch): 70M weak 73.5%, 160M W2S 78.8%, ceiling 83.0% -> PGR 0.55.
- BLOCKER: every Oscar run so far trained on CPU (`run_metadata.json` `"device": "cpu"`). The venv has
  `torch 2.14.0+cu130`, which needs a CUDA 13 driver; Oscar GPU nodes have driver 12.9, so
  `torch.cuda.is_available()` is False and `device: auto` silently falls back to CPU (4 cores). Fix: install
  the `cu126` wheel (`pip install --force-reinstall torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126`).
  This is why base_smoke took 25 min (87% inside the train loop) vs the ~4 min dry-run estimate.
  The dry-run estimator has never been calibrated on a real GPU.
- Smoke results (all CPU, 1 seed, 1k/1k train, p70m/p160m/p410m), PGR / agreement / error-correction:

  | pair | base_smoke (auto, 2 ep) | base_smoke_final (final, 2 ep) | base_smoke_final_1ep (final, 1 ep) |
  |---|---|---|---|
  | 70m->160m | -0.02 / 0.875 / 0.258 | 0.35 / 0.860 / 0.313 | -0.37 / 0.893 / 0.063 |
  | 160m->410m | -0.43 / 0.881 / 0.229 | 0.24 / 0.881 / 0.333 | n/a (ceiling < weak) |
  | 70m->410m | 0.38 / 0.810 / 0.478 | 0.59 / 0.835 / 0.467 | -0.32 / 0.892 / 0.056 |

  - `final` vs `auto` is NOT a selection effect: `auto` already picked the last step for all W2S students.
    Differences come from run-to-run nondeterminism across nodes (e.g. p70m teacher 0.760 vs 0.755 at the
    same step), amplified by PGR's small denominator (ceiling - weak is only 0.04-0.10). One seed cannot
    separate settings; keep 3 seeds.
  - 1 epoch on 1k examples (32 steps) is undertrained: p70m teacher 0.548, near zero-shot; its students
    are at chance. Uninformative about the full run.
  - Decision for base_full: keep `selection: auto`, `epochs: 2`, 3 seeds.
- Next steps: fix torch -> GPU smoke on a pinned 48 GB card to calibrate timing -> base_full -> chain_full.
  2.8B full fine-tuning (~45 GB weights+AdamW state before activations) will not fit on 48 GB.

## Conventions
- Local env: `.venv` (uv, Python 3.12). Run `.venv/bin/python -m pytest -q`.
- CLI: `python -m src.experiment --config configs/<cfg>.yaml [--dry-run|--force|--seeds ...|--analysis-only]`.
- Plots: matplotlib PNGs in `src/plotting.py`, palette/roles at the top of that file.
- Not implemented (listed as extensions): the paper's auxiliary confidence loss.
