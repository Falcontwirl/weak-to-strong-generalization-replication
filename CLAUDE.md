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
- 2026-09-24: torch reinstalled as 2.14.0+cu126 (BLOCKER fix applied, not yet verified on a GPU node);
  run.sbatch now exits if torch.cuda.is_available() is False. GPU smoke submitted as job 6677798
  (base_smoke, --force, l40s, 30 min): VERIFIED device=cuda on L40S, whole run 1:17 wall (vs 25 min CPU).
  But p70m GT teacher 0.604 on GPU vs ~0.76 on CPU, same config -> 70m->* pairs collapse (PGR -0.33, 0.06);
  160m->410m PGR 0.80. Unresolved whether seed noise or bf16 autocast; 3-seed smoke submitted as job 6680034
  (--seeds 0 1 2, reuses GPU seed 0). If p70m teacher stays ~0.6 on all seeds, test with autocast off.
- Job 6680034 result: seed noise, not bf16. p70m GT teacher on GPU = 0.604 / 0.627 / 0.734 (seeds 0/1/2),
  close to the CPU ~0.76 (1 seed). The smoke config (1k ex, 64 steps) is just noisy. Run a 3-seed
  autocast-off smoke only if base_full's weakest teacher (p160m) is still erratic across seeds.
  PGR is per-seed then nan-averaged: seeds with ceiling <= weak are dropped, so a PGR with no ± may be 1 seed.
- 2.8B cannot be fully fine-tuned on 48 GB -> `configs/base_full_48gb.yaml` (160M/410M/1B/1.4B, 6 pairs,
  3 seeds, full FT). pythia-1.4b prefetched 2026-09-24. Submitted as job 6681614 (l40s, 2 h limit).
- 6681614 DONE (25:18 wall, 40 jobs, all ok) -> outputs/base_full_48gb-1b976761/summary.md. PGR (3 seeds):
  160m->410m 0.55, 160m->1b 0.45, 160m->1.4b 0.27; pairs with 410m/1b as weak have ceiling-weak gaps of
  0.5-3 pts, so their PGR is noise (-0.91 +/- 1.99 etc). SST-2 saturates ~0.9. p410m GT ceiling (strong_train)
  0.862 < GT teacher (weak_train) 0.882 on all seeds: small split effect, not the smoke-run noise.
- chain_full_48gb submitted 2026-09-25 as job 6698279 (l40s, 2 h); log logs/w2s-6698279.out. Risk: M2->M3->M4 links
  had ~0 W2S gain in base run (SST-2 saturates), so chain vs direct may be within noise.
  6698279 DONE (22:34, 34 jobs) -> outputs/chain_full_48gb-6faef36c/summary.md. Chain end p1.4b 0.863 = direct
  p160m->p1.4b 0.863 (diff +0.000, CI [-0.026, +0.024]): no chain advantage. Per-link gain shrinks 1.5/0.8/0.6 pts.
  Clear effect: chain errors far more correlated (phi 0.57-0.77) than GT-trained same sizes (0.33-0.48).
- 2026-09-25: added (see Experiment log) the auxiliary confidence loss (`train.conf_loss`), 8-bit AdamW for
  p2.8b (`optimizer: adamw8bit`, bitsandbytes 0.50.2 installed with --no-deps to keep torch cu126), BoolQ
  support (`dataset.text_template`, `train.truncation_side`), and `--reuse-from RUN_DIR`. New keys enter
  job cache keys only when set (conf_loss only for weak jobs), so all 74 existing jobs keep their keys (verified).

## Experiment log
One row per GPU run (all l40s, 3 seeds unless noted). Results live in `outputs/<dir>/summary.md`.
Dry-run time estimates assume 48 tokens/example (SST-2-like); BoolQ inputs are ~5x longer, expect 3-5x the estimate.

| # | config | job | status | reuses | key result |
|---|---|---|---|---|---|
| E1 | base_full_48gb (SST-2, naive, 160M-1.4B) | 6681614 | done 25 min | - | PGR 160m->410m/1b/1.4b = 0.55/0.45/0.27; stronger teachers: gain ~0 (saturated ~0.90) |
| E2 | chain_full_48gb (SST-2 chain) | 6698279 | done 23 min | - | chain = direct at p1.4b (0.863 vs 0.863); chain errors more correlated (phi 0.57-0.77 vs 0.33-0.48) |
| E3 | base_full_48gb_conf (SST-2, conf loss) | 6723025 | done 17 min | E1 GT jobs (22) | conf HURTS: worse than naive in 17/18 paired W2S jobs; 160m->410m 0.758 vs naive 0.850 (below 0.835 teacher); PGR 160m->1b -0.05, ->1.4b 0.24 |
| E4 | base_full_2p8b (SST-2, + p2.8b adamw8bit) | 6723026 | done 30 min; 2.8B fits L40S (~105 s/job) | E1 (40) | p2.8b ceiling 0.925. W2S into p2.8b: from 160m 0.875 (PGR 0.45 +/- 0.08), 410m 0.892 (0.11 +/- 0.42), 1b 0.903 (0.40 +/- 0.36), 1.4b 0.915 (0.70 +/- 0.53). 160m-teacher PGR by student 410m/1b/1.4b/2.8b = 0.55/0.45/0.27/0.45: not monotone |
| E5 | base_full_2p8b_conf | 6723029 | done 22 min (hold came too late) | E3 + E4 | conf hurts at 2.8B too: 160m->2.8b 0.857 vs naive 0.875; 1b->2.8b 0.878 vs 0.903 |
| E6 | boolq_base (BoolQ, naive, 160M-1.4B) | 6723027 | done 72 min | - | UNINFORMATIVE: p160m GT teacher = 0.622 +/- 0.000 = majority class (always "yes"); its students copy it exactly (agree 1.000). GT ceilings only 0.634/0.690/0.712 (1.4b +/- 0.058). 410m/1b teachers: students gain nothing (PGR -0.19..0.02) |
| E7 | boolq_base_conf | 6723030 | done 40 min | E6 GT jobs | conf students 0.52-0.57, BELOW the 0.622 majority baseline in every pair |
| E8 | chain_boolq (BoolQ chain, 2k/generation) | 6723028 | done 37 min | - | DEGENERATE: M1 (160m) always predicts "yes" (0.622), so every chain and direct student is constant too (phi 1.000). No information about chaining |
| E9 | base_full_2p8b_conf_ref (SST-2, conf with openai/weak-to-strong settings: alpha 0.5, `schedule: ref`) | 6728955 | done 36 min | E4 (29) | ref settings REMOVE the damage (vs E5: +2.6 pts mean, up to +12.6) but give NO gain over naive E4 (-0.3 pts mean, max |diff| 3.0, 20/30 slightly worse). 160m->p2.8b 0.875 = naive 0.875 |
| E10 | base_full_48gb_conf_a0 (SST-2, conf alpha 0 = naive, sanity check) | 6728956 | done 15 min | E1 (22) | PASS: all 18 W2S accuracies identical to E1 (max diff 0.0000); conf wiring is correct |

Conf-loss caveats: alpha 0.75 / 20% linear warmup are from memory of the paper's appendix (not verified);
W2S jobs still select checkpoints on teacher-labeled val (`selection: auto`), which may partly undo the
conf loss's disagreement with the teacher; check `best_step` in train_log.json.

## Next steps
- Conf loss (E3/E5/E7) consistently hurts. On BoolQ its students (0.52-0.57) are near what forced
  62%-positive predictions with an uninformative ranking would give (0.62^2 + 0.38^2 = 0.53), i.e. the loss
  amplifies the student's own noise. Suspects, not verified: (a) alpha 0.75 from memory, too strong for small
  students; (b) zero-initialized head -> all student logits start equal, so early hardened targets are
  arbitrary while alpha ramps from step 0; (c) per-batch (32) threshold noise; (d) an implementation bug.
  Before any more conf runs: compare against the paper's released code, try alpha 0.25-0.5 and starting the
  ramp after warmup, on one SST-2 pair with 3 seeds.
- 2026-09-26 code comparison vs openai/weak-to-strong (weak_to_strong/loss.py `logconf_loss_fn`, copy in
  ~/scratch/claude_tmp/w2s_ref). SAME: mixed-target CE, class-balance threshold (equivalent form), soft weak
  labels, threshold over the full batch of 32, zero-init head (`normal_(std=0.0)`). DIFFERENT: (1) aux_coef 0.5
  (repo default, used by train_weak_to_strong.py) vs our 0.75; (2) schedule: repo coef = aux*step_frac for the
  first 10% (<= 0.05) then jumps to aux; ours ramps linearly to alpha over 20% (0.375 already at 10%);
  (3) repo optimizer Adam + cosine, no LR warmup, vs our AdamW(wd 0) + 10% warmup/linear decay (all jobs).
  Not a factor: checkpoint selection (16/18 conf students picked step 312/314, 2 the final step). No outright
  bug found; the repo is the open-source GPT-2 replication, so its defaults may differ from the paper's GPT-4 runs.
  Note: models.py/README say the paper inits the head from unembedding rows; the released code zero-inits.
  Added `conf_loss.schedule: ref` (default `linear` unchanged, so E3/E5/E7 keys are unchanged); 39 tests pass.
  E9/E10 results (2026-09-26): E10 reproduces E1 exactly -> no wiring bug. E9 -> our alpha 0.75 + early linear
  ramp caused the damage in E3/E5/E7; with the repo's settings the conf loss is neutral at 160M-2.8B on SST-2.
- 2026-09-28 checked against the PAPER itself (arXiv 2312.09390; PDF + text in ~/scratch/claude_tmp/w2s_ref):
  - Tasks: Table 1 lists 22 NLP datasets incl. GLUE SST-2 and BoolQ (both in the paper). Released code has
    only amazon_polarity, sciq, anthropic_hh, cosmos_qa, boolq (no SST-2).
  - NEW DEVIATION: the paper rebalances any dataset whose majority class is > 55% (drops dominant-class
    examples) and balances the test set, so chance = 50% everywhere. We did not: BoolQ is 62% "yes" (the
    160m teacher's always-"yes" = 0.622 would be 0.50 in the paper's setup); SST-2 train is ~56% positive.
  - Conf loss (App. A.4): alpha_max 0.75 for the LARGEST students, 0.5 otherwise, LINEAR warmup 0 -> alpha_max
    over the first 20%; threshold t so exactly half the batch is predicted positive (fine on balanced data).
    So E3/E5/E7's schedule WAS the paper's; the deviation was alpha 0.75 for all students. The released
    code's settings (E9: 0.5, step schedule over 10%) differ from the paper. E3 vs E9 changed alpha and
    schedule together, so which one caused the damage is NOT separated. Paper says the loss had "small or
    neutral effect" for most pairs/datasets and big gains in a few.
  - Head init: paper inits from unembedding rows of "0"/"1" (README right); ours and the released code zero-init.
  - Checkpoint selection: paper early-stops on weak-label val accuracy (= ours); released code uses final.
  - 2 epochs, batch 32, soft weak labels, weak/strong halves of the data: same as ours.
- BoolQ at this scale (4k examples, 2 epochs, lr as SST-2) is too hard: the 160m teacher never beats
  "always yes" and ceilings are 0.63-0.71. Options: more data/epochs or lr tuning for BoolQ, drop p160m as the
  bottom rung (start the ladder at p410m), or pick a task of intermediate difficulty.
- E4 is the best new result: 2.8B works, and the 160m->2.8b gain (+4.0 pts, PGR 0.45) is the largest so far.

## Conventions
- Local env: `.venv` (uv, Python 3.12). Run `.venv/bin/python -m pytest -q`.
- CLI: `python -m src.experiment --config configs/<cfg>.yaml [--dry-run|--force|--seeds ...|--analysis-only|--reuse-from DIR]`.
- Plots: matplotlib PNGs in `src/plotting.py`, palette/roles at the top of that file.
- Tests need `HF_HOME=$HOME/scratch/hf_cache` (tiny Pythia 14m/31m are cached there).
