# Experiments

Every run so far, in order. All GPU runs used one NVIDIA L40S (48 GB) on Brown's Oscar cluster, 3 seeds
(0, 1, 2), 2 epochs, soft-label cross-entropy, and checkpoint selection on the supervisor-labeled
validation set. Accuracies are test accuracy, mean over seeds. PGR = (student - teacher) / (ceiling - teacher).
Full reports are in `outputs/<run>/summary.md` (not in git); the write-up figures are in `docs/figures/`.

## Main experiments (GPU)

| # | run (config) | dataset | what it tested | result | what it accomplished |
|---|---|---|---|---|---|
| E1 | `base_full_48gb` | SST-2 | Base weak-to-strong on Pythia 160M, 410M, 1B, 1.4B (6 pairs) | PGR from the 160M teacher: 0.55 / 0.45 / 0.27 into 410M / 1B / 1.4B. Stronger teachers: gains ~0, because SST-2 saturates around 0.90 | First real replication: a weak teacher's students recover part of the gap to the ceiling |
| E2 | `chain_full_48gb` | SST-2 | Daisy chain 160M -> 410M -> 1B -> 1.4B vs direct supervision, with true-label upper bounds | Chain = direct at 1.4B (0.863 vs 0.863, 95% CI [-0.026, +0.024]). Per-link gain shrinks 1.5 / 0.8 / 0.6 pts. Chain errors far more correlated (phi 0.57-0.77) than independently trained models (0.33-0.48) | No detectable chain advantage; the chain propagates its specific errors |
| E3 | `base_full_48gb_conf` | SST-2 | Auxiliary confidence loss (alpha 0.75 for every student, linear 20% warmup) | Worse than naive in 17 of 18 paired students; 160M -> 410M fell to 0.758 (naive 0.850), below its own 0.835 teacher | Showed the conf loss, as configured, hurts |
| E4 | `base_full_2p8b` | SST-2 | Adds Pythia 2.8B (8-bit AdamW to fit 48 GB); 10 pairs | 2.8B ceiling 0.925. 160M -> 2.8B: 0.835 -> 0.875 (PGR 0.45 +/- 0.08), significant in all 3 seeds | Best result: the largest gain so far; confirmed 2.8B trains on an L40S |
| E5 | `base_full_2p8b_conf` | SST-2 | Conf loss (as in E3) with 2.8B students | Still hurts: 160M -> 2.8B 0.857 vs naive 0.875 | Conf-loss damage is not specific to small students |
| E6 | `boolq_base` | BoolQ | Base weak-to-strong on a harder task | 160M teacher always answers "yes" (0.622 = majority class); its students copy it. Ceilings only 0.63-0.71 | Uninformative: BoolQ is too hard at this scale, and was not class-balanced as in the paper |
| E7 | `boolq_base_conf` | BoolQ | Conf loss on BoolQ | Students 0.52-0.57, below the 0.622 majority baseline in every pair | Conf loss plus unbalanced data amplifies noise |
| E8 | `chain_boolq` | BoolQ | Daisy chain on BoolQ (2,000 examples per generation) | M1 always predicts "yes", so every chain and direct student is constant too | Degenerate; no information about chaining |
| E9 | `base_full_2p8b_conf_ref` | SST-2 | Conf loss with the released code's settings (alpha 0.5, step schedule) | Damage gone (+2.6 pts vs E5 on average) but no gain over naive (-0.3 pts) | The E3/E5/E7 damage came from our alpha and schedule; the conf loss is neutral on SST-2 |
| E10 | `base_full_48gb_conf_a0` | SST-2 | Sanity check: conf loss with alpha 0 (should equal naive) | All 18 students identical to E1 (max diff 0.0000) | Confirms the conf-loss code is wired correctly |

## Setup and smoke runs

| run | hardware | what it tested | outcome |
|---|---|---|---|
| `base_smoke` | CPU, then L40S | Small smoke test (70M / 160M / 410M, 1k examples) | CPU runs exposed that torch fell back to CPU (CUDA 13 wheel vs driver 12.9); fixed with the cu126 wheel. GPU rerun (3 seeds) showed that seed noise, not bf16, explained a weak 70M teacher |
| `base_smoke_final`, `base_smoke_final_1ep` | CPU, 1 seed | Last-checkpoint selection and 1 epoch vs 2 | Differences were run-to-run noise; 1 epoch undertrains. Kept 2 epochs, best-on-val selection, 3 seeds |
| `base_full` | CPU | Original 160M-2.8B config written for an A100 | No jobs completed; replaced by `base_full_48gb` because Oscar has no A100 for this account |

## Known deviations from the paper

- No class balancing (the paper rebalances datasets above 55% majority and balances the test set). Mild for SST-2 (~56% positive); it made the BoolQ runs degenerate.
- Zero-initialized classification head (the paper initializes it from the "0"/"1" unembedding rows). Effect untested.
- Conf loss in E3/E5/E7 used alpha 0.75 for every student (the paper uses 0.75 only for its largest students, 0.5 otherwise).
- Model sizes: Pythia 160M-2.8B instead of the paper's GPT-2 to GPT-4-scale models.
