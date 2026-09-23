# Weak-to-strong generalization: replication + daisy chain

A small, transparent research codebase that

1. **replicates the core experiment of Burns et al. (2023), *Weak-to-Strong Generalization*.** A small model trained on ground truth labels data for a larger model. How much of the gap between the weak supervisor and the strong model's ground-truth ceiling does the strong student recover?
2. **extends it to a daisy chain.** If M1 supervises M2, M2 supervises M3, and M3 supervises M4, does weak-to-strong gain compound, plateau or degrade? How does that compare to supervising M4 directly with M1?

The weak-to-strong setup is an analogue of *scalable oversight*: humans (weak) supervising models that are more capable than they are. A chain of successively stronger models is one proposed route to bootstrapping such supervision. This repo measures, on a controlled classification task, whether that route keeps its gains.

## Research setup

**Task.** SST-2 binary sentiment (GLUE). GLUE's test labels are hidden, so the **official validation split (872 examples) is the final held-out test set**. Every training, labeling and model-selection split is carved from the train split.

**Model ladder.** Pythia 160M → 410M → 1B → 2.8B (EleutherAI). One family with the same pretraining data and tokenizer, so capacity is the main variable. Each model is fine-tuned in full with a linear classification head on the last token, as in the paper. The head is zero-initialized (the paper initializes it from the unembedding rows for tokens "0"/"1").

**Soft labels.** By default students train on the teacher's full probability vector (`soft_ce`: cross-entropy against the teacher distribution, which equals KL up to a constant). Setting `train.loss: hard_ce` trains on argmax labels instead.

**No ground-truth leakage.**
- Split roles are disjoint by construction and asserted at load time, and the test set is never trained on, labeled or used for selection (asserted).
- A weakly supervised student's dataset holds only `input_ids` and `target_probs`. Ground truth is stored in label artifacts for analysis only, and a test asserts it never reaches a training batch.
- Checkpoint selection for weakly supervised students uses **teacher-labeled** validation data. Ground-truth-trained jobs select on ground-truth validation. Set `train.selection` to `val_gt` or `final` for ablations.

### Stage 1: base weak-to-strong (`experiment: base_w2s`)

For every pair (weak W < strong S) in the ladder (6 pairs by default):

| quantity | definition |
|---|---|
| weak performance | W trained on ground-truth labels of `weak_train`; test accuracy |
| W2S performance | S trained on `strong_train` labeled by that W; test accuracy |
| strong ceiling | S trained on ground-truth labels of `strong_train`; test accuracy |
| **PGR** | (W2S − weak) / (ceiling − weak). NaN and flagged when ceiling ≤ weak |

There is also a zero-shot baseline for each model (prompt plus next-token probability of " negative"/" positive").

### Stage 2: daisy chain (`experiment: daisy_chain`)

```
 Ground truth
      |
      v
     M1 -----> labels D2 -----> M2 -----> labels D3 -----> M3 -----> labels D4 -----> M4
      |                         |                         |                         |
      +-------------------------+-------------------------+-------------------------+
                               held-out ground-truth evaluation
```

| condition | what trains | on |
|---|---|---|
| seed (M1) | M1 on ground-truth labels | `seed_true_train` |
| **chain** | Mk on labels from chain-M(k−1) | `g_k` (fresh, disjoint) |
| **direct** | Mj on labels from chain-Mi, with j − i ≥ 2 (M1→M3, M1→M4, M2→M4) | `g_j` |
| **GT upper** | Mk on ground-truth labels | `g_k` (the same examples, so matched) |
| zero-shot | no task training | none |
| teacher eval | every teacher on test and on the exact split it labels | none |

Chain-M4, direct M1→M4, direct M2→M4 and GT-M4 all train on **the same `g4` examples**, so the final comparison is fully matched.

### Metrics

Every student is compared with its teacher on the test set. The metrics are:
- accuracy, W2S gain (student − teacher), PGR (against the immediate teacher, and in the chain also **PGR vs M1**)
- agreement
- accuracy given the teacher was right or wrong
- **error inheritance** (share of teacher errors copied), **error correction** (share fixed) and **new-error rate** (share of teacher-correct examples broken)
- Brier score and ECE (15 bins)
- correction rate by teacher-confidence bin

The chain adds three more analyses:
- per-example C/W trajectories
- consecutive-generation transitions (C→C, W→C, C→W, W→W)
- error correlation (φ, κ, P(err_b | err_a)) among chain generations, **compared with the same statistic for independently GT-trained models of the same sizes**. This control separates "errors propagate along the chain" from "models of these sizes naturally fail on the same hard examples".

Statistics: mean ± std over seeds, and paired bootstrap 95% CIs (10k resamples of test examples) for (a) student vs immediate teacher, (b) chain-MN vs direct→MN, and (c) every student vs its GT upper bound.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
On the GPU machine, `bash scripts/setup_remote.sh` does this, prints the CUDA device, and runs the fast unit tests.

## Exact commands

```bash
# tests (unit + tiny CPU end-to-end runs of both experiments; ~1 min, downloads tiny Pythia models)
python -m pytest -q
python -m pytest -q -m "not integration"      # unit tests only

# laptop / CPU: full pipeline on tiny models (numbers are meaningless; checks the plumbing)
python -m src.experiment --config configs/base_tiny_cpu.yaml
python -m src.experiment --config configs/chain_tiny_cpu.yaml

# A100: Stage 1
bash scripts/run_base_smoke.sh               # ~3-5 min, 70M/160M/410M, 1 seed
bash scripts/run_base_full.sh --dry-run      # prints the job list + time estimate
bash scripts/run_base_full.sh                # ~60-70 min, 160M..2.8B, 6 pairs, 3 seeds

# A100: Stage 2
bash scripts/run_chain_smoke.sh              # ~3-5 min, 3-model chain, 1 seed
bash scripts/run_chain_full.sh               # ~60 min, 4-model chain + controls, 3 seeds
```

Every script is `python -m src.experiment --config <cfg>` plus the arguments you pass. The CLI flags are:

| flag | effect |
|---|---|
| `--dry-run` | print the job graph and a rough A100 time estimate |
| `--seeds 0` | override the seeds (for example, one seed first to fit the budget) |
| `--force` | recompute cached jobs |
| `--analysis-only` | rebuild metrics, plots and summary from cached jobs |
| `--output-dir DIR` | write to DIR instead of the default output directory |

**Fitting a 1–2 h A100 budget.** Each full config is about 1 hour for 3 seeds; the estimates are rough. The first run also downloads about 8 GB of weights. Recommended order:
1. `run_base_smoke.sh`, then read its `summary.md`.
2. `run_base_full.sh`.
3. `run_chain_full.sh --seeds 0` if time remains.

Runs resume: interrupting and re-running reuses every finished job. Adding seeds later (for example `--seeds 0 1 2`) trains only the new ones.

## Output structure

`outputs/<config-name>-<config-hash>/` (the hash excludes seeds, so adding seeds reuses the directory):

```
resolved_config.yaml    run_metadata.json (packages, git commit, GPU, dataset fingerprint, seeds)
split_indices.json      example ids for every role, including test
metrics.csv             tidy: one row per seed / condition / model / teacher
metrics_summary.csv     mean/std over seeds
bootstrap.csv           paired bootstrap CIs
failures.json           failed / skipped jobs (also listed in summary.md; never silently dropped)
summary.md              auto-generated report
confidence_bins.{csv,png}
base_w2s:     w2s_accuracy.png  pgr_grid.png
daisy_chain:  accuracy_by_generation.png  teacher_vs_student.png  chain_vs_direct.png
              error_transition.png  error_correlation.png  example_trajectories.csv
              error_transitions.csv  error_correlation.csv
jobs/<job-name>-<hash>/
   job.json  done.json  train_log.json (val curve, best step)
   test_preds.parquet   (example_id, probs, pred, label, correct)
   labels_<split>.parquet (+ .json)   weak-label artifact written by teachers:
       example_id, text, teacher_model, teacher_job, teacher_generation, hard_label,
       prob_k, logit_k, ground_truth_label (analysis only), teacher_correct
   checkpoint/          only if train.save_checkpoints: true (bf16 safetensors)
```

A job's hash covers its model config, the training config, the ids of its training, validation and test splits, its seed and, recursively, its teacher's hash. Changing a teacher therefore invalidates all its descendants and nothing else.

## GPU and memory

- Models are held in fp32 master weights with bf16 autocast on CUDA. Pythia-2.8B with AdamW needs about 45 GB, which fits comfortably on an 80 GB A100.
- For a 40 GB GPU, add `lora: true` (plus an optional `lora_config: {r, alpha, dropout, target_modules}`) or `gradient_checkpointing: true` to the 2.8B entry in `models:`.
- The best-validation state is kept in CPU RAM during training (about 11 GB for 2.8B).
- Checkpoints are off by default in the full configs; downstream jobs only need label artifacts. Turning them on costs about 30 GB per seed.

## Changing the ladder or the dataset

- **Ladder.** Edit `models:`, listed weak to strong. Each entry takes `name`, `id` (a Hugging Face model id), `lr`, `batch_size`, and optionally `params` (used as the x-axis), `epochs`, `grad_accum`, `lora` and `gradient_checkpointing`. Model ids are checked with `AutoConfig` for sequence-classification support before anything is downloaded.
  - The chain uses every model in order and needs splits `seed_true_train, g2..gN`. A 2- or 3-model ladder works as a smoke test.
  - For `base_w2s`, `base_w2s.pairs: [[p160m, p2.8b], ...]` restricts the pairs.
  - For the chain, `daisy_chain.direct: [[p160m, p2.8b]]` restricts the direct controls.
- **Dataset.** Any Hugging Face classification dataset with a labeled held-out split:
  ```yaml
  dataset: {path: imdb, name: null, text_field: text, label_field: label,
            label_names: [neg, pos], train_split: train, test_split: test}
  zero_shot: {verbalizers: [" negative", " positive"]}
  ```
  Multi-class works in the training and metrics code. The zero-shot verbalizers need one entry per label, each with a distinct first token.

## Repository layout

```
src/data.py          dataset loading, deterministic disjoint splits, leakage assertions
src/models.py        model/tokenizer loading, zero-init head, optional LoRA, id verification
src/train.py         training loop, soft/hard targets, batch-key guard, prediction
src/label.py         weak-label artifacts + validation (probabilities sum to 1, label-map match)
src/evaluate.py      test predictions, zero-shot scoring
src/metrics.py       pure metric functions (unit-tested on hand-built arrays)
src/jobs.py          Job = (model, train_split, label_source); cached, resumable execution
src/runner.py        job graph: label requests, topological order, content hashes, failure isolation
src/analysis.py      shared aggregation/report helpers
src/experiments/     base_w2s.py (Stage 1), daisy_chain.py (Stage 2): job-graph builders + analysis
src/plotting.py      figures
src/experiment.py    CLI entry point
configs/             {base,chain}_{tiny_cpu,smoke,full}.yaml
scripts/             setup_remote.sh, run_{base,chain}_{smoke,full}.sh
tests/               splits, metrics, label artifacts / leakage, job graphs, end-to-end integration
```

## Known limitations and scientific caveats

- **This is an analogue, not a simulation** of humans supervising superhuman models. Weak-model errors differ from human errors, and the pretrained students may already "know" the task.
- **Bigger is not guaranteed to be better.** Check the GT upper bounds: if a student's ceiling does not exceed its teacher's accuracy, the pair has no real capability gap. `summary.md` flags these pairs, and their PGR is not interpreted.
- **Chains confound capability scaling with self-training.** Repeated pseudo-labeling has its own noisy-label dynamics, which is why the direct and GT-upper controls are mandatory.
- **SST-2 has a compressed accuracy range** (roughly 85–93% for this ladder). With 872 test examples, one example is about 0.11 percentage points, so small PGR differences are within noise. Rely on seeds and the bootstrap CIs, and avoid over-claiming from one seed.
- **The paper's auxiliary confidence loss and bootstrapping-with-intermediate-models variants are not implemented** (listed as extensions). Hyperparameters are fixed per model size and were not tuned on test. Learning rates follow common Pythia fine-tuning values and were sanity-checked on a small run: 70M→160M on 2,000 examples gave weak 73.5%, W2S 78.8%, ceiling 83.0%, PGR 0.55.
- **Model selection uses validation data.** Weak jobs early-stop on accuracy against teacher labels on a held-out validation set, as the paper does (its Appendix A). Ground-truth jobs select on ground-truth validation labels.
