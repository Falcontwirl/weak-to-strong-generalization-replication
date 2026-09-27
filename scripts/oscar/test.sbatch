#!/bin/bash
# Full test suite (incl. tiny-Pythia CPU integration tests) on a compute node. Submit from the repo root:
#   sbatch scripts/oscar/test.sbatch
# Gate GPU runs on it with: sbatch --dependency=afterok:<test job id> ... scripts/oscar/run.sbatch
#SBATCH -J w2s-test
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # tiny Pythia models and SST-2 are cached there
export TOKENIZERS_PARALLELISM=false
module purge
source .venv/bin/activate
python -m pytest -q
