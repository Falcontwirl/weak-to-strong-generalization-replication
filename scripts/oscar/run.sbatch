#!/bin/bash
# Submit from the repo root, e.g.:
#   sbatch --export=ALL,CONFIG=configs/base_smoke.yaml -t 00:30:00 scripts/oscar/run.sbatch
#   sbatch --export=ALL,CONFIG=configs/base_full.yaml  scripts/oscar/run.sbatch
#   sbatch --export=ALL,CONFIG=configs/chain_full.yaml,EXTRA="--seeds 0" scripts/oscar/run.sbatch
# Command-line sbatch flags override the defaults below (e.g. -p, -t, --gres=gpu:<type>:1, --constraint).
#SBATCH -J w2s
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${CONFIG:?set CONFIG, e.g. sbatch --export=ALL,CONFIG=configs/base_full.yaml scripts/oscar/run.sbatch}"

export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # weights were prefetched on the login node
export TOKENIZERS_PARALLELISM=false

module purge
module load python/3.11 2>/dev/null || module load python
source .venv/bin/activate

echo "job $SLURM_JOB_ID on $(hostname): CONFIG=$CONFIG EXTRA=${EXTRA:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv
# Full fine-tuning of Pythia-2.8B needs ~45 GB of GPU memory; warn early on smaller cards.
MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$MEM_MB" -lt 70000 ] && grep -q "pythia-2.8b" "$CONFIG"; then
  echo "WARNING: ${MEM_MB} MB GPU; set 'lora: true' (or gradient_checkpointing: true) on the 2.8b model in $CONFIG"
fi

# Resumable: if the job hits its time limit, resubmitting the same command reuses finished jobs.
python -m src.experiment --config "$CONFIG" ${EXTRA:-}
