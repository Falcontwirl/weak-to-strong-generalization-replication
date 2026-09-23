#!/usr/bin/env bash
# One-time setup on an Oscar LOGIN node (has internet). Usage: bash scripts/oscar/setup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# Keep the ~8 GB of model weights out of the home-directory quota.
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
mkdir -p "$HF_HOME" logs

module purge
module load python/3.11 2>/dev/null || module load python   # `module avail python` lists versions
python --version

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt     # PyTorch wheels ship their own CUDA runtime; no `module load cuda` needed

python -m pytest -q -m "not integration"
python scripts/prefetch.py configs/base_smoke.yaml configs/base_full.yaml configs/chain_smoke.yaml configs/chain_full.yaml
echo "Setup done. HF cache: $HF_HOME"
