#!/usr/bin/env bash
# Usage: bash scripts/run_chain_full.sh [extra args, e.g. --dry-run | --force | --seeds 0]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate
python -m src.experiment --config configs/chain_full.yaml "$@" 2>&1 | tee -a "outputs/chain_full.log"
