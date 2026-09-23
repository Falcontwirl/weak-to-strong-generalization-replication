#!/usr/bin/env bash
# Usage: bash scripts/run_base_smoke.sh [extra args, e.g. --dry-run | --force | --seeds 0]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate
python -m src.experiment --config configs/base_smoke.yaml "$@" 2>&1 | tee -a "outputs/base_smoke.log"
