#!/usr/bin/env bash
# One-time setup on the GPU machine. Assumes Python >= 3.10 and an NVIDIA driver.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# Fast unit tests (no downloads)
python -m pytest -q -m "not integration"
