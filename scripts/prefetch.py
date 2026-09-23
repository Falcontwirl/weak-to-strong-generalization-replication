"""Download the dataset and every model in the given configs into the HF cache (run on a node with internet).

    python scripts/prefetch.py configs/base_full.yaml configs/chain_full.yaml
"""
import sys

from datasets import load_dataset
from huggingface_hub import snapshot_download

sys.path.insert(0, ".")
from src.utils import load_config  # noqa: E402

seen = set()
for path in sys.argv[1:]:
    cfg = load_config(path)
    d = cfg["dataset"]
    if (d["path"], d.get("name")) not in seen:
        seen.add((d["path"], d.get("name")))
        load_dataset(d["path"], d.get("name"))
        print(f"dataset {d['path']}/{d.get('name')} cached")
    for m in cfg["models"]:
        if m["id"] not in seen:
            seen.add(m["id"])
            snapshot_download(m["id"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
            print(f"model {m['id']} cached")
