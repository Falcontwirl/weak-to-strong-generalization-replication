"""End-to-end CPU run of the base experiment on tiny Pythia models (downloads ~150MB once)."""
import json

import pandas as pd
import pytest

from src.experiment import main

pytestmark = pytest.mark.integration

ARTIFACTS = ["resolved_config.yaml", "run_metadata.json", "split_indices.json", "metrics.csv", "metrics_summary.csv",
             "bootstrap.csv", "summary.md", "w2s_accuracy.png", "pgr_grid.png", "confidence_bins.png",
             "failures.json"]


def test_base_tiny_cpu_end_to_end(tmp_path):
    out = main(["--config", "configs/base_tiny_cpu.yaml", "--output-dir", str(tmp_path)])
    for a in ARTIFACTS:
        assert (out / a).exists(), a
    assert json.loads((out / "failures.json").read_text()) == []
    m = pd.read_csv(out / "metrics.csv")
    assert set(m.condition) == {"teacher_gt", "ceiling_gt", "w2s", "zero_shot"}
    w = m[m.condition == "w2s"]
    assert len(w) == 3 and w[["test_acc", "teacher_test_acc", "agreement"]].notna().all().all()
    # A second invocation reuses every cached job.
    main(["--config", "configs/base_tiny_cpu.yaml", "--output-dir", str(tmp_path)])
