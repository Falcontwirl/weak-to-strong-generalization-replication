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


def test_conf_loss_reuses_gt_jobs_end_to_end(tmp_path):
    import yaml

    naive = main(["--config", "configs/base_tiny_cpu.yaml", "--output-dir", str(tmp_path / "naive")])
    cfg = yaml.safe_load(open("configs/base_tiny_cpu.yaml"))
    cfg["name"] = "base_tiny_cpu_conf"
    cfg["train"]["conf_loss"] = {"alpha": 0.75, "warmup_frac": 0.2}
    cfg_path = tmp_path / "conf.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))  # split roles are carved in `sizes` key order
    out = main(["--config", str(cfg_path), "--output-dir", str(tmp_path / "conf"), "--reuse-from", str(naive)])
    assert json.loads((out / "failures.json").read_text()) == []
    for d in (out / "jobs").iterdir():
        info = json.loads((d / "done.json").read_text())
        job = json.loads((d / "job.json").read_text())
        if job["kind"] != "train":
            continue
        if job["label_source"] is None:  # GT jobs: byte-identical copies of the naive run's
            assert (naive / "jobs" / d.name / "done.json").exists()
            assert info["conf_loss"] is None
        else:  # W2S jobs: retrained with the confidence loss
            assert not (naive / "jobs" / d.name).exists()
            assert info["conf_loss"] == {"alpha": 0.75, "warmup_frac": 0.2}
