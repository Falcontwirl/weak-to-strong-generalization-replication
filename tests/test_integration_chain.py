"""End-to-end CPU run of the daisy-chain experiment on tiny Pythia models, plus resume behaviour."""
import json
import shutil

import pandas as pd
import pytest
import yaml

from src.experiment import main

pytestmark = pytest.mark.integration

ARTIFACTS = ["resolved_config.yaml", "run_metadata.json", "split_indices.json", "metrics.csv", "bootstrap.csv",
             "example_trajectories.csv", "error_transitions.csv", "error_correlation.csv", "summary.md",
             "accuracy_by_generation.png", "teacher_vs_student.png", "chain_vs_direct.png", "error_transition.png",
             "error_correlation.png", "confidence_bins.png", "failures.json"]


@pytest.fixture
def cfg_path(tmp_path):
    cfg = yaml.safe_load(open("configs/chain_tiny_cpu.yaml"))
    cfg["train"]["save_checkpoints"] = True
    cfg["zero_shot"] = {"enabled": False}
    p = tmp_path / "chain.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_chain_tiny_cpu_end_to_end(tmp_path, cfg_path):
    out_dir = tmp_path / "out"
    out = main(["--config", str(cfg_path), "--output-dir", str(out_dir)])
    for a in ARTIFACTS:
        assert (out / a).exists(), a
    assert json.loads((out / "failures.json").read_text()) == []
    m = pd.read_csv(out / "metrics.csv")
    assert (m.condition == "chain").sum() == 3 and (m.condition == "direct").sum() == 3
    assert (m.condition == "gt_upper").sum() == 3 and (m.condition == "seed_gt").sum() == 1

    # Every student's training split is disjoint from every other role and labeled by its declared teacher.
    splits = json.loads((out / "split_indices.json").read_text())["splits"]
    m1_dir = next((out / "jobs").glob("s0_seed_gt_*"))
    labels = pd.read_parquet(m1_dir / "labels_g2.parquet")
    assert set(labels.example_id) == set(splits["g2"])
    assert set(labels.teacher_job) == {"s0/seed_gt/p14m"}
    assert not set(labels.example_id) & set(splits["test"])

    # Resume: a deleted label artifact is regenerated from the saved checkpoint without retraining.
    shutil.move(m1_dir / "labels_g2.parquet", tmp_path / "old.parquet")
    main(["--config", str(cfg_path), "--output-dir", str(out_dir)])
    regen = pd.read_parquet(m1_dir / "labels_g2.parquet")
    old = pd.read_parquet(tmp_path / "old.parquet")
    assert (regen.hard_label == old.hard_label).mean() > 0.95  # bf16 checkpoint round-trip
