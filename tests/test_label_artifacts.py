import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from src.label import build_label_frame, load_labels, save_labels, validate_label_frame
from src.train import BATCH_KEYS, ClassificationDataset, make_collate, targets_from_teacher

LABELS = ["negative", "positive"]


def split_df(n=6):
    return pd.DataFrame({"example_id": [f"train:{i}" for i in range(n)], "text": [f"t{i}" for i in range(n)],
                         "label": np.array([0, 1] * (n // 2))})


def teacher_frame(df):
    # Teacher is confidently wrong on every example, so its targets must differ from ground truth.
    logits = np.stack([df["label"] * 3.0, (1 - df["label"]) * 3.0], axis=1)
    return build_label_frame(df, logits, "teacher/model", "s0/teacher", 1)


def test_label_frame_schema_and_probs(tmp_path):
    df = split_df()
    lf = teacher_frame(df)
    for c in ["example_id", "text", "teacher_model", "teacher_generation", "hard_label", "prob_0", "prob_1",
              "logit_0", "logit_1", "ground_truth_label", "teacher_correct"]:
        assert c in lf.columns
    assert np.allclose(lf[["prob_0", "prob_1"]].sum(1), 1.0)
    assert not lf["teacher_correct"].any()
    save_labels(lf, tmp_path / "labels_x.parquet", LABELS)
    back = load_labels(tmp_path / "labels_x.parquet", LABELS)
    pd.testing.assert_frame_equal(back, lf)


def test_label_mapping_mismatch_rejected(tmp_path):
    lf = teacher_frame(split_df())
    save_labels(lf, tmp_path / "l.parquet", LABELS)
    with pytest.raises(AssertionError, match="mapping"):
        load_labels(tmp_path / "l.parquet", ["positive", "negative"])


def test_invalid_probs_rejected():
    lf = teacher_frame(split_df())
    lf.loc[0, "prob_0"] = 0.9
    lf.loc[0, "prob_1"] = 0.9
    with pytest.raises(AssertionError):
        validate_label_frame(lf, 2)


@pytest.mark.parametrize("loss", ["soft_ce", "hard_ce"])
def test_student_batches_never_expose_ground_truth(loss):
    df = split_df()
    lf = teacher_frame(df)
    targets = targets_from_teacher(df, lf.sample(frac=1, random_state=0), 2, loss)  # shuffled -> must realign
    # Targets come from the (always wrong) teacher, never from ground truth.
    assert (targets.argmax(1) != df["label"].to_numpy()).all()
    if loss == "soft_ce":
        assert np.allclose(targets, lf[["prob_0", "prob_1"]].to_numpy())
    ds = ClassificationDataset([[1, 2, 3]] * len(df), targets)
    for item in ds:
        assert set(item) == {"input_ids", "target_probs"}
    for batch in DataLoader(ds, batch_size=4, collate_fn=make_collate(0)):
        assert set(batch) <= BATCH_KEYS
        assert "ground_truth_label" not in batch and "label" not in batch


def test_missing_teacher_labels_rejected():
    df = split_df()
    lf = teacher_frame(df).iloc[:-1]
    with pytest.raises(AssertionError, match="missing"):
        targets_from_teacher(df, lf, 2, "soft_ce")
