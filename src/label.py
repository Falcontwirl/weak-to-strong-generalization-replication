"""Weak-label artifacts: what a teacher says about a split, plus analysis-only ground truth."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .train import softmax_np
from .utils import read_json, write_json

PROB_TOL = 1e-4


def build_label_frame(split_df: pd.DataFrame, logits: np.ndarray, teacher_model: str,
                      teacher_job: str, teacher_generation: int) -> pd.DataFrame:
    probs = softmax_np(logits.astype(np.float64))
    df = pd.DataFrame({
        "example_id": split_df["example_id"].to_numpy(),
        "text": split_df["text"].to_numpy(),
        "teacher_model": teacher_model,
        "teacher_job": teacher_job,
        "teacher_generation": teacher_generation,
        "hard_label": probs.argmax(-1).astype(np.int64),
    })
    for k in range(probs.shape[1]):
        df[f"prob_{k}"] = probs[:, k]
        df[f"logit_{k}"] = logits[:, k].astype(np.float64)
    # Stored for analysis only; never read by the training code path (see train.targets_from_teacher).
    df["ground_truth_label"] = split_df["label"].to_numpy()
    df["teacher_correct"] = df["hard_label"] == df["ground_truth_label"]
    return df


def validate_label_frame(df: pd.DataFrame, num_labels: int) -> None:
    probs = df[[f"prob_{k}" for k in range(num_labels)]].to_numpy()
    if not np.allclose(probs.sum(-1), 1.0, atol=PROB_TOL):
        raise AssertionError("Teacher probabilities do not sum to 1")
    if ((probs < 0) | (probs > 1)).any():
        raise AssertionError("Teacher probabilities outside [0, 1]")
    if not (df["hard_label"].to_numpy() == probs.argmax(-1)).all():
        raise AssertionError("hard_label disagrees with argmax of probabilities")
    if df["example_id"].duplicated().any():
        raise AssertionError("Duplicate example ids in label artifact")


def save_labels(df: pd.DataFrame, path: Path, label_names: list[str]) -> None:
    validate_label_frame(df, len(label_names))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), {"label_names": label_names, "n": len(df)})


def load_labels(path: Path, expected_label_names: list[str]) -> pd.DataFrame:
    meta = read_json(path.with_suffix(".json"))
    if meta["label_names"] != list(expected_label_names):
        raise AssertionError(f"Label mapping mismatch: teacher {meta['label_names']} vs student {expected_label_names}")
    df = pd.read_parquet(path)
    validate_label_frame(df, len(expected_label_names))
    return df
