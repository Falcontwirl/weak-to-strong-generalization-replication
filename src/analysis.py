"""Shared analysis helpers used by every experiment's ``analyze``."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .jobs import JobResult

GROUP_KEYS = ["experiment", "condition", "model", "teacher", "generation"]


def probs(df: pd.DataFrame) -> np.ndarray:
    cols = sorted([c for c in df.columns if c.startswith("prob_")], key=lambda c: int(c.split("_")[1]))
    return df[cols].to_numpy()


def pred_metrics(preds: pd.DataFrame, ece_bins: int) -> dict:
    p = probs(preds)
    return {"test_acc": M.accuracy(preds["pred"], preds["label"]), "brier": M.brier(p, preds["label"]),
            "ece": M.ece(p, preds["label"], ece_bins), "n_test": len(preds)}


def student_vs_teacher(student: pd.DataFrame, teacher: pd.DataFrame) -> dict:
    assert (student["example_id"].to_numpy() == teacher["example_id"].to_numpy()).all()
    return M.teacher_student_metrics(teacher["pred"], student["pred"], student["label"])


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std over seeds for every numeric column, grouped by condition identity."""
    keys = [k for k in GROUP_KEYS if k in df.columns]
    num = [c for c in df.select_dtypes("number").columns if c not in keys + ["seed"]]
    g = df.groupby(keys, dropna=False)
    out = g[num].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.size()
    return out.reset_index()


def pm(mean, std, digits: int = 3) -> str:
    if mean is None or not np.isfinite(mean):
        return "n/a"
    if std is None or not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def md_table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def failures_md(results: dict[str, JobResult]) -> str:
    bad = [r for r in results.values() if r.status != "done"]
    if not bad:
        return "All jobs completed successfully; no runs were excluded."
    rows = [{"job": r.job.name, "status": r.status, "error": (r.error or "").replace("|", "/")[:200]} for r in bad]
    return ("The following jobs failed or were skipped. Metrics that depend on them are missing "
            "(not imputed):\n\n" + md_table(rows, ["job", "status", "error"]))


def ok(results: dict[str, JobResult], *names: str) -> bool:
    return all(n in results and results[n].status == "done" for n in names)
