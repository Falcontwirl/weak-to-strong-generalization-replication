"""Pure metric functions on numpy arrays. No I/O, no torch."""
from __future__ import annotations

import math

import numpy as np


def _b(x) -> np.ndarray:
    return np.asarray(x).astype(bool)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def accuracy(pred, label) -> float:
    pred, label = np.asarray(pred), np.asarray(label)
    return float((pred == label).mean()) if len(pred) else float("nan")


def pgr(weak_acc: float, w2s_acc: float, ceiling_acc: float, eps: float = 1e-12) -> tuple[float, bool]:
    """Performance gap recovered. Returns (value, valid); invalid (NaN) when ceiling <= weak."""
    denom = ceiling_acc - weak_acc
    if not np.isfinite(denom) or denom <= eps:
        return float("nan"), False
    return float((w2s_acc - weak_acc) / denom), True


def teacher_student_metrics(teacher_pred, student_pred, label) -> dict:
    t, s, y = np.asarray(teacher_pred), np.asarray(student_pred), np.asarray(label)
    tc, sc = t == y, s == y
    tw = ~tc
    return {
        "agreement": float((t == s).mean()),
        "acc_given_teacher_correct": _safe_div((sc & tc).sum(), tc.sum()),
        "acc_given_teacher_wrong": _safe_div((sc & tw).sum(), tw.sum()),
        # Of the teacher's errors: student reproduces the same wrong prediction.
        "error_inheritance": _safe_div(((s == t) & tw).sum(), tw.sum()),
        # Of the teacher's errors: student gets it right.
        "error_correction": _safe_div((sc & tw).sum(), tw.sum()),
        # Of the teacher's correct examples: student gets it wrong.
        "new_error_rate": _safe_div((~sc & tc).sum(), tc.sum()),
        "n_teacher_errors": int(tw.sum()),
    }


def brier(probs, label) -> float:
    probs, label = np.asarray(probs, dtype=float), np.asarray(label)
    onehot = np.eye(probs.shape[1])[label]
    return float(((probs - onehot) ** 2).sum(-1).mean())


def ece(probs, label, n_bins: int = 15) -> float:
    probs, label = np.asarray(probs, dtype=float), np.asarray(label)
    conf, pred = probs.max(-1), probs.argmax(-1)
    correct = pred == label
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def paired_bootstrap(correct_a, correct_b, n: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Bootstrap CI for acc(a) - acc(b) with examples resampled jointly (paired)."""
    a, b = _b(correct_a).astype(float), _b(correct_b).astype(float)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_le_zero": float((boots <= 0).mean()), "n": int(len(d))}


def transitions(correct_prev, correct_next) -> dict:
    p, q = _b(correct_prev), _b(correct_next)
    n = len(p)
    return {"C->C": float((p & q).sum() / n), "W->C": float((~p & q).sum() / n),
            "C->W": float((p & ~q).sum() / n), "W->W": float((~p & ~q).sum() / n)}


def error_correlation(err_a, err_b) -> dict:
    """Phi coefficient, Cohen's kappa and conditional error rate between two error indicators."""
    a, b = _b(err_a), _b(err_b)
    n11, n10, n01, n00 = (a & b).sum(), (a & ~b).sum(), (~a & b).sum(), (~a & ~b).sum()
    denom = math.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    phi = float((n11 * n00 - n10 * n01) / denom) if denom else float("nan")
    n = len(a)
    po = (n11 + n00) / n
    pe = ((n11 + n10) / n) * ((n11 + n01) / n) + ((n01 + n00) / n) * ((n10 + n00) / n)
    kappa = float((po - pe) / (1 - pe)) if pe != 1 else float("nan")
    return {"phi": phi, "kappa": kappa, "p_err_b_given_err_a": _safe_div(n11, n11 + n10),
            "p_err_b_given_correct_a": _safe_div(n01, n01 + n00)}


def confidence_bins(teacher_probs, teacher_pred, label, student_pred, n_bins: int = 5) -> list[dict]:
    """Bin by teacher max-probability (equal-width over [0.5, 1] for binary / [1/C, 1] generally)."""
    tp = np.asarray(teacher_probs, dtype=float)
    conf = tp.max(-1)
    t, y, s = np.asarray(teacher_pred), np.asarray(label), np.asarray(student_pred)
    lo = 1.0 / tp.shape[1]
    edges = np.linspace(lo, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        tw = m & (t != y)
        rows.append({
            "bin": b, "conf_low": float(edges[b]), "conf_high": float(edges[b + 1]), "n": int(m.sum()),
            "teacher_acc": _safe_div((t[m] == y[m]).sum(), m.sum()),
            "student_acc": _safe_div((s[m] == y[m]).sum(), m.sum()),
            "n_teacher_errors": int(tw.sum()),
            "error_correction": _safe_div(((s == y) & tw).sum(), tw.sum()),
        })
    return rows
