import math

import numpy as np
import pytest

from src import metrics as M


def test_accuracy():
    assert M.accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75


def test_pgr_values_and_invalid_denominator():
    assert M.pgr(0.7, 0.8, 0.9) == (pytest.approx(0.5), True)
    assert M.pgr(0.7, 0.95, 0.9)[0] == pytest.approx(1.25)
    assert M.pgr(0.7, 0.6, 0.9)[0] == pytest.approx(-0.5)
    v, ok = M.pgr(0.9, 0.8, 0.9)
    assert math.isnan(v) and not ok
    v, ok = M.pgr(0.9, 0.8, 0.85)
    assert math.isnan(v) and not ok


def test_teacher_student_metrics_hand_example():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    t = np.array([0, 0, 1, 1, 1, 1, 0, 0])  # teacher wrong on idx 2,3,6,7
    s = np.array([0, 1, 1, 0, 1, 1, 1, 0])  # student: wrong on 1 (new), copies 2 and 7, fixes 3 and 6
    m = M.teacher_student_metrics(t, s, y)
    assert m["agreement"] == pytest.approx(5 / 8)
    assert m["n_teacher_errors"] == 4
    assert m["error_inheritance"] == pytest.approx(2 / 4)
    assert m["error_correction"] == pytest.approx(2 / 4)
    assert m["new_error_rate"] == pytest.approx(1 / 4)
    assert m["acc_given_teacher_correct"] == pytest.approx(3 / 4)
    assert m["acc_given_teacher_wrong"] == pytest.approx(2 / 4)


def test_teacher_student_no_teacher_errors_is_nan():
    m = M.teacher_student_metrics([0, 1], [0, 1], [0, 1])
    assert math.isnan(m["error_correction"]) and m["new_error_rate"] == 0.0


def test_brier_and_ece():
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert M.brier(p, [0, 1]) == 0.0
    assert M.brier(p, [1, 0]) == pytest.approx(2.0)
    assert M.ece(p, [0, 1]) == pytest.approx(0.0)
    # all predictions 0.8 confident, half correct -> ECE = |0.5 - 0.8| = 0.3
    p = np.array([[0.8, 0.2]] * 4)
    assert M.ece(p, [0, 0, 1, 1]) == pytest.approx(0.3)


def test_paired_bootstrap():
    a = np.array([1] * 90 + [0] * 10)
    b = np.array([1] * 80 + [0] * 20)
    r = M.paired_bootstrap(a, b, n=2000, seed=0)
    assert r["diff"] == pytest.approx(0.1)
    assert r["ci_low"] <= 0.1 <= r["ci_high"]
    assert r["ci_low"] > 0  # a dominates b example-wise
    same = M.paired_bootstrap(a, a, n=500)
    assert same["diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0


def test_transitions_sum_to_one():
    t = M.transitions([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert t == {"C->C": 0.4, "W->C": 0.2, "C->W": 0.2, "W->W": 0.2}


def test_error_correlation():
    e = np.array([1, 1, 0, 0, 0, 0])
    r = M.error_correlation(e, e)
    assert r["phi"] == pytest.approx(1.0) and r["kappa"] == pytest.approx(1.0)
    assert r["p_err_b_given_err_a"] == 1.0 and r["p_err_b_given_correct_a"] == 0.0
    r = M.error_correlation(e, 1 - e)
    assert r["phi"] == pytest.approx(-1.0)


def test_confidence_bins():
    probs = np.array([[0.55, 0.45], [0.52, 0.48], [0.95, 0.05], [0.99, 0.01]])
    t = probs.argmax(1)  # all predict 0
    y = np.array([1, 0, 1, 0])  # teacher wrong on 0 (low conf) and 2 (high conf)
    s = np.array([1, 0, 0, 0])  # student fixes the low-confidence error only
    rows = M.confidence_bins(probs, t, y, s, n_bins=2)
    assert rows[0]["n"] == 2 and rows[0]["error_correction"] == 1.0
    assert rows[1]["n"] == 2 and rows[1]["error_correction"] == 0.0
