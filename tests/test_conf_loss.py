"""Auxiliary confidence loss targets and schedule."""
import pytest
import torch

from src.train import conf_coef, conf_targets


def test_coef_zero_returns_weak_labels():
    weak = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    logits = torch.randn(4, 2)
    assert torch.allclose(conf_targets(logits, weak, 0.0), weak)


def test_hardened_predictions_match_weak_class_balance():
    torch.manual_seed(0)
    n = 1000
    weak = torch.zeros(n, 2)
    weak[:300, 1] = 1.0  # 30% positive weak labels
    weak[300:, 0] = 1.0
    logits = torch.randn(n, 2)  # student's ranking is unrelated to the weak labels
    hard = conf_targets(logits, weak, 1.0)
    assert set(hard.flatten().tolist()) <= {0.0, 1.0}
    assert abs(hard[:, 1].mean().item() - 0.3) < 0.01
    # the positives are the student's most confident positives
    p1 = torch.softmax(logits, -1)[:, 1]
    assert p1[hard[:, 1] == 1].min() >= p1[hard[:, 1] == 0].max()


def test_targets_are_distributions():
    weak = torch.softmax(torch.randn(64, 2), -1)
    t = conf_targets(torch.randn(64, 2), weak, 0.5)
    assert torch.allclose(t.sum(-1), torch.ones(64))


def test_coef_warmup_schedule():
    conf = {"alpha": 0.75, "warmup_frac": 0.2}
    assert conf_coef(0, 100, conf) == 0.0
    assert conf_coef(10, 100, conf) == pytest.approx(0.375)
    assert conf_coef(20, 100, conf) == pytest.approx(0.75)
    assert conf_coef(90, 100, conf) == pytest.approx(0.75)


def test_multiclass_rejected():
    with pytest.raises(ValueError):
        conf_targets(torch.randn(4, 3), torch.softmax(torch.randn(4, 3), -1), 0.5)


def test_ref_schedule_matches_openai_logconf():
    conf = {"alpha": 0.5, "warmup_frac": 0.1, "schedule": "ref"}
    assert conf_coef(0, 100, conf) == 0.0
    assert conf_coef(5, 100, conf) == pytest.approx(0.025)   # alpha * step_frac during warmup
    assert conf_coef(10, 100, conf) == pytest.approx(0.05)
    assert conf_coef(11, 100, conf) == pytest.approx(0.5)    # then jumps to alpha
    with pytest.raises(ValueError):
        conf_coef(1, 100, {"alpha": 0.5, "schedule": "cosine"})


def test_alpha_zero_is_exactly_the_weak_labels():
    weak = torch.softmax(torch.randn(32, 2), -1)
    conf = {"alpha": 0.0, "warmup_frac": 0.2}
    for step in (0, 10, 100):
        assert torch.equal(conf_targets(torch.randn(32, 2), weak, conf_coef(step, 100, conf)), weak)
