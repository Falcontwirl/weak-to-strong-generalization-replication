import pytest

from src.data import TEST_ROLE, assert_disjoint, make_splits

TRAIN = [f"train:{i}" for i in range(1000)]
TEST = [f"test:{i}" for i in range(100)]
SIZES = {"val": 50, "weak_train": 200, "strong_train": 200}


def test_splits_disjoint_and_sized():
    s = make_splits(TRAIN, TEST, SIZES, split_seed=7)
    assert {k: len(v) for k, v in s.items()} == {**SIZES, TEST_ROLE: 100}
    all_ids = [i for v in s.values() for i in v]
    assert len(all_ids) == len(set(all_ids))


def test_splits_deterministic_and_seed_dependent():
    a = make_splits(TRAIN, TEST, SIZES, split_seed=7)
    b = make_splits(TRAIN, TEST, SIZES, split_seed=7)
    c = make_splits(TRAIN, TEST, SIZES, split_seed=8)
    assert a == b
    assert a["weak_train"] != c["weak_train"]


def test_test_split_never_drawn_from_train():
    s = make_splits(TRAIN, TEST, SIZES, split_seed=7, test_limit=20)
    assert len(s[TEST_ROLE]) == 20 and all(i.startswith("test:") for i in s[TEST_ROLE])
    assert not any(i.startswith("test:") for k, v in s.items() if k != TEST_ROLE for i in v)


def test_leakage_assertion_fires():
    with pytest.raises(AssertionError, match="Leakage"):
        assert_disjoint({"a": ["x", "y"], "b": ["y", "z"]})
    with pytest.raises(AssertionError):
        assert_disjoint({"a": ["x", "x"]})


def test_oversized_request_fails():
    with pytest.raises(ValueError):
        make_splits(TRAIN, TEST, {"val": 600, "weak_train": 600}, split_seed=0)


def test_test_role_reserved():
    with pytest.raises(ValueError):
        make_splits(TRAIN, TEST, {"test": 10}, split_seed=0)
