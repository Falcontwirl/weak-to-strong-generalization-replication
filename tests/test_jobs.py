"""Job-graph resolution: label requests, dependency order, cache keys, test-split guard."""
import copy

import pandas as pd
import pytest

from src.data import TaskData, make_splits
from src.experiments import base_w2s
from src.jobs import Job, RunContext
from src.runner import prepare
from src.utils import DEFAULTS, deep_merge


def ctx(tmp_path, **train):
    cfg = deep_merge(DEFAULTS, {
        "name": "t", "splits": {"sizes": {"val": 5, "weak_train": 10, "strong_train": 10}},
        "models": [{"name": "a", "id": "x/a", "lr": 1e-4}, {"name": "b", "id": "x/b", "lr": 1e-4},
                   {"name": "c", "id": "x/c", "lr": 1e-4}],
        "train": train,
    })
    train_ids, test_ids = [f"train:{i}" for i in range(40)], [f"test:{i}" for i in range(5)]
    splits = make_splits(train_ids, test_ids, cfg["splits"]["sizes"], 0)
    ex = pd.DataFrame({"text": "x", "label": 0}, index=pd.Index(train_ids + test_ids, name="example_id"))
    return RunContext(cfg=cfg, out_dir=tmp_path, task=TaskData(ex, splits, ["n", "p"], {}), device="cpu")


def test_base_graph_structure(tmp_path):
    c = ctx(tmp_path)
    jobs = prepare(base_w2s.build_jobs(c.cfg, 0, first=True), c)
    by = {j.name: j for j in jobs}
    assert sum(j.condition == "w2s" for j in jobs) == 3
    # teachers must label strong_train (student data) and val (teacher-labeled model selection)
    assert set(by["s0/teacher_gt/a"].label_splits) == {"strong_train", "val"}
    assert by["s0/ceiling_gt/c"].label_splits == ()
    order = [j.name for j in jobs]
    assert order.index("s0/teacher_gt/a") < order.index("s0/w2s/a->c")
    assert len({j.key for j in jobs}) == len(jobs)


def test_selection_gt_ablation_does_not_request_val_labels(tmp_path):
    c = ctx(tmp_path, selection="val_gt")
    jobs = prepare(base_w2s.build_jobs(c.cfg, 0, first=False), c)
    assert {j.name: j for j in jobs}["s0/teacher_gt/a"].label_splits == ("strong_train",)


def test_keys_change_with_teacher_and_seed(tmp_path):
    c = ctx(tmp_path)
    k0 = {j.name: j.key for j in prepare(base_w2s.build_jobs(c.cfg, 0, True), c)}
    k0b = {j.name: j.key for j in prepare(base_w2s.build_jobs(c.cfg, 0, True), c)}
    assert k0 == k0b
    c2 = copy.deepcopy(c)
    c2.cfg["models"][0]["lr"] = 5e-4  # changes teacher a -> must change its students' keys
    k2 = {j.name: j.key for j in prepare(base_w2s.build_jobs(c2.cfg, 0, True), c2)}
    assert k2["s0/w2s/a->b"] != k0["s0/w2s/a->b"]
    assert k2["s0/w2s/b->c"] == k0["s0/w2s/b->c"]
    k1 = {j.name.split("/", 1)[1]: j.key for j in prepare(base_w2s.build_jobs(c.cfg, 1, False), c)}
    assert k1["w2s/a->b"] != k0["s0/w2s/a->b"]


def test_training_on_test_split_forbidden(tmp_path):
    c = ctx(tmp_path)
    with pytest.raises(AssertionError):
        prepare([Job("bad", "a", 0, train_split="test")], c)


def test_unknown_label_source(tmp_path):
    c = ctx(tmp_path)
    with pytest.raises(ValueError):
        prepare([Job("s", "b", 0, train_split="strong_train", label_source="nope")], c)


def chain_ctx(tmp_path, n=4):
    c = ctx(tmp_path)
    c.cfg["experiment"] = "daisy_chain"
    c.cfg["models"] = [{"name": f"m{i}", "id": f"x/m{i}", "lr": 1e-4} for i in range(1, n + 1)]
    sizes = {"val": 5, "seed_true_train": 5, **{f"g{k}": 5 for k in range(2, n + 1)}}
    train_ids = [f"train:{i}" for i in range(40)]
    splits = make_splits(train_ids, [f"test:{i}" for i in range(5)], sizes, 0)
    c.task = TaskData(c.task.examples, splits, ["n", "p"], {})
    return c


def test_chain_graph_structure(tmp_path):
    from src.experiments import daisy_chain

    c = chain_ctx(tmp_path)
    jobs = prepare(daisy_chain.build_jobs(c.cfg, 0, first=False), c)
    by = {j.name: j for j in jobs}
    assert daisy_chain.direct_pairs(c.cfg) == [(1, 3), (1, 4), (2, 4)]
    # each chain student is supervised by its immediate predecessor on its own split
    assert by["s0/chain/g2/m2"].label_source == "s0/seed_gt/m1"
    assert by["s0/chain/g3/m3"].label_source == "s0/chain/g2/m2"
    assert by["s0/chain/g4/m4"].label_source == "s0/chain/g3/m3"
    assert all(by[f"s0/chain/g{k}/m{k}"].train_split == f"g{k}" for k in (2, 3, 4))
    # chain M4, direct M1->M4, direct M2->M4 and GT M4 share the same g4 examples
    assert {by[n].train_split for n in ["s0/chain/g4/m4", "s0/direct/m1->m4", "s0/direct/m2->m4",
                                        "s0/gt_upper/m4"]} == {"g4"}
    # teachers are asked to label exactly the splits their students need (+ val for selection)
    assert set(by["s0/seed_gt/m1"].label_splits) == {"g2", "g3", "g4", "val"}
    assert set(by["s0/chain/g2/m2"].label_splits) == {"g3", "g4", "val"}
    assert by["s0/chain/g4/m4"].label_splits == ()
    assert daisy_chain.required_splits(c.cfg) == ("val", "seed_true_train", "g2", "g3", "g4")


def test_chain_direct_pairs_validation(tmp_path):
    from src.experiments import daisy_chain

    c = chain_ctx(tmp_path)
    c.cfg["daisy_chain"] = {"direct": [["m1", "m2"]]}
    with pytest.raises(ValueError):
        daisy_chain.direct_pairs(c.cfg)
    c.cfg["daisy_chain"] = {"direct": [["m1", "m4"]]}
    assert daisy_chain.direct_pairs(c.cfg) == [(1, 4)]
