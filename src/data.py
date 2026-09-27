"""Dataset loading and deterministic, disjoint split construction.

Every example carries a stable string ``example_id`` ("train:<row>" / "test:<row>"), and every
split role is a list of example IDs persisted to ``split_indices.json``. The final test set is
the dataset's labeled held-out split and is never used for training, selection or labeling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import read_json, stable_hash, write_json

TEST_ROLE = "test"


@dataclass
class TaskData:
    examples: pd.DataFrame  # index: example_id; columns: text, label
    splits: dict[str, list[str]]  # role -> example ids (includes TEST_ROLE)
    label_names: list[str]
    info: dict

    def split(self, role: str) -> pd.DataFrame:
        ids = self.splits[role]
        df = self.examples.loc[ids].reset_index()
        return df[["example_id", "text", "label"]]

    def split_hash(self, role: str) -> str:
        return stable_hash(self.splits[role])


def load_examples(dcfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from datasets import load_dataset

    ds = load_dataset(dcfg["path"], dcfg.get("name"))
    tf, lf, tmpl = dcfg["text_field"], dcfg["label_field"], dcfg.get("text_template")

    def to_df(split: str, prefix: str) -> pd.DataFrame:
        d = ds[split]
        # text_template (e.g. "Passage: {passage}\nQuestion: {question}?") builds the input from several fields.
        texts = [tmpl.format(**row) for row in d] if tmpl else d[tf]
        df = pd.DataFrame({"text": texts, "label": np.asarray(d[lf], dtype=np.int64)})
        df["example_id"] = [f"{prefix}:{i}" for i in range(len(df))]
        return df.set_index("example_id")

    train = to_df(dcfg["train_split"], "train")
    test = to_df(dcfg["test_split"], "test")
    if (test["label"] < 0).any():
        raise ValueError(f"Split '{dcfg['test_split']}' has hidden labels (-1); choose a labeled test split.")
    info = {
        "path": dcfg["path"],
        "name": dcfg.get("name"),
        "train_split": dcfg["train_split"],
        "test_split": dcfg["test_split"],
        "text_template": tmpl,
        "fingerprint_train": getattr(ds[dcfg["train_split"]], "_fingerprint", None),
        "fingerprint_test": getattr(ds[dcfg["test_split"]], "_fingerprint", None),
        "n_train_pool": len(train),
        "n_test_full": len(test),
    }
    return train, test, info


def make_splits(train_ids: list[str], test_ids: list[str], sizes: dict[str, int], split_seed: int,
                test_limit: int | None = None) -> dict[str, list[str]]:
    """Carve disjoint roles (in the order given by ``sizes``) from a seeded permutation of the train pool."""
    total = sum(sizes.values())
    if total > len(train_ids):
        raise ValueError(f"Requested {total} training examples but pool has only {len(train_ids)}")
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(train_ids))
    splits, start = {}, 0
    for role, n in sizes.items():
        if role == TEST_ROLE:
            raise ValueError("'test' is reserved for the held-out evaluation split")
        splits[role] = [train_ids[i] for i in perm[start:start + n]]
        start += n
    test = list(test_ids)
    if test_limit is not None and test_limit < len(test):
        trng = np.random.default_rng(split_seed + 1)
        test = [test[i] for i in sorted(trng.choice(len(test), size=test_limit, replace=False))]
    splits[TEST_ROLE] = test
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for role, ids in splits.items():
        if len(set(ids)) != len(ids):
            raise AssertionError(f"Duplicate example ids within split '{role}'")
        for eid in ids:
            if eid in seen:
                raise AssertionError(f"Leakage: example {eid} appears in both '{seen[eid]}' and '{role}'")
            seen[eid] = role


def load_task(cfg: dict, out_dir: Path, force: bool = False) -> TaskData:
    dcfg, scfg = cfg["dataset"], cfg["splits"]
    train, test, info = load_examples(dcfg)
    splits = make_splits(list(train.index), list(test.index), dict(scfg["sizes"]), scfg["split_seed"],
                         dcfg.get("test_limit"))
    path = out_dir / "split_indices.json"
    payload = {"split_seed": scfg["split_seed"], "sizes": {k: len(v) for k, v in splits.items()}, "splits": splits}
    if path.exists() and not force:
        prev = read_json(path)
        if prev["splits"] != splits:
            raise RuntimeError(f"{path} differs from freshly computed splits; rerun with --force to overwrite.")
    else:
        write_json(path, payload)
    examples = pd.concat([train, test])
    return TaskData(examples=examples, splits=splits, label_names=list(dcfg["label_names"]), info=info)
