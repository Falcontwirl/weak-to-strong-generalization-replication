"""The reusable unit of every experiment: a training (or zero-shot) job.

A job trains ``model`` on ``train_split`` with targets from ``label_source`` (None => ground
truth; otherwise the name of another job whose label artifact for ``train_split`` is used),
evaluates on the held-out test split, and writes label artifacts for ``label_splits`` so
downstream jobs can use it as a teacher. Results are cached on disk by a content hash.
"""
from __future__ import annotations

import gc
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

from .data import TEST_ROLE, TaskData
from .evaluate import build_pred_frame, zero_shot_logits
from .label import build_label_frame, load_labels, save_labels
from .models import count_params, load_causal_lm, load_classifier
from .train import predict, targets_from_gt, targets_from_teacher, tokenize, train_model
from .utils import log, read_json, set_seed, write_json

VAL_ROLE = "val"


@dataclass
class Job:
    name: str  # unique within a run, e.g. "s0/w2s/p160m->p1b"
    model: str  # ladder model name
    seed: int
    kind: str = "train"  # train | zero_shot
    train_split: str | None = None
    label_source: str | None = None  # None => ground-truth labels
    condition: str = ""  # free-form tag used by analysis (teacher_gt, ceiling_gt, w2s, chain, ...)
    generation: int = 0
    label_splits: tuple[str, ...] = ()  # filled/extended by the runner
    meta: dict = field(default_factory=dict)
    key: str | None = None  # content hash, set by the runner

    @property
    def is_weak(self) -> bool:
        return self.label_source is not None


@dataclass
class RunContext:
    cfg: dict
    out_dir: Path
    task: TaskData
    device: str
    force: bool = False

    def model_cfg(self, name: str) -> dict:
        for m in self.cfg["models"]:
            if m["name"] == name:
                return m
        raise KeyError(name)


@dataclass
class JobResult:
    job: Job
    dir: Path
    status: str  # done | failed | skipped
    info: dict = field(default_factory=dict)
    error: str | None = None

    def test_preds(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "test_preds.parquet")

    def labels(self, split: str, label_names: list[str]) -> pd.DataFrame:
        return load_labels(self.dir / f"labels_{split}.parquet", label_names)


def job_dir(ctx: RunContext, job: Job) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", job.name.replace("->", "_to_"))
    return ctx.out_dir / "jobs" / f"{safe}-{job.key}"


def selection_mode(job: Job, tcfg: dict) -> str:
    sel = tcfg["selection"]
    if sel == "auto":
        return "val_teacher" if job.is_weak else "val_gt"
    if sel == "val_teacher" and not job.is_weak:
        return "val_gt"
    return sel


def _missing_labels(d: Path, job: Job) -> list[str]:
    return [s for s in job.label_splits if not (d / f"labels_{s}.parquet").exists()]


def run_job(job: Job, ctx: RunContext, source: JobResult | None) -> JobResult:
    d = job_dir(ctx, job)
    done = d / "done.json"
    if done.exists() and not ctx.force:
        missing = _missing_labels(d, job)
        if not missing:
            log(f"[cached] {job.name}")
            return JobResult(job, d, "done", read_json(done))
        if (d / "checkpoint").exists():
            log(f"[relabel] {job.name}: labeling {missing} from checkpoint")
            model, tok = _load_checkpoint(ctx, job, d)
            _write_labels(model, tok, ctx, job, d, missing)
            _free(model)
            return JobResult(job, d, "done", read_json(done))
        log(f"[retrain] {job.name}: missing labels {missing} and no checkpoint")
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "job.json", {k: v for k, v in job.__dict__.items()})
    log(f"[run] {job.name} ({job.kind}, model={job.model}, split={job.train_split}, "
        f"labels={'GT' if not job.is_weak else job.label_source})")
    set_seed(job.seed)
    t0 = time.time()
    if job.kind == "zero_shot":
        info = _run_zero_shot(job, ctx, d)
    else:
        info = _run_train(job, ctx, d, source)
    info["wall_seconds"] = time.time() - t0
    write_json(done, info)
    log(f"[done] {job.name}: test_acc={info['test_acc']:.4f} ({info['wall_seconds']:.0f}s)")
    return JobResult(job, d, "done", info)


def _run_zero_shot(job: Job, ctx: RunContext, d: Path) -> dict:
    zcfg, tcfg = ctx.cfg["zero_shot"], ctx.cfg["train"]
    mcfg = ctx.model_cfg(job.model)
    model, tok = load_causal_lm(mcfg["id"], ctx.device)
    tok.truncation_side = tcfg.get("truncation_side", "right")
    test = ctx.task.split(TEST_ROLE)
    logits = zero_shot_logits(model, tok, list(test["text"]), zcfg["prompt"], zcfg["verbalizers"], ctx.device,
                              tcfg["eval_batch_size"], tcfg["max_length"] + 32)
    preds = build_pred_frame(test, logits)
    preds.to_parquet(d / "test_preds.parquet", index=False)
    _free(model)
    return {"test_acc": float(preds["correct"].mean())}


def _run_train(job: Job, ctx: RunContext, d: Path, source: JobResult | None) -> dict:
    cfg, task = ctx.cfg, ctx.task
    tcfg, mcfg = cfg["train"], ctx.model_cfg(job.model)
    C = len(task.label_names)
    model, tok = load_classifier(mcfg, C, ctx.device)
    side = tcfg.get("truncation_side", "right")
    train_df, val_df = task.split(job.train_split), task.split(VAL_ROLE)
    train_ids = tokenize(tok, list(train_df["text"]), tcfg["max_length"], side)
    val_ids = tokenize(tok, list(val_df["text"]), tcfg["max_length"], side)

    if job.is_weak:
        assert source is not None and source.status == "done"
        teacher_labels = source.labels(job.train_split, task.label_names)
        targets = targets_from_teacher(train_df, teacher_labels, C, tcfg["loss"])
    else:
        targets = targets_from_gt(train_df["label"].to_numpy(), C)

    sel = selection_mode(job, tcfg)
    if sel == "val_gt":
        val_labels = val_df["label"].to_numpy()
    elif sel == "val_teacher":
        tv = source.labels(VAL_ROLE, task.label_names).set_index("example_id").loc[val_df["example_id"]]
        val_labels = tv["hard_label"].to_numpy()
    else:
        val_labels = None

    conf = tcfg.get("conf_loss") if job.is_weak else None
    tinfo = train_model(model, tok, train_ids, targets, val_ids, val_labels, mcfg=mcfg, tcfg=tcfg,
                        device=ctx.device, seed=job.seed, conf=conf)
    test = task.split(TEST_ROLE)
    logits = predict(model, tokenize(tok, list(test["text"]), tcfg["max_length"], side), tok.pad_token_id,
                     ctx.device, tcfg["eval_batch_size"])
    preds = build_pred_frame(test, logits)
    preds.to_parquet(d / "test_preds.parquet", index=False)
    _write_labels(model, tok, ctx, job, d, list(job.label_splits))
    if tcfg.get("save_checkpoints"):
        _save_checkpoint(model, tok, d)
    info = {"test_acc": float(preds["correct"].mean()), "selection": sel, "n_train": len(train_df),
            "n_params": count_params(model), "conf_loss": conf, **tinfo}
    write_json(d / "train_log.json", info)
    _free(model)
    return info


def _write_labels(model, tok, ctx: RunContext, job: Job, d: Path, splits: list[str]) -> None:
    tcfg = ctx.cfg["train"]
    mcfg = ctx.model_cfg(job.model)
    for split in splits:
        if split == TEST_ROLE:
            raise AssertionError("Refusing to generate training labels on the test split")
        df = ctx.task.split(split)
        logits = predict(model, tokenize(tok, list(df["text"]), tcfg["max_length"], tcfg.get("truncation_side", "right")),
                         tok.pad_token_id, ctx.device, tcfg["eval_batch_size"])
        frame = build_label_frame(df, logits, mcfg["id"], job.name, job.generation)
        save_labels(frame, d / f"labels_{split}.parquet", ctx.task.label_names)
        log(f"    labeled {split}: teacher acc {frame['teacher_correct'].mean():.4f} (n={len(frame)})")


def _save_checkpoint(model, tok, d: Path) -> None:
    ck = d / "checkpoint"
    model.to(torch.bfloat16).save_pretrained(ck, safe_serialization=True)
    tok.save_pretrained(ck)


def _load_checkpoint(ctx: RunContext, job: Job, d: Path):
    from transformers import AutoModelForSequenceClassification

    from .models import load_tokenizer

    mcfg = ctx.model_cfg(job.model)
    ck = d / "checkpoint"
    if mcfg.get("lora"):
        from peft import PeftModel

        base, tok = load_classifier({**mcfg, "lora": False}, len(ctx.task.label_names), ctx.device)
        model = PeftModel.from_pretrained(base, ck).to(ctx.device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(ck, dtype=torch.float32).to(ctx.device)
        tok = load_tokenizer(str(ck))
    return model, tok


def _free(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
