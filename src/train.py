"""Training loop, datasets and prediction.

Training targets are always a probability vector per example (``target_probs``): one-hot ground
truth for GT jobs, the teacher's soft labels (``soft_ce``) or one-hot teacher labels (``hard_ce``)
for weak jobs. Dataset items never contain the ground-truth label of a weakly-supervised example.
"""
from __future__ import annotations

import contextlib
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .utils import log

BATCH_KEYS = {"input_ids", "attention_mask", "target_probs"}


class ClassificationDataset(Dataset):
    """Tokenized texts plus a target distribution. Holds nothing else."""

    def __init__(self, input_ids: list[list[int]], target_probs: np.ndarray | None = None):
        self.input_ids = input_ids
        self.target_probs = None if target_probs is None else np.asarray(target_probs, dtype=np.float32)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        item = {"input_ids": self.input_ids[i]}
        if self.target_probs is not None:
            item["target_probs"] = self.target_probs[i]
        return item


def make_collate(pad_id: int):
    def collate(items):
        L = max(len(x["input_ids"]) for x in items)
        ids = torch.full((len(items), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(items), L), dtype=torch.long)
        for i, x in enumerate(items):
            n = len(x["input_ids"])
            ids[i, :n] = torch.tensor(x["input_ids"])
            mask[i, :n] = 1
        batch = {"input_ids": ids, "attention_mask": mask}
        if "target_probs" in items[0]:
            batch["target_probs"] = torch.tensor(np.stack([x["target_probs"] for x in items]))
        assert set(batch) <= BATCH_KEYS, f"unexpected batch keys {set(batch)}"
        return batch

    return collate


def tokenize(tok, texts: list[str], max_length: int, truncation_side: str = "right") -> list[list[int]]:
    # "left" keeps the end of long inputs, e.g. the question and "Answer:" after a long passage.
    tok.truncation_side = truncation_side
    enc = tok([t.strip() for t in texts], truncation=True, max_length=max_length, add_special_tokens=True)
    return [ids if len(ids) else [tok.eos_token_id] for ids in enc["input_ids"]]


def targets_from_gt(labels: np.ndarray, num_labels: int) -> np.ndarray:
    return np.eye(num_labels, dtype=np.float32)[labels]


def targets_from_teacher(split_df: pd.DataFrame, labels_df: pd.DataFrame, num_labels: int, loss: str) -> np.ndarray:
    """Align teacher labels to ``split_df`` by example_id. Only teacher outputs are read."""
    lab = labels_df.set_index("example_id")
    missing = set(split_df["example_id"]) - set(lab.index)
    if missing:
        raise AssertionError(f"Teacher labels missing for {len(missing)} examples of the student split")
    lab = lab.loc[split_df["example_id"]]
    probs = lab[[f"prob_{k}" for k in range(num_labels)]].to_numpy(np.float32)
    if loss == "soft_ce":
        return probs
    if loss == "hard_ce":
        return targets_from_gt(lab["hard_label"].to_numpy(), num_labels)
    raise ValueError(f"unknown loss {loss}")


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


def conf_coef(step: int, total: int, conf: dict) -> float:
    """Weight of the self-confidence term.

    schedule "linear" (default): ramp from 0 to ``alpha`` over ``warmup_frac`` of training.
    schedule "ref": as in openai/weak-to-strong ``logconf_loss_fn``: ``alpha * step_frac`` while
    step_frac <= warmup_frac (so at most alpha * warmup_frac), then jump to ``alpha``.
    """
    alpha, frac = float(conf["alpha"]), step / max(1, total)
    schedule = conf.get("schedule", "linear")
    if schedule == "ref":
        return alpha * (frac if frac <= conf.get("warmup_frac", 0.1) else 1.0)
    if schedule != "linear":
        raise ValueError(f"unknown conf_loss schedule {schedule}")
    warm = conf.get("warmup_frac", 0.2) * total
    ramp = 1.0 if warm <= 0 else min(1.0, step / warm)
    return alpha * ramp


def conf_targets(logits: torch.Tensor, weak_probs: torch.Tensor, coef: float) -> torch.Tensor:
    """Auxiliary confidence loss targets (Burns et al., 2023), binary tasks only.

    CE is linear in the target, so (1 - a) CE(f, weak) + a CE(f, hardened f) = CE(f, mixed target).
    The student's predictions are hardened with a batch-adaptive threshold chosen so the fraction
    predicted positive matches the weak labels' mean positive probability (as in the paper's code).
    """
    if weak_probs.shape[-1] != 2:
        raise ValueError("conf_loss supports binary tasks only")
    p1 = torch.softmax(logits.detach().float(), dim=-1)[:, 1]
    thr = torch.quantile(p1, 1.0 - weak_probs[:, 1].mean())
    hard1 = (p1 > thr).float()
    hardened = torch.stack([1.0 - hard1, hard1], dim=-1)
    return (1.0 - coef) * weak_probs + coef * hardened


def make_optimizer(params, mcfg: dict, tcfg: dict, device: str):
    name = mcfg.get("optimizer", "adamw")
    if name == "adamw":
        return torch.optim.AdamW(params, lr=float(mcfg["lr"]), weight_decay=tcfg["weight_decay"],
                                 fused=(device == "cuda"))
    if name == "adamw8bit":
        # 8-bit optimizer state (bitsandbytes): ~2 bytes/param instead of 8, so 2.8B full fine-tuning fits 48 GB.
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=float(mcfg["lr"]), weight_decay=tcfg["weight_decay"])
    raise ValueError(f"unknown optimizer {name}")


def autocast_ctx(device: str):
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict(model, input_ids: list[list[int]], pad_id: int, device: str, batch_size: int) -> np.ndarray:
    """Return logits [N, C] (float32). Sorted by length for efficiency, returned in original order."""
    model.eval()
    order = np.argsort([len(x) for x in input_ids])
    ds = ClassificationDataset([input_ids[i] for i in order])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=make_collate(pad_id))
    outs = []
    for batch in dl:
        with autocast_ctx(device):
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        outs.append(out.logits.float().cpu())
    logits = torch.cat(outs).numpy()
    res = np.empty_like(logits)
    res[order] = logits
    return res


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def train_model(model, tok, train_ids: list[list[int]], train_targets: np.ndarray,
                val_ids: list[list[int]], val_labels: np.ndarray | None, *,
                mcfg: dict, tcfg: dict, device: str, seed: int, conf: dict | None = None) -> dict:
    """Train in place; restore the best val state. ``val_labels`` None => no selection (final state).

    ``val_labels`` are hard labels to score val accuracy against: ground truth for GT jobs,
    teacher hard labels for weak jobs (so weak jobs never see ground truth).
    ``conf`` ({alpha, warmup_frac}) enables the auxiliary confidence loss; the caller passes it for weak jobs only.
    """
    bs = mcfg.get("batch_size", 32)
    accum = mcfg.get("grad_accum", tcfg.get("grad_accum", 1))
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ClassificationDataset(train_ids, train_targets), batch_size=bs, shuffle=True,
                    collate_fn=make_collate(tok.pad_token_id), generator=g, drop_last=False,
                    num_workers=tcfg.get("num_workers", 0))
    epochs = mcfg.get("epochs", tcfg["epochs"])
    steps_per_epoch = math.ceil(len(dl) / accum)
    total = steps_per_epoch * epochs
    eval_every = max(1, steps_per_epoch // max(1, tcfg["evals_per_epoch"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(params, mcfg, tcfg, device)
    warm = int(total * tcfg["warmup_frac"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    history, best = [], {"score": -1.0, "step": 0, "state": None}

    def evaluate(step):
        if val_labels is None:
            return
        logits = predict(model, val_ids, tok.pad_token_id, device, tcfg["eval_batch_size"])
        score = float((logits.argmax(-1) == val_labels).mean())
        history.append({"step": step, "val_score": score})
        if score > best["score"]:
            best.update(score=score, step=step,
                        state={k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
        model.train()

    model.train()
    step, t0, running = 0, time.time(), []
    for _ in range(epochs):
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_ctx(device):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            target = batch["target_probs"]
            if conf is not None:
                target = conf_targets(out.logits, target, conf_coef(step, total, conf))
            loss = soft_cross_entropy(out.logits, target) / accum
            loss.backward()
            running.append(loss.item() * accum)
            if (i + 1) % accum == 0 or i + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % eval_every == 0 or step == total:
                    evaluate(step)
                    log(f"    step {step}/{total} loss {np.mean(running[-50:]):.4f}"
                        + (f" val {history[-1]['val_score']:.4f}" if history else ""))
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"history": history, "best_step": best["step"] if best["state"] is not None else step,
            "best_val_score": best["score"] if best["state"] is not None else None,
            "total_steps": total, "train_seconds": time.time() - t0,
            "final_loss": float(np.mean(running[-50:])) if running else None}
