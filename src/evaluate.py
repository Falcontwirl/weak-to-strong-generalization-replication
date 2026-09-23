"""Held-out test predictions and the zero-shot (no task training) baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .train import softmax_np


def build_pred_frame(test_df: pd.DataFrame, logits: np.ndarray) -> pd.DataFrame:
    probs = softmax_np(logits.astype(np.float64))
    df = pd.DataFrame({"example_id": test_df["example_id"].to_numpy(), "pred": probs.argmax(-1)})
    for k in range(probs.shape[1]):
        df[f"prob_{k}"] = probs[:, k]
    df["label"] = test_df["label"].to_numpy()
    df["correct"] = df["pred"] == df["label"]
    return df


@torch.no_grad()
def zero_shot_logits(model, tok, texts: list[str], prompt: str, verbalizers: list[str], device: str,
                     batch_size: int, max_length: int) -> np.ndarray:
    """Score each label by the next-token logit of the first token of its verbalizer."""
    vids = []
    for v in verbalizers:
        ids = tok(v, add_special_tokens=False)["input_ids"]
        vids.append(ids[0])
    if len(set(vids)) != len(vids):
        raise ValueError(f"Verbalizers {verbalizers} share a first token")
    out = []
    prompts = [prompt.format(text=t.strip()) for t in texts]
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(device)
        logits = model(**enc).logits
        last = enc["attention_mask"].sum(-1) - 1
        lg = logits[torch.arange(len(last), device=device), last][:, vids]
        out.append(lg.float().cpu())
    return torch.cat(out).numpy()
