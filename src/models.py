"""Model / tokenizer loading. Classification uses a sequence-classification head on the last token."""
from __future__ import annotations

import torch

from .utils import log

try:  # keep job logs readable: drop per-load weight reports ("score.weight MISSING" is expected)
    from transformers.utils import logging as _hf_logging

    _hf_logging.set_verbosity_error()
    _hf_logging.disable_progress_bar()
except Exception:  # pragma: no cover
    pass


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_classifier(mcfg: dict, num_labels: int, device: str):
    """Load a seq-classification model in fp32 (autocast handles bf16 compute on GPU)."""
    from transformers import AutoModelForSequenceClassification

    tok = load_tokenizer(mcfg["id"])
    model = AutoModelForSequenceClassification.from_pretrained(
        mcfg["id"], num_labels=num_labels, dtype=torch.float32
    )
    model.config.pad_token_id = tok.pad_token_id
    # Zero-init the new linear head (as in OpenAI's weak-to-strong code): the pretrained last-token
    # hidden states have large norms, so a random head starts with huge, arbitrary logits.
    head = getattr(model, "score", None) or getattr(model, "classifier", None)
    if isinstance(head, torch.nn.Linear):
        torch.nn.init.zeros_(head.weight)
        if head.bias is not None:
            torch.nn.init.zeros_(head.bias)
    if mcfg.get("lora"):
        from peft import LoraConfig, TaskType, get_peft_model

        lcfg = mcfg.get("lora_config", {})
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lcfg.get("r", 16),
            lora_alpha=lcfg.get("alpha", 32),
            lora_dropout=lcfg.get("dropout", 0.05),
            target_modules=lcfg.get("target_modules"),
        ))
    if mcfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    return model.to(device), tok


def load_causal_lm(model_id: str, device: str):
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return model, load_tokenizer(model_id)


def verify_models(models: list[dict]) -> None:
    """Fail fast (before any weight download) if a model ID is missing or lacks a seq-cls head."""
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES

    for m in models:
        conf = AutoConfig.from_pretrained(m["id"])
        if conf.model_type not in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES:
            raise ValueError(f"{m['id']} ({conf.model_type}) has no AutoModelForSequenceClassification support")
        log(f"  verified {m['name']:>8s} = {m['id']} ({conf.model_type})")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
