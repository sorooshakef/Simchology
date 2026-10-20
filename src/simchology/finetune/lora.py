"""LoRA fine-tuning of decoder language models as sequence classifiers (PEFT)."""

from __future__ import annotations

import re
from pathlib import Path

from .._optional import is_available, require
from .._utils.warn import warn
from ..exceptions import SimchologyHardwareWarning
from .trainer import load_sequence_classifier, load_tokenizer, quiet_transformers


def prepare_tokenizer(tokenizer) -> bool:
    """Make sure the tokenizer has a padding token; return True if a new token was added."""
    if tokenizer.pad_token is not None:
        return False
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return False
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
        return False
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return True


def set_pad_token_id(model, pad_token_id: int | None) -> None:
    """Tell the classifier which token is padding, so it finds each sequence's last token.

    Vision-language models (e.g. Qwen3.5) read it from their nested text config, which
    the top-level ``config.pad_token_id`` does not reach. (During training the transformers
    Trainer copies it there itself, so a missing value only shows up after loading.)
    """
    model.config.pad_token_id = pad_token_id
    get_text_config = getattr(model.config, "get_text_config", None)
    text_config = get_text_config() if callable(get_text_config) else None
    if text_config is not None and text_config is not model.config:
        text_config.pad_token_id = pad_token_id


def _quantization_config(load_in_4bit: bool, device: str):
    if not load_in_4bit:
        return None
    if not device.startswith("cuda") or not is_available("bitsandbytes"):
        warn(
            "load_in_4bit=True needs a CUDA GPU and the 'bitsandbytes' package; loading the "
            "model in normal precision instead.", SimchologyHardwareWarning,
        )
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_quant_type="nf4")


# Vision encoders of vision-language backbones (e.g. Qwen3.5's ``model.visual``). Text-only
# training never sends gradients through them, so adapters there only cost memory.
VISION_MODULES = r"(.*\.)?(visual|vision_tower|vision_model)(\..*)?"


def has_vision_modules(model) -> bool:
    return any(re.fullmatch(VISION_MODULES, name) for name, _ in model.named_modules())


def _base_dtype(device: str):
    torch = require("torch", feature="LoRA fine-tuning")
    from ..hardware.device import device_info

    info = device_info(device)
    if info.kind in ("cuda", "mps", "xpu") and info.bf16:
        return torch.bfloat16
    return torch.float32


def build_lora_model(backbone: str, *, num_labels: int, regression: bool, params: dict, device: str):
    """Load ``backbone`` as a sequence classifier and wrap it with LoRA adapters."""
    peft = require("peft", feature="LoRA fine-tuning (method 'lora_classifier')", extra="finetune")
    tokenizer = load_tokenizer(backbone, revision=params.get("revision"),
                               trust_remote_code=params.get("trust_remote_code", False))
    added = prepare_tokenizer(tokenizer)
    quant = _quantization_config(bool(params.get("load_in_4bit")), device)
    extra = {"quantization_config": quant} if quant is not None else {}
    model = load_sequence_classifier(
        backbone, num_labels=num_labels, regression=regression, revision=params.get("revision"),
        trust_remote_code=params.get("trust_remote_code", False),
        dtype=None if quant is not None else _base_dtype(device), **extra,
    )
    if added:
        model.resize_token_embeddings(len(tokenizer))
    set_pad_token_id(model, tokenizer.pad_token_id)
    if quant is not None:
        model = peft.prepare_model_for_kbit_training(model)
    if params.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
    target = params["target_modules"]
    extra_config = {}
    if isinstance(target, str) and target.lower() == "all-linear" and has_vision_modules(model):
        extra_config["exclude_modules"] = VISION_MODULES
    config = peft.LoraConfig(
        task_type=peft.TaskType.SEQ_CLS, r=int(params["r"]), lora_alpha=int(params["alpha"]),
        lora_dropout=float(params["dropout"]), target_modules=target, **extra_config,
    )
    with quiet_transformers():
        model = peft.get_peft_model(model, config)
    head = [n for n, p in model.named_parameters() if p.requires_grad and ("score" in n or "classifier" in n)]
    if not head:
        raise RuntimeError("The classification head of the LoRA model is not trainable.")
    return model, tokenizer, added


def load_lora_model(backbone: str, adapter_dir: Path, *, num_labels: int, regression: bool,
                    params: dict, added_pad_token: bool, device: str, merge: bool = True):
    """Rebuild the base classifier and load the saved adapter (and head)."""
    peft = require("peft", feature="LoRA models", extra="finetune")
    tokenizer = load_tokenizer(str(adapter_dir))
    model = load_sequence_classifier(
        backbone, num_labels=num_labels, regression=regression, revision=params.get("revision"),
        trust_remote_code=params.get("trust_remote_code", False), dtype=_base_dtype(device),
    )
    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))
    set_pad_token_id(model, tokenizer.pad_token_id)
    with quiet_transformers():
        model = peft.PeftModel.from_pretrained(model, str(adapter_dir))
    if merge:
        model = model.merge_and_unload()
    return model, tokenizer
