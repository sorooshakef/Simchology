"""LoRA fine-tuning of decoder language models as sequence classifiers (PEFT)."""

from __future__ import annotations

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
    model.config.pad_token_id = tokenizer.pad_token_id
    if quant is not None:
        model = peft.prepare_model_for_kbit_training(model)
    if params.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
    config = peft.LoraConfig(
        task_type=peft.TaskType.SEQ_CLS, r=int(params["r"]), lora_alpha=int(params["alpha"]),
        lora_dropout=float(params["dropout"]), target_modules=params["target_modules"],
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
    model.config.pad_token_id = tokenizer.pad_token_id
    with quiet_transformers():
        model = peft.PeftModel.from_pretrained(model, str(adapter_dir))
    if merge:
        model = model.merge_and_unload()
    return model, tokenizer
