"""Decide whether this machine can run a task, and warn the user if it probably cannot."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .._utils.warn import warn
from ..exceptions import ConfigurationError, HardwareInsufficientError, SimchologyHardwareWarning
from .device import detect_hardware, device_info
from .estimate import GB, MemoryEstimate, estimate_memory, estimate_runtime, format_duration
from .models import model_parameter_count, model_shape

SLOW_SECONDS = 6 * 3600


@dataclass
class ResourceRequest:
    """What a method is about to do, so its cost can be checked before it starts."""

    task: str  # "inference" | "full_finetune" | "lora"
    model_id: str | None = None
    n_params: int | None = None
    seq_len: int = 512
    batch_size: int = 8
    n_items: int = 0  # number of texts or pairs to process (per epoch)
    epochs: int = 1
    hidden_size: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    gradient_checkpointing: bool = False
    lora_fraction: float = 0.01
    weight_bytes: float | None = None
    extra_bytes: int = 0
    description: str = ""  # e.g. "fine-tuning the cross-encoder 'jhu-clsp/mmBERT-base'"


@dataclass
class Verdict:
    level: str  # "ok" | "tight" | "slow" | "insufficient" | "unknown"
    message: str
    suggestions: list[str] = field(default_factory=list)
    estimate: MemoryEstimate | None = None
    available_bytes: int | None = None
    seconds: float | None = None
    device: str = "cpu"

    def __str__(self) -> str:
        text = self.message
        if self.suggestions:
            text += "\nSuggestions:\n" + "\n".join(f"  - {s}" for s in self.suggestions)
        return text


def _validate_policy(policy: str) -> None:
    if policy not in ("warn", "error", "off"):
        raise ConfigurationError("check_hardware must be 'warn', 'error' or 'off'")


def emit(verdict: Verdict, policy: str = "warn", stacklevel: int = 3) -> Verdict:
    """Turn a verdict into a warning (``policy="warn"``) or an error (``policy="error"``)."""
    _validate_policy(policy)
    if policy == "off" or verdict.level in ("ok", "unknown"):
        return verdict
    if verdict.level == "insufficient" and policy == "error":
        raise HardwareInsufficientError(str(verdict))
    warn(str(verdict), SimchologyHardwareWarning)
    return verdict


def _suggestions(req: ResourceRequest, kind: str) -> list[str]:
    out: list[str] = []
    if req.task == "full_finetune":
        out.append("use method='lora_classifier' (trains <1% of the weights)")
        if not req.gradient_checkpointing:
            out.append("set method_params={'gradient_checkpointing': True}")
    if req.seq_len > 256:
        out.append(f"lower max_length (currently {req.seq_len}), e.g. method_params={{'max_length': 256}}")
    if req.batch_size > 1:
        out.append("lower batch_size (use gradient accumulation to keep the effective batch size)")
    if kind == "cpu":
        out.append("run on a machine with a CUDA GPU or an Apple-silicon Mac")
    else:
        out.append("close other GPU-heavy applications, or use a machine with more GPU memory")
    return out


def check_resources(req: ResourceRequest, device: str = "auto", policy: str = "warn") -> Verdict:
    """Estimate memory/runtime of ``req`` on ``device`` and warn according to ``policy``."""
    _validate_policy(policy)
    if policy == "off":
        return Verdict("ok", "hardware check disabled")
    info = device_info(device)
    n_params = req.n_params
    if n_params is None and req.model_id:
        n_params = model_parameter_count(req.model_id).total
    if not n_params:
        return Verdict("unknown", f"Could not determine the size of {req.model_id!r}; "
                       "skipping the hardware check.", device=info.torch_device)
    hidden, layers, heads = req.hidden_size, req.n_layers, req.n_heads
    if req.model_id and not (hidden and layers):
        shape = model_shape(req.model_id)
        hidden, layers, heads = hidden or shape.hidden_size, layers or shape.n_layers, heads or shape.n_heads
    weight_bytes = req.weight_bytes
    if weight_bytes is None and req.task == "inference":
        weight_bytes = 2 if info.kind in ("cuda", "mps") else 4
    est = estimate_memory(
        req.task, n_params, weight_bytes=weight_bytes, batch_size=req.batch_size,
        seq_len=req.seq_len, hidden_size=hidden, n_layers=layers, n_heads=heads,
        lora_fraction=req.lora_fraction, gradient_checkpointing=req.gradient_checkpointing,
        extra_bytes=req.extra_bytes,
    )
    available = info.available_memory or 0
    budget = info.total_memory or available
    what = req.description or f"{req.task} with {req.model_id or 'the model'}"
    where = "this computer's RAM" if info.kind == "cpu" else f"the {info.name} ({info.torch_device})"
    shared = " (shared with the rest of the system)" if info.unified_memory else ""
    n_tokens = req.n_items * req.epochs * req.seq_len
    seconds = estimate_runtime(req.task, n_params, n_tokens, info.kind) if n_tokens else None
    base = dict(estimate=est, available_bytes=available, seconds=seconds, device=info.torch_device)
    if budget and est.total_bytes > budget:
        return Verdict(
            "insufficient",
            f"{what[0].upper()}{what[1:]} needs roughly {est.total_gb:.1f} GB of memory, but "
            f"{where} can use at most about {budget / GB:.1f} GB{shared}. It will probably "
            "fail with an out-of-memory error or be extremely slow.",
            _suggestions(req, info.kind), **base,
        )
    training = req.task in ("full_finetune", "lora")
    if (seconds is not None and seconds > SLOW_SECONDS) or (
            training and info.kind == "cpu" and (seconds is None or seconds > 600)):
        eta = f" (very rough estimate: {format_duration(seconds)})" if seconds else ""
        on = "this computer's CPU" if info.kind == "cpu" else where
        return Verdict(
            "slow",
            f"{what[0].upper()}{what[1:]} will be slow on {on}{eta}.",
            _suggestions(req, info.kind)[-1:], **base,
        )
    if available and est.total_bytes > 0.8 * available:
        return Verdict(
            "tight",
            f"{what[0].upper()}{what[1:]} needs roughly {est.total_gb:.1f} GB; about "
            f"{available / GB:.1f} GB is free right now on {where}{shared}. Close other "
            "memory-heavy applications if it fails or is very slow.",
            _suggestions(req, info.kind), **base,
        )
    eta = f"; estimated time {format_duration(seconds)}" if seconds else ""
    return Verdict("ok", f"{what} fits ({est.total_gb:.1f} GB of {available / GB:.1f} GB{eta}).", **base)


def check_ollama_model(
    model: str,
    size_bytes: int | None,
    *,
    num_ctx: int = 4096,
    n_layers: int | None = None,
    hidden_size: int | None = None,
    policy: str = "warn",
) -> Verdict:
    """Check that an Ollama model (plus its context cache) fits in GPU/unified memory."""
    _validate_policy(policy)
    report = detect_hardware()
    best = report.best
    if not size_bytes:
        return Verdict("unknown", f"Size of Ollama model {model!r} unknown; skipping check.")
    # KV cache: 2 (K,V) x layers x ctx x hidden x 2 bytes; fall back to ~10% of weights
    kv = 2 * n_layers * num_ctx * hidden_size * 2 if (n_layers and hidden_size) else int(0.1 * size_bytes)
    need = int((size_bytes + kv) * 1.1 + 0.5 * GB)
    est = MemoryEstimate(need, {"weights": size_bytes, "context cache": kv})
    if best.kind == "cpu":
        available = report.ram_available
        where = "system RAM (no supported GPU found, so Ollama will run on the CPU)"
    else:
        available = best.available_memory or report.ram_available
        where = f"{best.name}"
    if need > available:
        level = "insufficient" if best.kind == "cpu" or need > report.ram_available else "slow"
        msg = (f"Ollama model {model!r} needs about {need / GB:.1f} GB but only "
               f"{available / GB:.1f} GB is available on {where}.")
        if level == "slow":
            msg += " Ollama will offload part of it to the CPU, which is much slower."
        verdict = Verdict(level, msg, ["use a smaller model (e.g. a 1-4B parameter model)",
                                       "lower num_ctx", "close other memory-heavy applications"],
                          est, available)
    elif best.kind == "cpu":
        verdict = Verdict("slow", f"Ollama model {model!r} will run on the CPU, which can take "
                          "several seconds per judgment.", [], est, available)
    else:
        verdict = Verdict("ok", f"Ollama model {model!r} fits ({need / GB:.1f} GB).", [], est, available)
    return emit(verdict, policy, stacklevel=4)


def recommend_kev_model() -> str:
    """Suggest the Kev model size that should run on this machine."""
    report = detect_hardware()
    best = report.best
    if best.kind == "cuda" and (best.total_memory or 0) >= 40 * GB:
        return "jaredpalmer/kev-9b"
    if (best.kind == "mps" and report.ram_total >= 32 * GB) or (
        best.kind == "cuda" and (best.total_memory or 0) >= 20 * GB
    ):
        return "jaredpalmer/kev-4b"
    return "jaredpalmer/kev-0.8b"


def check_disk_space(size_bytes: int | None, path: str | Path | None = None,
                     policy: str = "warn") -> Verdict:
    """Warn before downloading ``size_bytes`` into a nearly full disk."""
    if not size_bytes:
        return Verdict("unknown", "download size unknown")
    target = Path(path) if path else Path.home()
    while not target.exists() and target != target.parent:
        target = target.parent
    free = shutil.disk_usage(target).free
    if size_bytes * 1.1 > free:
        verdict = Verdict("insufficient", f"Downloading the model needs about {size_bytes / GB:.1f} GB "
                          f"of disk space, but only {free / GB:.1f} GB is free.",
                          ["free disk space or set HF_HOME to a larger disk"])
        return emit(verdict, policy)
    return Verdict("ok", "enough disk space")
