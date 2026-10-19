"""Rough memory and runtime estimates for neural models.

These are deliberately simple, documented heuristics. They are good enough to tell a
user "this will not fit" or "this will take many hours", not to plan to the megabyte.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GB = 1024**3

# Effective training/inference throughput in FLOP/s. The MPS value is calibrated on a
# real run: fine-tuning mmBERT-base (307M parameters) on 2,328 pairs x 7 epochs at
# max_length 1024 took ~2.3 h on an Apple-silicon GPU.
EFFECTIVE_FLOPS = {"cpu": 0.45e12, "mps": 3.5e12, "xpu": 5e12, "cuda": 25e12}


@dataclass
class MemoryEstimate:
    total_bytes: int
    breakdown: dict[str, int] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / GB


def _approx_shape(n_params: int, hidden_size: int | None, n_layers: int | None):
    """Approximate hidden size and depth from the parameter count (P ~ 12 L h^2, L ~ h/64)."""
    if hidden_size and n_layers:
        return hidden_size, n_layers, False
    h = hidden_size or int(round((64 * n_params / 12) ** (1 / 3) / 64) * 64) or 64
    layers = n_layers or max(2, int(round(n_params / (12 * h * h))))
    return h, layers, True


def estimate_memory(
    task: str,
    n_params: int,
    *,
    weight_bytes: float | None = None,
    batch_size: int = 8,
    seq_len: int = 512,
    hidden_size: int | None = None,
    n_layers: int | None = None,
    n_heads: int | None = None,
    lora_fraction: float = 0.01,
    gradient_checkpointing: bool = False,
    extra_bytes: int = 0,
) -> MemoryEstimate:
    """Estimate peak memory for ``task`` in {"inference", "full_finetune", "lora"}."""
    assumptions: list[str] = []
    h, layers, approximated = _approx_shape(n_params, hidden_size, n_layers)
    heads = n_heads or max(1, h // 64)
    if approximated:
        assumptions.append(f"model shape approximated from size (hidden={h}, layers={layers})")
    act_bytes = 2  # activations in 16-bit precision
    breakdown: dict[str, int] = {}
    if task == "inference":
        wb = weight_bytes if weight_bytes is not None else 4
        breakdown["weights"] = int(n_params * wb)
        breakdown["activations"] = int(batch_size * seq_len * h * 8 * act_bytes)
        breakdown["attention"] = int(batch_size * heads * seq_len * seq_len * act_bytes)
        assumptions.append(f"weights stored with {wb:g} bytes per parameter")
    elif task in ("full_finetune", "lora"):
        if task == "full_finetune":
            # fp32 master weights (4) + gradients (4) + AdamW moments (8)
            breakdown["weights+gradients+optimizer"] = int(16 * n_params)
            assumptions.append("AdamW with fp32 weights, gradients and two moment buffers")
        else:
            wb = weight_bytes if weight_bytes is not None else 2
            trainable = int(n_params * lora_fraction)
            breakdown["frozen weights"] = int(n_params * wb)
            breakdown["adapter weights+gradients+optimizer"] = 16 * trainable
            assumptions.append(f"LoRA adapters ~{lora_fraction:.0%} of parameters")
        if gradient_checkpointing:
            per_layer = batch_size * seq_len * h * 2
            full_layer = batch_size * seq_len * h * (34 + 5 * heads * seq_len / h)
            breakdown["activations"] = int(layers * per_layer + full_layer)
            assumptions.append("gradient checkpointing enabled")
        else:
            # Korthikanti et al. (2022): bytes per layer for 16-bit activations
            breakdown["activations"] = int(layers * batch_size * seq_len * h * (34 + 5 * heads * seq_len / h))
    else:
        raise ValueError(f"unknown task {task!r}")
    subtotal = sum(breakdown.values()) + int(extra_bytes)
    overhead = int(0.2 * subtotal + 0.3 * GB)
    breakdown["framework overhead"] = overhead
    if extra_bytes:
        breakdown["other"] = int(extra_bytes)
    assumptions.append("+20% and 0.3 GB framework overhead")
    return MemoryEstimate(int(subtotal + overhead), breakdown, assumptions)


def estimate_runtime(
    task: str, n_params: int, n_tokens: int, device_kind: str
) -> float | None:
    """Very rough wall-clock seconds for processing ``n_tokens`` tokens.

    Training costs ~6 FLOPs per parameter and token (forward + backward), inference ~2.
    """
    flops_per_token = 6 if task in ("full_finetune", "lora") else 2
    if task == "lora":
        flops_per_token = 4  # no weight gradients for the frozen backbone
    throughput = EFFECTIVE_FLOPS.get(device_kind.split(":")[0])
    if throughput is None or n_params <= 0:
        return None
    return flops_per_token * n_params * n_tokens / throughput


def format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 2 * 86400:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"
