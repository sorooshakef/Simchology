"""Find out how large a model is without downloading its weights."""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from .._optional import is_available

# Fallback sizes of Simchology's default models, used when offline.
KNOWN_PARAM_COUNTS: dict[str, int] = {
    "jhu-clsp/mmBERT-base": 307_000_000,
    "jhu-clsp/mmBERT-small": 140_000_000,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 278_000_000,
    "sentence-transformers/distiluse-base-multilingual-cased-v2": 135_000_000,
    "bert-base-multilingual-cased": 178_000_000,
    "xlm-roberta-large": 560_000_000,
    "Qwen/Qwen3-0.6B": 752_000_000,
    "Qwen/Qwen3-Embedding-0.6B": 596_000_000,
}

_DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
                "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


@dataclass
class ParamInfo:
    total: int | None
    size_bytes: int | None
    source: str  # "safetensors-local" | "hub-metadata" | "config" | "known" | "name" | "unknown"


@dataclass
class ModelShape:
    hidden_size: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    max_positions: int | None = None


def _read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n))


def _weight_files(directory: Path) -> list[Path]:
    """Safetensors files of a model folder, including Sentence-Transformers modules (e.g. 2_Dense)."""
    files = sorted(directory.glob("*.safetensors"))
    try:
        modules = json.loads((directory / "modules.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        modules = []
    for module in modules if isinstance(modules, list) else []:
        sub = module.get("path") if isinstance(module, dict) else None
        if sub:
            files += sorted((directory / sub).glob("*.safetensors"))
    return files


def _count_local(directory: Path) -> ParamInfo | None:
    files = _weight_files(directory)
    if not files:
        return None
    total = 0
    size = 0
    for f in files:
        header = _read_safetensors_header(f)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            n = 1
            for dim in meta["shape"]:
                n *= int(dim)
            total += n
            size += n * _DTYPE_BYTES.get(meta.get("dtype", "F32"), 4)
    return ParamInfo(total, size, "safetensors-local")


def _cached_snapshot(model_id: str, revision: str | None) -> Path | None:
    if not is_available("huggingface_hub"):
        return None
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    except Exception:
        pass
    # huggingface_hub >= 1.0 rejects snapshots that lack files skipped on purpose (e.g. ONNX)
    try:
        from huggingface_hub import try_to_load_from_cache

        config = try_to_load_from_cache(model_id, "config.json", revision=revision)
        return Path(config).parent if isinstance(config, str) else None
    except Exception:
        return None


def _hub_metadata(model_id: str, revision: str | None) -> ParamInfo | None:
    if not is_available("huggingface_hub") or os.environ.get("HF_HUB_OFFLINE") == "1":
        return None
    try:
        from huggingface_hub import get_safetensors_metadata

        meta = get_safetensors_metadata(model_id, revision=revision)
        counts = meta.parameter_count  # dtype -> number of parameters
        total = int(sum(counts.values()))
        size = int(sum(n * _DTYPE_BYTES.get(dt, 4) for dt, n in counts.items()))
        return ParamInfo(total, size, "hub-metadata")
    except Exception:
        return None


def _from_config(model_id: str, revision: str | None) -> ParamInfo | None:
    if not (is_available("transformers") and is_available("torch")):
        return None
    try:
        import torch
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=True)
        with torch.device("meta"):
            model = AutoModel.from_config(config)
        total = int(sum(p.numel() for p in model.parameters()))
        return ParamInfo(total, total * 4, "config")
    except Exception:
        return None


def parse_size_from_name(name: str) -> int | None:
    """Parse sizes such as ``"qwen3:4b"``, ``"kev-0.8b"`` or ``"Qwen3-0.6B"``."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*([bm])(?![a-z])", name.lower())
    if not match:
        return None
    value = float(match.group(1))
    return int(value * (1e9 if match.group(2) == "b" else 1e6))


def model_parameter_count(model_id: str, *, revision: str | None = None) -> ParamInfo:
    """Number of parameters of a Hugging Face model (local path or hub id).

    Tries, in order: safetensors headers of a local or cached copy (no download), the
    hub's safetensors metadata (small range requests only), a meta-device instantiation
    from a cached config, Simchology's table of default models, and finally the size in
    the model's name. Never downloads the weights.
    """
    path = Path(model_id)
    if path.is_dir():
        info = _count_local(path)
        if info is not None:
            return info
    snapshot = _cached_snapshot(model_id, revision)
    if snapshot is not None:
        info = _count_local(snapshot)
        if info is not None:
            return info
    for finder in (_hub_metadata, _from_config):
        info = finder(model_id, revision)
        if info is not None:
            return info
    if model_id in KNOWN_PARAM_COUNTS:
        n = KNOWN_PARAM_COUNTS[model_id]
        return ParamInfo(n, n * 4, "known")
    n = parse_size_from_name(model_id)
    if n is not None:
        return ParamInfo(n, None, "name")
    return ParamInfo(None, None, "unknown")


def cached_revision(model_id: str, revision: str | None = None) -> str | None:
    """Commit hash of the locally cached copy of a hub model (None for local folders or if not cached)."""
    if Path(model_id).is_dir():
        return None
    snapshot = _cached_snapshot(model_id, revision)
    return snapshot.name if snapshot is not None and snapshot.parent.name == "snapshots" else None


def local_parameter_count(model_id: str, *, revision: str | None = None) -> int | None:
    """Exact number of parameters from local files only (no estimates, no network).

    Reads the safetensors headers of a local folder or cached snapshot; for models stored
    as ``pytorch_model.bin`` only, instantiates the cached config on the meta device.
    """
    path = Path(model_id)
    folder = path if path.is_dir() else _cached_snapshot(model_id, revision)
    info = _count_local(folder) if folder is not None else None
    if info is None:
        info = _from_config(model_id, revision)
    return info.total if info is not None else None


def model_shape(model_id: str, *, revision: str | None = None) -> ModelShape:
    """Hidden size, depth and heads from a model's config (cached or tiny download)."""
    if not is_available("transformers"):
        return ModelShape()
    try:
        from transformers import AutoConfig

        offline = os.environ.get("HF_HUB_OFFLINE") == "1"
        config = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=offline)
    except Exception:
        return ModelShape()
    cfg = getattr(config, "text_config", None) or config

    def first(*names: str) -> int | None:
        for n in names:
            value = getattr(cfg, n, None)
            if isinstance(value, int):
                return value
        return None

    return ModelShape(
        hidden_size=first("hidden_size", "dim", "d_model", "n_embd"),
        n_layers=first("num_hidden_layers", "n_layers", "num_layers", "n_layer"),
        n_heads=first("num_attention_heads", "n_heads", "num_heads", "n_head"),
        max_positions=first("max_position_embeddings", "n_positions", "max_seq_len"),
    )
