"""Tokenised pair datasets for supervised pair scoring."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class PairDataset:
    """A map-style dataset of pre-tokenised examples (padding happens per batch).

    Plain Python, so that importing it does not import torch; torch's DataLoader and
    the transformers Trainer accept any object with ``__len__`` and ``__getitem__``.
    """

    def __init__(self, features: list[dict[str, Any]], labels: Sequence | None = None) -> None:
        if labels is not None and len(labels) != len(features):
            raise ValueError("features and labels must have the same length")
        self.features = features
        self.labels = None if labels is None else list(labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = dict(self.features[i])
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


def batch_to_features(encoding) -> list[dict[str, Any]]:
    """Split a tokenizer's batch output (dict of lists) into one dict per example."""
    keys = [k for k in encoding.keys() if k in ("input_ids", "attention_mask", "token_type_ids")]
    n = len(encoding["input_ids"])
    return [{k: encoding[k][i] for k in keys} for i in range(n)]


def format_template(template: str, response: str, reference: str) -> str:
    return template.format(response=response, reference=reference)


def fit_template(tokenizer, template: str, response: str, reference: str, max_length: int | None) -> str:
    """Fill ``template``, shortening the *response* so that the whole prompt fits ``max_length``.

    Decoder classifiers read the last token, so the end of the prompt (the reference and
    the answer cue) must never be truncated away.
    """
    text = format_template(template, response, reference)
    if max_length is None:
        return text
    n = len(tokenizer(text, add_special_tokens=True)["input_ids"])
    if n <= max_length:
        return text
    base = len(tokenizer(format_template(template, "", reference), add_special_tokens=True)["input_ids"])
    budget = max_length - base
    if budget <= 0:
        return format_template(template, "", reference)
    ids = tokenizer(response, add_special_tokens=False)["input_ids"][:budget]
    shortened = tokenizer.decode(ids, skip_special_tokens=True)
    # decoding/re-encoding can change the length by a token or two; trim until it fits
    while ids and len(tokenizer(format_template(template, shortened, reference))["input_ids"]) > max_length:
        ids = ids[:-1]
        shortened = tokenizer.decode(ids, skip_special_tokens=True)
    return format_template(template, shortened, reference)


def pair_lengths(tokenizer, left: Sequence[str], right: Sequence[str] | None = None) -> np.ndarray:
    """Token counts of single texts (``right=None``) or text pairs, without truncation."""
    if right is None:
        enc = tokenizer(list(left), add_special_tokens=True, truncation=False)
    else:
        enc = tokenizer(list(left), list(right), add_special_tokens=True, truncation=False)
    return np.array([len(ids) for ids in enc["input_ids"]], dtype=int)


def model_max_positions(config, tokenizer) -> int | None:
    cfg = getattr(config, "text_config", None) or config
    for name in ("max_position_embeddings", "n_positions", "max_seq_len"):
        value = getattr(cfg, name, None)
        if isinstance(value, int) and value > 0:
            return value
    tml = getattr(tokenizer, "model_max_length", None)
    if isinstance(tml, int) and 0 < tml < 1_000_000:
        return tml
    return None


def resolve_max_length(
    lengths: np.ndarray, *, model_max: int | None, cap: int, requested: int | str = "auto"
) -> int:
    """``requested`` if an int, else min(model maximum, ``cap``, 95th percentile of ``lengths``)."""
    if requested != "auto":
        value = int(requested)
        return min(value, model_max) if model_max else value
    p95 = int(np.ceil(np.percentile(lengths, 95))) if len(lengths) else cap
    limit = min(cap, model_max) if model_max else cap
    return int(max(16, min(limit, p95)))
