"""JSON schemas for structured LLM output and robust parsing of the answers.

Works with responses of the Ollama and OpenAI clients: token log-probabilities may be
pydantic objects or plain dicts with ``token``, ``logprob`` and ``top_logprobs``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

YES = ("yes", "ja", "true", "y")
NO = ("no", "nein", "false", "n")


def binary_schema() -> dict:
    return {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
        "required": ["answer"],
        "additionalProperties": False,
    }


def rating_schema(lo: int, hi: int) -> dict:
    return {
        "type": "object",
        "properties": {"rating": {"type": "integer", "minimum": lo, "maximum": hi}},
        "required": ["rating"],
        "additionalProperties": False,
    }


def answer_json(kind: str, value: Any) -> str:
    """The JSON an assistant would give (used for few-shot examples)."""
    if kind == "rating":
        return json.dumps({"rating": int(value)})
    return json.dumps({"answer": "yes" if value else "no"})


def _load_json(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*?\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def parse_binary(text: str) -> float | None:
    """1.0 for yes, 0.0 for no, None if the answer cannot be read."""
    data = _load_json(text)
    value = None
    if isinstance(data, Mapping):
        value = data.get("answer", data.get("similar", data.get("contained")))
        if isinstance(value, bool):
            return float(value)
    elif isinstance(data, bool):
        return float(data)
    candidate = str(value if value is not None else text or "").strip().strip('"\'.!').lower()
    if candidate in YES:
        return 1.0
    if candidate in NO:
        return 0.0
    words = re.findall(r"[a-zäöü]+", str(text or "").lower())
    hits = {w for w in words if w in YES + NO}
    if len(hits) == 1:
        return 1.0 if hits.pop() in YES else 0.0
    return None


def parse_rating(text: str, lo: int, hi: int) -> float | None:
    """The rating as a number in [lo, hi], or None."""
    data = _load_json(text)
    value = None
    if isinstance(data, Mapping):
        value = data.get("rating", data.get("score"))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        value = data
    if value is None:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", str(text or ""))
        value = numbers[0] if len(numbers) == 1 else None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if lo <= value <= hi:
        return value
    return None


def rating_to_unit(value: float, lo: int, hi: int) -> float:
    return (float(value) - lo) / (hi - lo)


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    try:
        return obj[key]  # ollama's SubscriptableBaseModel
    except (TypeError, KeyError, IndexError):
        return getattr(obj, key, None)


def _clean_token(token: Any) -> str:
    return str(token or "").strip().strip("\"'{}:,").strip().lower()


def _positions(logprobs: Any) -> list:
    """The per-token entries, whatever the client's container looks like."""
    if logprobs is None:
        return []
    content = _get(logprobs, "content") if not isinstance(logprobs, Sequence) else None
    if content is not None:
        logprobs = content
    return list(logprobs) if isinstance(logprobs, Sequence) else []


def prob_yes_from_logprobs(logprobs: Any) -> float | None:
    """P(yes) / (P(yes) + P(no)) at the first token position that answers yes or no."""
    for pos in _positions(logprobs):
        tok = _clean_token(_get(pos, "token"))
        if tok not in YES + NO or tok in ("y", "n"):
            continue
        candidates = list(_get(pos, "top_logprobs") or []) or [pos]
        p_yes = p_no = 0.0
        for c in candidates:
            word = _clean_token(_get(c, "token"))
            lp = _get(c, "logprob")
            if lp is None:
                continue
            if word in YES:
                p_yes += math.exp(lp)
            elif word in NO:
                p_no += math.exp(lp)
        if p_yes + p_no == 0:
            return None
        return p_yes / (p_yes + p_no)
    return None


def expected_rating_from_logprobs(logprobs: Any, lo: int, hi: int) -> float | None:
    """Σ k·p(k) over the rating digits at the first position that holds a rating."""
    for pos in _positions(logprobs):
        tok = _clean_token(_get(pos, "token"))
        if not tok.isdigit() or not lo <= int(tok) <= hi:
            continue
        candidates = list(_get(pos, "top_logprobs") or []) or [pos]
        weights: dict[int, float] = {}
        for c in candidates:
            word = _clean_token(_get(c, "token"))
            lp = _get(c, "logprob")
            if word.isdigit() and lo <= int(word) <= hi and lp is not None:
                weights[int(word)] = weights.get(int(word), 0.0) + math.exp(lp)
        total = sum(weights.values())
        if total == 0:
            return None
        return sum(k * w for k, w in weights.items()) / total
    return None
