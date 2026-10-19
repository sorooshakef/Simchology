"""Ending-perturbation data augmentation for supervised pair scoring.

Ported from the "Fine-Tuning BERT-perturbed" notebook. Every row is duplicated with a
perturbed copy of its *response* text, split 50/50 between:

* ``no_period``   - the final sentence terminator (. ! ? ...) is removed;
* ``cut_midword`` - terminator removed AND the final word severed at its midpoint.

The reference text and the label are never touched. Training on such twins teaches a
cross-encoder that a severed ending carries no information about similarity, which
greatly reduces prediction flips on truncated responses.

Always split *before* augmenting, so that a row and its twin end up in the same split.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

VARIANT_COL = "_simchology_variant"
ORIG_ID_COL = "_simchology_orig_id"

TERMINAL_PUNCT_RE = re.compile(r"[.!?…]+$")


def strip_terminal_punct(text: str) -> str:
    """Drop trailing whitespace and any run of sentence-final punctuation."""
    return TERMINAL_PUNCT_RE.sub("", text.rstrip()).rstrip()


def perturb_no_period(text: str) -> str:
    """Variant A: the final sentence loses its terminator."""
    return strip_terminal_punct(text)


def perturb_cut_midword(text: str) -> str:
    """Variant B: terminator removed AND the final word severed at its midpoint.

    The surviving fragment is the first ceil(len/2) characters, so "hier" -> "hi" and
    "Kaiserreichs" -> "Kaiser". Such fragments are usually not words, which is the
    point: the tokenizer re-segments them into rare sub-word pieces.
    """
    stripped = strip_terminal_punct(text)
    parts = stripped.split()
    if not parts:
        return stripped
    last = parts[-1]
    keep = (len(last) + 1) // 2
    if keep >= len(last):  # 1-character words cannot be halved
        keep = len(last) - 1
    if keep <= 0:  # drop the word entirely as a last resort
        return " ".join(parts[:-1]).rstrip(" ,;:").rstrip()
    return " ".join(parts[:-1] + [last[:keep]])


PERTURBATIONS: dict[str, Callable[[str], str]] = {
    "no_period": perturb_no_period,
    "cut_midword": perturb_cut_midword,
}


def lacks_terminal_punct(text: str) -> bool:
    return strip_terminal_punct(text) == text.rstrip()


def share_lacking_terminal_punct(texts: Sequence[str]) -> float:
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return 0.0
    return float(np.mean([lacks_terminal_punct(t) for t in texts]))


def assign_variants(
    texts: Sequence[str], seed: int, cut_share: float = 0.5, fallback_when_noop: bool = True
) -> np.ndarray:
    """Exactly balanced assignment of perturbation variants.

    A text without terminal punctuation would be unchanged by ``no_period``, producing a
    duplicate of the original row. Those rows are forced to ``cut_midword`` first, and the
    remaining ``cut_midword`` slots are drawn from the rest, so the overall split still
    lands on ``cut_share`` exactly.
    """
    n = len(texts)
    n_cut = int(round(n * cut_share))
    forced = np.array([lacks_terminal_punct(t) for t in texts], dtype=bool)

    kinds = np.where(forced, "cut_midword", "no_period").astype(object)
    if fallback_when_noop:
        remaining = n_cut - int(forced.sum())
        eligible = np.flatnonzero(~forced)
    else:
        kinds[:] = "no_period"
        remaining = n_cut
        eligible = np.arange(n)
    if remaining > 0 and len(eligible):
        chosen = np.random.RandomState(seed).choice(
            eligible, size=min(remaining, len(eligible)), replace=False
        )
        kinds[chosen] = "cut_midword"
    return kinds


def augment(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    seed: int = 42,
    cut_share: float = 0.5,
    fallback_when_noop: bool = True,
) -> pd.DataFrame:
    """Return ``df`` plus one perturbed copy of every row (only ``text_col`` is changed).

    Two helper columns are added: ``_simchology_variant`` ("original", "no_period" or
    "cut_midword") and ``_simchology_orig_id`` (the index of the original row). Each
    original row is immediately followed by its perturbed twin.
    """
    if not df.index.is_unique:
        df = df.reset_index(drop=True)
    original = df.copy()
    original[VARIANT_COL] = "original"
    original[ORIG_ID_COL] = original.index

    perturbed = df.copy()
    perturbed[ORIG_ID_COL] = perturbed.index
    texts = perturbed[text_col].fillna("").astype(str).tolist()
    kinds = assign_variants(texts, seed, cut_share, fallback_when_noop)
    perturbed[text_col] = [PERTURBATIONS[k](t) for t, k in zip(texts, kinds)]
    perturbed[VARIANT_COL] = kinds
    perturbed.index = [f"{i}_p" for i in perturbed.index]
    perturbed.index.name = df.index.name

    combined = pd.concat([original, perturbed])
    order = (combined[VARIANT_COL] != "original").astype(int)
    position = pd.Series(range(len(df)), index=df.index)
    combined = combined.assign(
        _pos=position.reindex(combined[ORIG_ID_COL]).to_numpy(), _order=order.to_numpy()
    )
    combined = combined.sort_values(["_pos", "_order"], kind="mergesort")
    return combined.drop(columns=["_pos", "_order"])


def check_augmentation_integrity(
    df: pd.DataFrame, *, text_col: str, ref_col: str, label_col: str
) -> int:
    """Assert that twins are paired, references/labels unchanged; return #no-op perturbations."""
    orig = df[df[VARIANT_COL] == "original"].set_index(ORIG_ID_COL)
    pert = df[df[VARIANT_COL] != "original"].set_index(ORIG_ID_COL)
    assert len(orig) == len(pert), "unpaired rows"
    assert set(orig.index) == set(pert.index), "pair mismatch"
    pert = pert.loc[orig.index]
    assert (orig[ref_col].to_numpy() == pert[ref_col].to_numpy()).all(), "reference was modified"
    labels_equal = (orig[label_col].to_numpy() == pert[label_col].to_numpy()) | (
        orig[label_col].isna().to_numpy() & pert[label_col].isna().to_numpy()
    )
    assert labels_equal.all(), "label drift"
    assert not df.index.duplicated().any(), "duplicate ids"
    o = orig[text_col].astype(str).str.rstrip().to_numpy()
    p = pert[text_col].astype(str).str.rstrip().to_numpy()
    return int((o == p).sum())
