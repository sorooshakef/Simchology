"""Input validation with messages written for users, not developers."""

from __future__ import annotations

import difflib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .._utils.warn import warn as _warn
from ..exceptions import ConfigurationError, SimchologyDataWarning


def did_you_mean(name: str, options: Sequence[str]) -> str:
    matches = difflib.get_close_matches(str(name), [str(o) for o in options], n=3, cutoff=0.5)
    return f" Did you mean {', '.join(repr(m) for m in matches)}?" if matches else ""


def validate_columns(df: pd.DataFrame, columns: Sequence[str], *, what: str = "data") -> None:
    if not isinstance(df, pd.DataFrame):
        raise ConfigurationError(f"{what} must be a pandas DataFrame, got {type(df).__name__}.")
    for col in columns:
        if col not in df.columns:
            raise ConfigurationError(
                f"Column {col!r} not found in {what}.{did_you_mean(col, list(df.columns))} "
                f"Available columns: {', '.join(map(str, df.columns))}."
            )


def validate_pair_columns(columns: Sequence[str]) -> tuple[str, str]:
    if isinstance(columns, str) or len(columns) != 2:
        raise ConfigurationError(
            "columns must name exactly two columns: (response, reference), "
            "e.g. columns=('text', 'info')."
        )
    if columns[0] == columns[1]:
        raise ConfigurationError("The two columns in `columns` must be different.")
    return str(columns[0]), str(columns[1])


def coerce_text(
    df: pd.DataFrame, columns: tuple[str, str], *, missing: str = "nan", what: str = "data",
    warn: bool = True,
) -> tuple[list[str], list[str], np.ndarray]:
    """Return the two text columns as lists of str plus a mask of usable rows.

    ``missing="nan"`` (default) marks rows with a missing text so they receive a missing
    prediction; ``missing="empty"`` treats missing texts as empty strings.
    """
    if missing not in ("nan", "empty"):
        raise ConfigurationError("missing must be 'nan' or 'empty'")
    a = df[columns[0]]
    b = df[columns[1]]
    mask = ~(a.isna().to_numpy() | b.isna().to_numpy())
    n_missing = int((~mask).sum())
    if n_missing:
        if missing == "empty":
            mask = np.ones(len(df), dtype=bool)
            if warn:
                _warn(f"{n_missing} row(s) of {what} have a missing text; they are "
                              "treated as empty strings.", SimchologyDataWarning)
        elif warn:
            _warn(f"{n_missing} row(s) of {what} have a missing text in "
                          f"{columns[0]!r} or {columns[1]!r}; their prediction will be missing.",
                          SimchologyDataWarning)
    left = ["" if pd.isna(x) else str(x) for x in a.tolist()]
    right = ["" if pd.isna(x) else str(x) for x in b.tolist()]
    return left, right, mask
