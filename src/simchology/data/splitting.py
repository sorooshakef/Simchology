"""Split coded data into training, development and test sets without leakage."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .._utils.logging import note
from .._utils.seeds import derive_seed
from ..exceptions import ConfigurationError
from .labels import LabelInfo, detect_label_type, to_numeric_labels
from .validation import validate_columns


@dataclass
class Splits:
    train: pd.DataFrame
    dev: pd.DataFrame | None
    test: pd.DataFrame
    split_by: str | None
    seed: int
    summary: pd.DataFrame

    def __repr__(self) -> str:
        return f"Splits(split_by={self.split_by!r})\n{self.summary.to_string(index=False)}"


def _stratum_labels(values: np.ndarray, info: LabelInfo) -> pd.Series:
    s = pd.Series(values)
    if info.kind == "binary":
        return s.fillna(-1).astype(int).astype(str)
    binned = pd.qcut(s.rank(method="first"), q=min(5, max(1, s.nunique())), labels=False, duplicates="drop")
    return binned.fillna(-1).astype(int).astype(str)


def resolve_split_by(df: pd.DataFrame, response_col: str, split_by: str | None) -> str | None:
    if split_by in (None, "row", "none"):
        return None
    if split_by == "auto":
        return response_col if df[response_col].duplicated().any() else None
    validate_columns(df, [split_by], what="the coded data")
    return split_by


def _split_once(index, strata, groups, size, seed):
    from sklearn.model_selection import (
        GroupShuffleSplit,
        StratifiedGroupKFold,
        train_test_split,
    )

    index = np.asarray(index)
    if size <= 0 or len(index) == 0:
        return index, index[:0]
    if groups is None:
        for stratify in (strata, strata.str.split("|").str[0], None):
            try:
                rest, held = train_test_split(
                    index, test_size=size, random_state=seed,
                    stratify=None if stratify is None else stratify.to_numpy(),
                )
                return np.asarray(rest), np.asarray(held)
            except ValueError:
                continue
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    n_splits = max(2, int(round(1 / size)))
    if n_groups < 2:
        raise ConfigurationError("Cannot make a grouped split with fewer than two groups.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if n_groups >= n_splits:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            rest_pos, held_pos = next(splitter.split(index, strata.to_numpy(), groups))
        else:
            splitter = GroupShuffleSplit(n_splits=1, test_size=size, random_state=seed)
            rest_pos, held_pos = next(splitter.split(index, groups=groups))
    return index[rest_pos], index[held_pos]


def split_coded(
    coded: pd.DataFrame,
    *,
    label_col: str,
    columns: tuple[str, str] = ("text", "info"),
    group_col: str | None = None,
    split_by: str | None = "auto",
    test_size: float = 0.2,
    dev_size: float = 0.0,
    seed: int = 42,
    positive_label: object | None = None,
    verbose: bool = True,
) -> Splits:
    """Split coded pairs into train/dev/test sets.

    The split is stratified by label (and ``group_col`` if given). With
    ``split_by="auto"`` (default), rows that share the same response text are kept in the
    same split, so the test set measures performance on *new* responses. Pass
    ``split_by="row"`` to split individual rows (as in many earlier analyses) or the name
    of a column (e.g. a participant or chain id) to keep its values together.

    ``test_size`` and ``dev_size`` are fractions of all coded rows.
    """
    validate_columns(coded, [*columns, label_col] + ([group_col] if group_col else []),
                     what="the coded data")
    if not 0 <= test_size < 1 or not 0 <= dev_size < 1 or test_size + dev_size >= 1:
        raise ConfigurationError("test_size and dev_size must be fractions that sum to less than 1.")
    df = coded[coded[label_col].notna()]
    values = to_numeric_labels(df[label_col], positive_label)
    info = detect_label_type(df[label_col], positive_label=positive_label)
    strata = _stratum_labels(values, info)
    if group_col:
        strata = strata + "|" + df[group_col].astype(str).to_numpy()
    strata.index = range(len(df))
    key = resolve_split_by(df, columns[0], split_by)
    if split_by == "auto" and key is not None:
        note(
            f"{len(df):,} coded rows contain only {df[columns[0]].nunique():,} distinct responses; "
            "splitting by response text so that no response appears in both the training and "
            "the test data (use split_by='row' to split rows instead).",
            verbose,
        )
    groups = None if key is None else df[key].astype(str).to_numpy()
    positions = np.arange(len(df))
    rest, test = _split_once(positions, strata, groups, test_size, seed)
    dev = positions[:0]
    if dev_size > 0:
        rel = dev_size / (1 - test_size)
        sub_groups = None if groups is None else groups[rest]
        sub_strata = strata.iloc[rest].reset_index(drop=True)
        rest_local, dev_local = _split_once(np.arange(len(rest)), sub_strata, sub_groups, rel,
                                            derive_seed(seed, 1))
        rest, dev = rest[rest_local], rest[dev_local]
    train_df = df.iloc[np.sort(rest)]
    dev_df = df.iloc[np.sort(dev)] if dev_size > 0 else None
    test_df = df.iloc[np.sort(test)]
    splits = Splits(train_df, dev_df, test_df, key, seed,
                    _summary(train_df, dev_df, test_df, columns[0], label_col, positive_label))
    if key is not None:
        assert_no_leakage(splits, key)
    return splits


def _summary(train, dev, test, response_col, label_col, positive_label) -> pd.DataFrame:
    rows = []
    for name, d in (("train", train), ("dev", dev), ("test", test)):
        if d is None:
            continue
        y = to_numeric_labels(d[label_col], positive_label) if len(d) else np.array([])
        rows.append({
            "split": name,
            "rows": len(d),
            "distinct_responses": int(d[response_col].nunique()),
            "mean_label": float(np.nanmean(y)) if len(y) else float("nan"),
        })
    return pd.DataFrame(rows)


def assert_no_leakage(splits: Splits, key: str) -> None:
    """Raise if any value of ``key`` occurs in more than one split."""
    seen: dict[str, set] = {}
    for name, d in (("train", splits.train), ("dev", splits.dev), ("test", splits.test)):
        if d is not None and len(d):
            seen[name] = set(d[key].astype(str))
    names = list(seen)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = seen[a] & seen[b]
            if overlap:
                raise AssertionError(f"{len(overlap)} value(s) of {key!r} occur in both {a} and {b}.")
