"""Build pairs of texts that are known (or very likely) to be dissimilar.

Raw similarity values are not comparable across methods: contextual embeddings, for
example, give even unrelated texts cosine similarities well above 0. Simchology
therefore interprets scores relative to a *baseline*: the distribution of scores of
dissimilar pairs from the same data. The pairs come from

* ``coded_negatives`` - pairs that human coders judged dissimilar (hard negatives);
* ``cross_group``     - each response re-paired with references from *other* groups,
                         e.g. another topic or stimulus (safe, easier negatives);
* ``random``          - random re-pairings, when no group information exists (a few
                         such pairs may happen to be similar).

Re-pairings within the same group that were never coded are deliberately not used: a
response may well contain other references of its own topic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._utils.warn import warn
from ..config import GROUP, LABEL, LEFT, RIGHT
from ..exceptions import ConfigurationError, SimchologyDataWarning

STRATEGIES = ("auto", "coded_negatives", "cross_group", "random", "coded_negatives+cross_group")


def _positive_keys(pairs: pd.DataFrame, negative_threshold: float | None) -> set[tuple[str, str]]:
    if LABEL not in pairs:
        return set(zip(pairs[LEFT], pairs[RIGHT]))  # unlabelled: any observed pairing may be similar
    lab = pairs[LABEL].to_numpy(dtype=float)
    thr = 0.0 if negative_threshold is None else negative_threshold
    pos = ~(lab <= thr)
    return set(zip(pairs.loc[pos, LEFT], pairs.loc[pos, RIGHT]))


def _sample_repairs(pairs, n, rng, *, cross_group, exclude):
    lefts = pairs[LEFT].drop_duplicates().to_numpy()
    if cross_group:
        groups_of_left = pairs.groupby(LEFT)[GROUP].agg(lambda s: set(s.astype(str)))
        refs = pairs[[RIGHT, GROUP]].drop_duplicates(RIGHT)
        ref_text = refs[RIGHT].to_numpy()
        ref_group = refs[GROUP].astype(str).to_numpy()
    else:
        ref_text = pairs[RIGHT].drop_duplicates().to_numpy()
        ref_group = None
    if len(lefts) == 0 or len(ref_text) == 0:
        return pd.DataFrame(columns=[LEFT, RIGHT])
    out, seen = [], set()
    attempts = 0
    max_attempts = 50 * n + 1000
    while len(out) < n and attempts < max_attempts:
        batch = max(64, 2 * (n - len(out)))
        li = rng.integers(0, len(lefts), batch)
        ri = rng.integers(0, len(ref_text), batch)
        attempts += batch
        for a, b in zip(li, ri):
            left, right = lefts[a], ref_text[b]
            key = (left, right)
            if key in seen or key in exclude or left == right:
                continue
            if cross_group and ref_group[b] in groups_of_left[left]:
                continue
            seen.add(key)
            out.append(key)
            if len(out) >= n:
                break
    return pd.DataFrame(out, columns=[LEFT, RIGHT])


def build_baseline_pairs(
    pairs: pd.DataFrame,
    *,
    strategy: str = "auto",
    n_pairs: int = 2000,
    min_pairs: int = 100,
    negative_threshold: float | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Return dissimilar pairs with columns ``left``, ``right``, ``source`` (+ ``group_left``).

    ``pairs`` is a standardised frame with ``left``, ``right`` and optionally ``label``
    and ``group``. With ``strategy="auto"``:

    =============  =============  ==============================================
    coded labels   group column   baseline pairs
    =============  =============  ==============================================
    yes            yes            coded negatives + cross-group re-pairings
    yes            no             coded negatives (+ random re-pairings if too few)
    no             yes            cross-group re-pairings
    no             no             random re-pairings (with a warning)
    =============  =============  ==============================================

    When both coded negatives and cross-group pairs are used, at most half of the
    ``n_pairs`` are coded negatives, so the baseline mixes hard and easy negatives.
    """
    if strategy not in STRATEGIES:
        raise ConfigurationError(f"baseline must be one of {STRATEGIES}; got {strategy!r}")
    rng = np.random.default_rng(seed)
    has_labels = LABEL in pairs and pairs[LABEL].notna().any()
    has_groups = GROUP in pairs and pairs[GROUP].astype(str).nunique() > 1
    if strategy == "auto":
        if has_labels and has_groups:
            strategy = "coded_negatives+cross_group"
        elif has_labels:
            strategy = "coded_negatives"
        elif has_groups:
            strategy = "cross_group"
        else:
            strategy = "random"
    if "coded_negatives" in strategy and not has_labels:
        raise ConfigurationError("baseline='coded_negatives' needs coded data with labels.")
    if "cross_group" in strategy and not has_groups:
        raise ConfigurationError("baseline='cross_group' needs group_col with at least two groups.")

    exclude = _positive_keys(pairs, negative_threshold)
    parts = []
    if "coded_negatives" in strategy:
        thr = 0.0 if negative_threshold is None else negative_threshold
        neg = pairs[pairs[LABEL].to_numpy(dtype=float) <= thr][[LEFT, RIGHT]].drop_duplicates()
        cap = n_pairs // 2 if "cross_group" in strategy else n_pairs
        if len(neg) > cap:
            neg = neg.sample(cap, random_state=seed)
        parts.append(neg.assign(source="coded_negative"))
    remaining = n_pairs - sum(len(p) for p in parts)
    if "cross_group" in strategy and remaining > 0:
        parts.append(_sample_repairs(pairs, remaining, rng, cross_group=True, exclude=exclude)
                     .assign(source="cross_group"))
    if strategy == "random" or (strategy == "coded_negatives" and sum(len(p) for p in parts) < min_pairs):
        need = n_pairs - sum(len(p) for p in parts)
        if need > 0:
            parts.append(_sample_repairs(pairs, need, rng, cross_group=False, exclude=exclude)
                         .assign(source="random"))
            warn(
                "The baseline includes random re-pairings of responses and references. Some "
                "of them may happen to be similar, which makes thresholds stricter. Pass "
                "group_col (e.g. the topic or stimulus) so that responses are re-paired only "
                "with references from other groups.", SimchologyDataWarning,
            )
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[LEFT, RIGHT, "source"])
    if GROUP in pairs:
        first_group = pairs.drop_duplicates(LEFT).set_index(LEFT)[GROUP]
        out["group_left"] = out[LEFT].map(first_group)
    out.attrs["strategy"] = strategy
    return out
