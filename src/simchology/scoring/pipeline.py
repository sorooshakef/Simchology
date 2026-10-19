"""Score row-aligned text pairs: deduplicate, segment, score unique pairs, aggregate.

This generalises the original coding scripts' approach, where each response was split
into sentences, every sentence was compared with the reference, and the best
(maximum) similarity counted for the row.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .._utils.logging import progress
from ..exceptions import ConfigurationError
from ..text.normalize import light_normalize

AGGREGATES = ("max", "mean", "max_mean")

SegmentFn = Callable[[Sequence[str]], list[list[str]]]


class ScoringPipeline:
    """Compute one similarity per row with optional sentence segmentation.

    Parameters
    ----------
    method:
        A :class:`~simchology.methods.base.SimilarityMethod`.
    segment_left, segment_right:
        Functions splitting texts into segments (e.g. ``Segmenter.split``) or None.
    aggregate:
        How segment scores are combined into the row score: ``"max"`` (best-matching
        sentence, default), ``"mean"`` (average over all sentence pairs) or
        ``"max_mean"`` (for each reference sentence the best response sentence, then the
        mean over reference sentences: how well the whole reference is covered).
    light_normalize_inputs:
        Apply Unicode/whitespace normalisation before scoring (for neural methods).
    """

    def __init__(
        self,
        method,
        *,
        segment_left: SegmentFn | None = None,
        segment_right: SegmentFn | None = None,
        aggregate: str = "max",
        light_normalize_inputs: bool = False,
        chunk_size: int = 50_000,
        show_progress: bool = False,
    ) -> None:
        if aggregate not in AGGREGATES:
            raise ConfigurationError(f"aggregate must be one of {AGGREGATES}; got {aggregate!r}")
        self.method = method
        self.segment_left = segment_left
        self.segment_right = segment_right
        self.aggregate = aggregate
        self.light = light_normalize_inputs
        self.chunk_size = chunk_size
        self.show_progress = show_progress

    def _segments(self, texts: list[str], fn: SegmentFn | None) -> list[list[str]]:
        if self.light:
            texts = [light_normalize(t) for t in texts]
        if fn is None:
            return [[t] for t in texts]
        return [s if s else [t] for s, t in zip(fn(texts), texts)]

    def units(self, left: Sequence[str], right: Sequence[str]) -> list[str]:
        """All unique segments of both columns (the corpus for vocabulary-based methods)."""
        segs = self._segments(list(dict.fromkeys(left)), self.segment_left)
        segs += self._segments(list(dict.fromkeys(right)), self.segment_right)
        return list(dict.fromkeys(s for group in segs for s in group))

    def score(
        self, left: Sequence[str], right: Sequence[str], *, return_details: bool = False
    ):
        """Return one score per row (and, with ``return_details``, the best response segment)."""
        left, right = list(left), list(right)
        if len(left) != len(right):
            raise ValueError("left and right must have the same length")
        n = len(left)
        if n == 0:
            empty = np.zeros(0)
            return (empty, []) if return_details else empty

        uniq_l = list(dict.fromkeys(left))
        uniq_r = list(dict.fromkeys(right))
        pos_l = {t: i for i, t in enumerate(uniq_l)}
        pos_r = {t: i for i, t in enumerate(uniq_r)}
        seg_l = self._segments(uniq_l, self.segment_left)
        seg_r = self._segments(uniq_r, self.segment_right)

        # unique row pairs (many rows share both texts)
        row_keys = [(pos_l[a], pos_r[b]) for a, b in zip(left, right)]
        uniq_rows = list(dict.fromkeys(row_keys))

        # expand row pairs into segment pairs
        pair_left: list[str] = []
        pair_right: list[str] = []
        offsets = np.zeros(len(uniq_rows) + 1, dtype=np.int64)
        for k, (i, j) in enumerate(uniq_rows):
            for s in seg_l[i]:
                for t in seg_r[j]:
                    pair_left.append(s)
                    pair_right.append(t)
            offsets[k + 1] = len(pair_left)

        # score each unique segment pair once
        seg_keys = list(zip(pair_left, pair_right))
        uniq_seg = list(dict.fromkeys(seg_keys))
        seg_scores = np.empty(len(uniq_seg))
        chunks = range(0, len(uniq_seg), self.chunk_size)
        for start in progress(chunks, total=len(chunks), desc=f"scoring ({self.method.name})",
                              enabled=self.show_progress and len(chunks) > 1):
            block = uniq_seg[start:start + self.chunk_size]
            seg_scores[start:start + len(block)] = self.method.score_pairs(
                [a for a, _ in block], [b for _, b in block]
            )
        lookup = dict(zip(uniq_seg, seg_scores))
        expanded = np.fromiter((lookup[k] for k in seg_keys), dtype=float, count=len(seg_keys))

        row_scores = np.empty(len(uniq_rows))
        best_seg: list[str | None] = [None] * len(uniq_rows)
        for k, (i, j) in enumerate(uniq_rows):
            block = expanded[offsets[k]:offsets[k + 1]]
            n_l, n_r = len(seg_l[i]), len(seg_r[j])
            mat = block.reshape(n_l, n_r)
            if np.all(np.isnan(mat)):
                row_scores[k] = np.nan
                continue
            if self.aggregate == "max":
                row_scores[k] = np.nanmax(mat)
            elif self.aggregate == "mean":
                row_scores[k] = np.nanmean(mat)
            else:
                row_scores[k] = float(np.nanmean(np.nanmax(mat, axis=0)))
            if return_details:
                best_seg[k] = seg_l[i][int(np.nanargmax(np.nanmax(mat, axis=1)))]

        index = {key: k for k, key in enumerate(uniq_rows)}
        rows = np.fromiter((index[key] for key in row_keys), dtype=np.int64, count=n)
        out = row_scores[rows]
        if return_details:
            return out, [best_seg[r] for r in rows]
        return out
