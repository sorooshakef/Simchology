"""Explanations: which parts of two texts made them similar."""

from __future__ import annotations

import html
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Explanation:
    """Why a method assigned a similarity score to a (response, reference) pair.

    Attributes
    ----------
    alignments:
        One row per token/word with the token it was matched to and the match strength.
        Columns: ``side`` ("response"/"reference"), ``token``, ``matched_token``,
        ``similarity`` and optionally ``weight``.
    shared, only_response, only_reference:
        For set-based methods: units in both texts / only in one.
    winning_segment:
        For segmented responses: the sentence that produced the (maximum) score.
    """

    method: str
    response: str
    reference: str
    score: float
    components: dict[str, float] = field(default_factory=dict)
    alignments: pd.DataFrame | None = None
    shared: list[str] | None = None
    only_response: list[str] | None = None
    only_reference: list[str] | None = None
    matrix: np.ndarray | None = field(default=None, repr=False)
    response_tokens: list[str] | None = None
    reference_tokens: list[str] | None = None
    winning_segment: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        if self.alignments is not None:
            return self.alignments
        rows = [{"side": "both", "token": t} for t in self.shared or []]
        rows += [{"side": "response only", "token": t} for t in self.only_response or []]
        rows += [{"side": "reference only", "token": t} for t in self.only_reference or []]
        return pd.DataFrame(rows, columns=["side", "token"])

    def __str__(self) -> str:
        lines = [f"{self.method}: similarity {self.score:.4f}"]
        if self.components:
            lines.append("  " + ", ".join(f"{k}={v:.4f}" for k, v in self.components.items()))
        if self.winning_segment is not None and self.winning_segment != self.response:
            lines.append(f"  best-matching sentence: {self.winning_segment!r}")
        if self.shared is not None:
            lines.append(f"  shared: {', '.join(self.shared) or '-'}")
            lines.append(f"  only in response: {', '.join(self.only_response or []) or '-'}")
            lines.append(f"  only in reference: {', '.join(self.only_reference or []) or '-'}")
        if self.alignments is not None and len(self.alignments):
            lines.append("  alignments:")
            show = self.alignments.head(20)
            for r in show.itertuples(index=False):
                lines.append(f"    [{r.side}] {r.token!r} -> {r.matched_token!r} ({r.similarity:.3f})")
            if len(self.alignments) > len(show):
                lines.append(f"    ... {len(self.alignments) - len(show)} more (see .to_frame())")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)

    __repr__ = __str__

    def _repr_html_(self) -> str:
        def shade(tokens, sims):
            parts = []
            for t, s in zip(tokens, sims):
                alpha = 0.0 if s is None or np.isnan(s) else float(np.clip(s, 0, 1))
                parts.append(
                    f'<span style="background: rgba(46,160,67,{alpha:.2f}); padding:1px 2px; '
                    f'border-radius:3px" title="{alpha:.3f}">{html.escape(str(t))}</span>'
                )
            return " ".join(parts)

        head = f"<b>{html.escape(self.method)}</b>: similarity {self.score:.4f}"
        if self.alignments is not None and len(self.alignments):
            a = self.alignments
            resp = a[a["side"] == "response"]
            ref = a[a["side"] == "reference"]
            return (
                f"<div>{head}<br><i>response</i>: {shade(resp['token'], resp['similarity'])}"
                f"<br><i>reference</i>: {shade(ref['token'], ref['similarity'])}</div>"
            )
        return f"<div>{head}<pre>{html.escape(str(self))}</pre></div>"
