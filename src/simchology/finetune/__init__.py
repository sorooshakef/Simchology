"""Fine-tuning helpers for supervised pair scoring (cross-encoders, LoRA classifiers).

Nothing here imports torch or transformers at module level.
"""

from .datasets import PairDataset, fit_template, pair_lengths, resolve_max_length

__all__ = ["PairDataset", "fit_template", "pair_lengths", "resolve_max_length"]
