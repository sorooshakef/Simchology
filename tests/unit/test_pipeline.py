import numpy as np
import pytest

from simchology.methods.registry import make_method
from simchology.scoring.pipeline import ScoringPipeline
from simchology.text.preprocess import PreprocessConfig, Preprocessor
from simchology.text.segment import Segmenter


class CountingJaccard:
    """Wrap a method and count how many pairs it is asked to score."""

    def __init__(self):
        self.inner = make_method("jaccard", {"unit": "surface"})
        self.inner.bind(Preprocessor(PreprocessConfig.from_options("en", {"backend": "regex"}),
                                     verbose=False), self.inner.runtime)
        self.name = "jaccard"
        self.calls = 0

    def score_pairs(self, left, right):
        self.calls += len(left)
        return self.inner.score_pairs(left, right)


def test_max_over_sentences_and_dedup():
    m = CountingJaccard()
    seg = Segmenter("regex", "en", verbose=False)
    pipe = ScoringPipeline(m, segment_left=seg.split)
    text = "Giraffes are tall. Moai are statues."
    left = [text, text, text]
    right = ["giraffes tall", "moai statues", "giraffes tall"]
    s, best = pipe.score(left, right, return_details=True)
    assert np.allclose(s, [1.0, 1.0, 1.0])
    assert best[0] == "Giraffes are tall." and best[1] == "Moai are statues."
    assert m.calls == 4  # 2 unique references x 2 sentences


def test_aggregates():
    m = CountingJaccard()
    seg = Segmenter("regex", "en", verbose=False)
    text = "Giraffes are tall. Moai are statues."
    mean = ScoringPipeline(m, segment_left=seg.split, aggregate="mean").score([text], ["giraffes tall"])
    assert mean[0] == pytest.approx(0.5)
    cover = ScoringPipeline(m, segment_left=seg.split, segment_right=seg.split, aggregate="max_mean")
    assert cover.score([text], ["Giraffes are tall. Water flows."])[0] == pytest.approx(0.5)


def test_units_are_unique_segments():
    m = CountingJaccard()
    seg = Segmenter("regex", "en", verbose=False)
    pipe = ScoringPipeline(m, segment_left=seg.split)
    units = pipe.units(["Ein Satz. Noch einer.", "Ein Satz. Noch einer."], ["x y"])
    assert units == ["Ein Satz.", "Noch einer.", "x y"]


def test_empty_input():
    assert ScoringPipeline(CountingJaccard()).score([], []).shape == (0,)
