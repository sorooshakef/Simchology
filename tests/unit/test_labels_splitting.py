import numpy as np
import pandas as pd
import pytest

import simchology as sim
from simchology.data.labels import binarize_labels, detect_label_type, to_numeric_labels
from simchology.data.splitting import assert_no_leakage
from simchology.exceptions import ConfigurationError


def test_detect_binary_and_gradient():
    assert detect_label_type(pd.Series([0, 1, 1, 0])).kind == "binary"
    assert detect_label_type(pd.Series([True, False])).kind == "binary"
    info = detect_label_type(pd.Series([1, 2, 3, 4, 5]))
    assert info.kind == "gradient" and info.min == 1 and info.max == 5


def test_string_labels_need_positive_label():
    with pytest.raises(ConfigurationError, match="positive_label"):
        to_numeric_labels(pd.Series(["yes", "no"]))
    assert list(to_numeric_labels(pd.Series(["yes", "no"]), positive_label="yes")) == [1.0, 0.0]


def test_binarize_gradient_requires_threshold():
    info = detect_label_type(pd.Series([0, 0.5, 1]))
    with pytest.raises(ConfigurationError, match="positive_threshold"):
        binarize_labels(np.array([0, 0.5, 1]), info, None)
    assert list(binarize_labels(np.array([0, 0.5, 1]), info, 0.5)) == [0, 1, 1]


def test_auto_split_groups_by_response(demo_de):
    splits = sim.split_coded(demo_de, label_col="label", group_col="topic", test_size=0.2,
                             dev_size=0.1, verbose=False)
    assert splits.split_by == "text"
    assert_no_leakage(splits, "text")
    n = len(demo_de)
    assert abs(len(splits.test) / n - 0.2) < 0.08
    assert abs(len(splits.dev) / n - 0.1) < 0.08
    assert len(splits.train) + len(splits.dev) + len(splits.test) == n
    # stratification keeps the label rate similar
    assert abs(splits.test["label"].mean() - demo_de["label"].mean()) < 0.1


def test_row_split_matches_sklearn_proportions(demo_de):
    splits = sim.split_coded(demo_de, label_col="label", split_by="row", test_size=0.2, verbose=False)
    assert splits.split_by is None
    assert len(splits.test) == 120
    # a row split leaks responses across splits - which is why "auto" groups
    assert set(splits.train["text"]) & set(splits.test["text"])


def test_split_by_column(demo_de):
    splits = sim.split_coded(demo_de, label_col="label", split_by="participant", verbose=False)
    assert_no_leakage(splits, "participant")


def test_split_is_reproducible(demo_de):
    a = sim.split_coded(demo_de, label_col="label", seed=1, verbose=False)
    b = sim.split_coded(demo_de, label_col="label", seed=1, verbose=False)
    assert a.test.index.equals(b.test.index)
