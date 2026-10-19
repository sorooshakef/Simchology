"""Optional regression test against the original Python reimplementation of coding.R.

Runs only when the private study data are available:

    SIMCHOLOGY_CODED_CSV=/path/data_coded_long.csv \
    SIMCHOLOGY_UNCODED_CSV=/path/data_uncoded_long.csv \
    SIMCHOLOGY_INFOS_CSV=/path/infos.csv \
    python -m pytest tests/regression -m "" -q

It needs spaCy with ``de_core_news_lg``. The notebook ("Reimplementing Coding
Approaches.ipynb") evaluated each method on all coded rows with thresholds derived from
the reference infos (variants a-d); this test reproduces that protocol through
Simchology's public API and compares F1 with the notebook's printed values.
"""

import os

import pandas as pd
import pytest

import simchology as sim
from simchology.text.spacy_models import spacy_model_installed

CODED = os.environ.get("SIMCHOLOGY_CODED_CSV")
UNCODED = os.environ.get("SIMCHOLOGY_UNCODED_CSV")
INFOS = os.environ.get("SIMCHOLOGY_INFOS_CSV")

pytestmark = pytest.mark.skipif(
    not (CODED and UNCODED and INFOS) or not spacy_model_installed("de_core_news_lg"),
    reason="private study data / de_core_news_lg not available",
)

# (method, simchology method params, threshold variant, F1 printed by the notebook)
CASES = [
    ("levenshtein", {"normalize": False, "unit": "surface"}, "reference_d", 0.7672),
    ("jaccard", {"unit": "lemma"}, "reference_d", 0.8901),
    ("tfidf", {"unit": "lemma"}, "reference_c", 0.9036),
]


@pytest.fixture(scope="module")
def frames():
    coded = pd.read_csv(CODED)
    uncoded = pd.read_csv(UNCODED)
    infos = pd.read_csv(INFOS, sep=";").melt(id_vars="topic", var_name="info.id", value_name="info")
    infos = infos.dropna(subset=["info"])
    for df in (coded, uncoded, infos):
        df["info"] = df["info"].astype(str).str.split().str.join(" ")
    coded["text"] = coded["text"].fillna("").astype(str)
    uncoded["text"] = uncoded["text"].fillna("").astype(str)
    return coded, uncoded, infos


@pytest.mark.parametrize("method,params,threshold,expected_f1", CASES)
def test_reference_threshold_f1_matches_notebook(frames, method, params, threshold, expected_f1):
    coded, uncoded, infos = frames
    assessor = sim.SimilarityAssessor(
        method, "binary", columns=("text", "info"), group_col="topic", threshold=threshold,
        language="de", segment="spacy", preprocess={"replacements": {"ō": "o"}},
        method_params=params, verbose=False,
    )
    assessor.fit(unlabeled=pd.concat([coded, uncoded]), references=infos[["info", "topic"]])
    report = assessor.evaluate(coded.assign(included=coded["included"].astype(int)))
    assert report.metrics["f1"] == pytest.approx(expected_f1, abs=0.02)
