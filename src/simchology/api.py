"""One-call functions: :func:`assess`, :func:`baseline`, :func:`compare_methods`, :func:`explain`."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ._utils.logging import note
from ._utils.warn import warn
from .assessor import SimilarityAssessor
from .baseline.baseline import Baseline
from .config import DEFAULT_MODELS, DEFAULT_OUTPUT_COL
from .data.labels import guess_label_col
from .data.splitting import split_coded
from .exceptions import (
    BackendUnavailableError,
    ConfigurationError,
    MissingDependencyError,
    ModelNotAvailableError,
    SimchologyDataWarning,
)
from .methods.base import SimilarityMethod
from .methods.registry import PRESETS, get_method_class, make_method


def assess(
    data: pd.DataFrame,
    columns: Sequence[str] = ("text", "info"),
    method: str | SimilarityMethod = "sbert",
    output: str = "gradient",
    *,
    coded: pd.DataFrame | None = None,
    label_col: str | None = None,
    group_col: str | None = None,
    references: pd.DataFrame | Sequence[str] | None = None,
    corpus: Sequence[str] | pd.DataFrame | None = None,
    threshold: Any = "auto",
    baseline_quantile: float = 0.95,
    baseline: str = "auto",
    baseline_pairs: int = 2000,
    gradient_scale: str = "raw",
    positive_threshold: float | None = None,
    positive_label: Any = None,
    language: str = "auto",
    segment: str = "auto",
    segment_references: bool = False,
    aggregate: str = "max",
    preprocess: dict | None = None,
    split_by: str | None = "auto",
    test_size: float = 0.2,
    dev_size: float = 0.1,
    augment: bool | str = "auto",
    refit_on_all_coded: bool = False,
    method_params: dict | None = None,
    device: str = "auto",
    check_hardware: str = "warn",
    allow_remote: bool = False,
    missing: str = "nan",
    seed: int = 42,
    output_col: str = DEFAULT_OUTPUT_COL,
    keep_scores: bool = False,
    verbose: bool = True,
    return_assessor: bool = False,
):
    """Add a ``predicted_similarity`` column to ``data``.

    Parameters
    ----------
    data:
        DataFrame with one row per pair of texts to compare.
    columns:
        ``(response, reference)``: the two text columns. For asymmetric methods the
        second column is the reference (e.g. the information that should be recalled).
    method:
        A method name (see :func:`list_methods`), e.g. ``"jaccard"``, ``"tfidf"``,
        ``"sbert"``, ``"bertscore"``, ``"cross_encoder"``, ``"llm"`` or ``"systemone"``.
    output:
        ``"binary"`` (0/1: similar or not) or ``"gradient"`` (a continuous score).
    coded:
        Optional coded pairs with human judgments (``label_col``). They are split into
        training and held-out test pairs: the training pairs are used for fine-tuning,
        baselines and thresholds, the test pairs only to report agreement.
        If ``coded`` is None but ``label_col`` is a column of ``data``, the rows of
        ``data`` that have a label are used as coded data.
    group_col:
        Optional column with the topic/stimulus/item of each pair, used to stratify the
        split and to build baselines by re-pairing texts across groups.
    references:
        Extra references (e.g. all infos of a study) for reference-based thresholds.
    corpus:
        Extra texts to learn vocabularies/idf/topics from (lexical and topic methods).
    threshold:
        How binary decisions are made: ``"auto"`` (0.5 for probabilities, otherwise the
        baseline quantile), ``"baseline"``, ``"baseline_group"``, ``"optimize"`` (max F1 on
        the coded training pairs), ``"optimize_youden"``, ``"reference_a"`` ...
        ``"reference_d"`` (the variants of the original coding scripts) or a number.
    baseline_quantile:
        Quantile of the dissimilar-pair baseline used as threshold (default 0.95).
    baseline:
        How dissimilar baseline pairs are built: ``"auto"``, ``"coded_negatives"``,
        ``"cross_group"``, ``"random"`` or ``"coded_negatives+cross_group"``.
    gradient_scale:
        ``"raw"`` scores, or relative to the baseline (``"baseline_percentile"``,
        ``"baseline_z"``, ``"baseline_rescaled"``), or ``"calibrated"`` against the codes.
    positive_threshold:
        For gradient labels with binary output: label values >= this count as similar.
    language:
        ``"auto"`` or a language code (``"de"``, ``"en"``, ...).
    segment:
        Sentence segmentation of responses: ``"auto"``, ``"none"``, ``"regex"``,
        ``"spacy"`` or ``"sat"`` (robust to missing punctuation).
    aggregate:
        How sentence scores are combined: ``"max"`` (best sentence), ``"mean"``, ``"max_mean"``.
    preprocess:
        Preprocessing options for string/lexical/topic methods, e.g.
        ``{"unit": "lemma", "remove_stopwords": False, "replacements": {"10m": "10 Meter"}}``.
    split_by:
        ``"auto"`` keeps all rows of a response in the same split; ``"row"`` splits rows;
        or a column name (e.g. ``"participant"``).
    method_params:
        Parameters of the method, e.g. ``{"model": "..."}`` (see the method's docstring).
    check_hardware:
        ``"warn"`` (default), ``"error"`` or ``"off"``.
    allow_remote:
        Allow sending texts to non-local servers (hosted LLM APIs). Off by default.
    keep_scores:
        Also add the raw ``similarity_score`` (and the threshold used).
    return_assessor:
        Also return the fitted :class:`SimilarityAssessor` (for evaluation details,
        explanations, saving).
    """
    if coded is None and label_col is not None and label_col in data.columns:
        coded = data[data[label_col].notna()]
    assessor = SimilarityAssessor(
        method, output, columns=columns, group_col=group_col, threshold=threshold,
        baseline_quantile=baseline_quantile, baseline=baseline, baseline_pairs=baseline_pairs,
        gradient_scale=gradient_scale, positive_threshold=positive_threshold,
        positive_label=positive_label, language=language, segment=segment,
        segment_references=segment_references, aggregate=aggregate, preprocess=preprocess,
        method_params=method_params, device=device, check_hardware=check_hardware,
        allow_remote=allow_remote, missing=missing, seed=seed, output_col=output_col, verbose=verbose,
    )
    assessor.fit(
        coded, label_col=label_col, unlabeled=data, corpus=corpus, references=references,
        split_by=split_by, test_size=test_size, dev_size=dev_size, augment=augment,
        refit_on_all_coded=refit_on_all_coded,
    )
    if coded is not None and assessor.method_.info.requires_labels and not refit_on_all_coded:
        _warn_in_sample(data, coded, assessor)
    if verbose and assessor.report_ is not None:
        print(assessor.report_)
    if coded is None and verbose and output == "binary":
        note("No coded data were given, so the agreement of these judgments with human "
             "coders is unknown.", verbose)
    out = assessor.predict(data, keep_scores=keep_scores)
    out.attrs["simchology"] = {
        "method": assessor.method_name, "output": output,
        "threshold": assessor.threshold_.describe() if assessor.threshold_ else None,
        "metrics": assessor.report_.metrics if assessor.report_ else None,
    }
    return (out, assessor) if return_assessor else out


def _warn_in_sample(data, coded, assessor) -> None:
    a, b = assessor.columns
    if a not in coded or b not in coded:
        return
    train = assessor.splits_.train if assessor.splits_ is not None else coded
    seen = set(zip(train[a].astype(str), train[b].astype(str)))
    overlap = sum((x, y) in seen for x, y in zip(data[a].astype(str), data[b].astype(str)))
    if overlap:
        warn(
            f"{overlap} row(s) of data were also used to train '{assessor.method_name}'; their "
            "predictions are in-sample and will look better than on new data.",
            SimchologyDataWarning,
        )


def baseline(
    data: pd.DataFrame,
    columns: Sequence[str] = ("text", "info"),
    method: str | SimilarityMethod = "sbert",
    *,
    label_col: str | None = None,
    group_col: str | None = None,
    strategy: str = "auto",
    n_pairs: int = 2000,
    method_params: dict | None = None,
    language: str = "auto",
    segment: str = "auto",
    aggregate: str = "max",
    preprocess: dict | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> Baseline:
    """Score dissimilar pairs built from ``data`` to see what values unrelated texts get.

    With ``label_col``, pairs coded as dissimilar are included; with ``group_col``,
    responses are re-paired with references of other groups. See
    :func:`~simchology.baseline.build_baseline_pairs`.
    """
    assessor = SimilarityAssessor(
        method, "gradient", columns=columns, group_col=group_col, baseline=strategy,
        baseline_pairs=n_pairs, language=language, segment=segment, aggregate=aggregate,
        preprocess=preprocess, method_params=method_params, seed=seed, verbose=verbose,
    )
    coded = data if label_col else None
    assessor.fit(coded, label_col=label_col, unlabeled=data, test_size=0.0)
    std, mask = assessor._standardize(data, labels=label_col is not None, what="the data")
    return assessor._compute_baseline(std[mask])


def _method_specs(methods) -> list[tuple[str, Any, dict]]:
    if isinstance(methods, str) and methods in PRESETS:
        methods = PRESETS[methods]
    elif isinstance(methods, (str, SimilarityMethod)):
        methods = [methods]
    specs = []
    for m in methods:
        if isinstance(m, SimilarityMethod):
            specs.append((m.name, m, {}))
        elif isinstance(m, tuple):
            name, params = m
            specs.append((f"{name} {params}" if params else name, name, dict(params)))
        elif isinstance(m, dict):
            m = dict(m)
            name = m.pop("method")
            label = m.pop("label", None)
            params = m.pop("params", m)
            specs.append((label or (f"{name} {params}" if params else name), name, params))
        else:
            specs.append((str(m), m, {}))
    return specs


def compare_methods(
    coded: pd.DataFrame,
    methods: str | Sequence[Any] = "fast",
    *,
    columns: Sequence[str] = ("text", "info"),
    label_col: str | None = None,
    group_col: str | None = None,
    output: str = "binary",
    threshold: Any = "auto",
    baseline_quantile: float = 0.95,
    gradient_scale: str = "raw",
    positive_threshold: float | None = None,
    references: pd.DataFrame | Sequence[str] | None = None,
    corpus: Sequence[str] | pd.DataFrame | None = None,
    language: str = "auto",
    segment: str = "auto",
    split_by: str | None = "auto",
    test_size: float = 0.2,
    dev_size: float = 0.1,
    include_expensive: bool = False,
    skip_unavailable: bool = True,
    check_hardware: str = "warn",
    allow_remote: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate several methods on the same held-out coded pairs.

    ``methods`` is a preset (``"fast"``: string and lexical methods; ``"standard"``: adds
    topic, word-vector and neural embedding methods; ``"all"``) or a list of method
    names, ``(name, params)`` tuples, ``{"method": ..., "params": {...}}`` dicts or method
    instances. Expensive methods (fine-tuning, LLMs) only run with
    ``include_expensive=True``. Returns one row per method with its agreement metrics.
    """
    label_col = label_col or guess_label_col(coded)
    if label_col is None:
        raise ConfigurationError("Pass label_col=... (the column with the human judgments).")
    splits = split_coded(
        coded, label_col=label_col, columns=tuple(columns),
        group_col=group_col if group_col in coded else None, split_by=split_by,
        test_size=test_size, dev_size=dev_size, seed=seed, verbose=verbose,
    )
    rows = []
    for label, spec, params in _method_specs(methods):
        row: dict[str, Any] = {"method": label}
        try:
            cls = spec.__class__ if isinstance(spec, SimilarityMethod) else get_method_class(spec)
            row["family"] = cls.info.family
            if cls.info.cost_tier == "heavy" and not include_expensive:
                row["status"] = "skipped: expensive (use include_expensive=True)"
                rows.append(row)
                continue
            method = make_method(spec, params)
            note(f"Evaluating '{label}' ...", verbose)
            t0 = time.perf_counter()
            assessor = SimilarityAssessor(
                method, output, columns=columns, group_col=group_col, threshold=threshold,
                baseline_quantile=baseline_quantile, gradient_scale=gradient_scale,
                positive_threshold=positive_threshold, language=language, segment=segment,
                check_hardware=check_hardware, allow_remote=allow_remote, seed=seed, verbose=False,
            )
            assessor.fit(label_col=label_col, splits=splits, corpus=corpus, references=references,
                         unlabeled=None)
            row["seconds"] = round(time.perf_counter() - t0, 2)
            report = assessor.report_
            row["status"] = "ok"
            row["rule"] = report.threshold if report else ""
            row.update({k: v for k, v in (report.metrics if report else {}).items()})
        except (MissingDependencyError, BackendUnavailableError, ModelNotAvailableError) as exc:
            if not skip_unavailable:
                raise
            row["status"] = "skipped: " + str(exc).splitlines()[0][:200]
        except Exception as exc:  # keep going: one failing method should not stop the comparison
            if not skip_unavailable:
                raise
            row["status"] = f"failed: {type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
        rows.append(row)
    table = pd.DataFrame(rows)
    key = "f1" if output == "binary" else "spearman_rho"
    if key in table:
        table = table.sort_values(key, ascending=False, na_position="last").reset_index(drop=True)
    return table


def explain(
    response: str,
    reference: str,
    method: str | SimilarityMethod = "bertscore",
    *,
    method_params: dict | None = None,
    language: str = "auto",
    segment: str = "auto",
    top_k: int = 10,
):
    """Explain which parts of ``response`` and ``reference`` made them (dis)similar."""
    df = pd.DataFrame({"text": [response], "info": [reference]})
    assessor = SimilarityAssessor(method, "gradient", columns=("text", "info"), language=language,
                                  segment=segment, method_params=method_params, verbose=False)
    assessor.fit(unlabeled=df)
    return assessor.explain(response, reference, top_k=top_k)


def download_resources(
    language: str = "de",
    methods: Sequence[str] = ("tfidf", "word_vectors", "sbert", "bertscore"),
    *,
    sat: bool = False,
) -> None:
    """Download the models needed for ``methods`` now (e.g. before going offline).

    Installs the spaCy pipeline for ``language`` and pre-downloads Hugging Face models.
    Ollama models are pulled with ``ollama pull <model>``.
    """
    from ._optional import is_available
    from .text.spacy_models import default_spacy_model, spacy_model_installed

    name = default_spacy_model(language)
    if name and is_available("spacy") and not spacy_model_installed(name):
        import spacy.cli

        note(f"Downloading spaCy pipeline {name} ...")
        spacy.cli.download(name)
    for m in methods:
        cls = get_method_class(m)
        model = cls.defaults.get("model") or cls.defaults.get("backbone") or DEFAULT_MODELS.get(cls.info.name)
        if cls.info.family == "prompt":
            note(f"'{cls.info.name}' runs on a local server; e.g. `ollama pull {DEFAULT_MODELS['ollama']}`.")
            continue
        if model and "/" in str(model) and is_available("huggingface_hub"):
            from huggingface_hub import snapshot_download

            note(f"Downloading {model} ...")
            snapshot_download(model)
    if sat:
        from ._optional import require

        require("wtpsplit", feature="SaT segmentation", extra="sat").SaT(DEFAULT_MODELS["sat"])

