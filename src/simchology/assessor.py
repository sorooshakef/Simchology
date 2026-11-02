"""The :class:`SimilarityAssessor`: fit once on coded data, then predict similarity for any data."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._utils.logging import note
from ._utils.provenance import collect_provenance, format_model, spacy_model_record
from ._utils.seeds import derive_seed, set_seed
from ._utils.warn import warn
from .baseline.baseline import Baseline
from .baseline.construct import build_baseline_pairs
from .baseline.scaling import GradientScaler, RawScaler, make_scaler
from .baseline.thresholds import Threshold, make_threshold
from .config import DEFAULT_OUTPUT_COL, GROUP, LABEL, LEFT, RIGHT, SCORE_COL, THRESHOLD_COL
from .data.augmentation import augment as augment_frame
from .data.augmentation import share_lacking_terminal_punct
from .data.labels import (
    LabelInfo,
    binarize_labels,
    detect_label_type,
    guess_label_col,
    to_numeric_labels,
)
from .data.splitting import Splits, split_coded
from .data.validation import coerce_text, validate_columns, validate_pair_columns
from .evaluation.metrics import binary_metrics, bootstrap_ci, gradient_metrics
from .evaluation.report import EvaluationReport
from .evaluation.robustness import robustness_report
from .exceptions import ConfigurationError, NotFittedError, SimchologyDataWarning
from .explain.explanation import Explanation
from .methods.base import Runtime, SimilarityMethod
from .methods.registry import make_method
from .scoring.pipeline import ScoringPipeline
from .text.preprocess import PreprocessConfig, Preprocessor
from .text.segment import Segmenter
from .text.tokenize import detect_language

OUTPUTS = ("binary", "gradient")


class SimilarityAssessor:
    """Assess the similarity of text pairs with one of Simchology's methods.

    Typical use::

        assessor = SimilarityAssessor("sbert", output="binary", columns=("text", "info"))
        assessor.fit(coded_df, label_col="included", group_col="topic")
        print(assessor.report_)              # agreement with the coders on held-out pairs
        result = assessor.predict(new_df)    # adds a 'predicted_similarity' column

    See :func:`simchology.assess` for a description of all parameters.
    """

    def __init__(
        self,
        method: str | SimilarityMethod = "sbert",
        output: str = "gradient",
        *,
        columns: Sequence[str] = ("text", "info"),
        group_col: str | None = None,
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
        method_params: dict | None = None,
        device: str = "auto",
        check_hardware: str = "warn",
        allow_remote: bool = False,
        missing: str = "nan",
        seed: int = 42,
        output_col: str = DEFAULT_OUTPUT_COL,
        verbose: bool = True,
    ) -> None:
        if output not in OUTPUTS:
            raise ConfigurationError(f"output must be 'binary' or 'gradient'; got {output!r}")
        if check_hardware not in ("warn", "error", "off"):
            raise ConfigurationError("check_hardware must be 'warn', 'error' or 'off'")
        self.columns = validate_pair_columns(columns)
        self.output = output
        self.group_col = group_col
        self.threshold = threshold
        self.baseline_quantile = baseline_quantile
        self.baseline = baseline
        self.baseline_pairs = baseline_pairs
        self.gradient_scale = gradient_scale
        self.positive_threshold = positive_threshold
        self.positive_label = positive_label
        self.language = language
        self.segment = segment
        self.segment_references = segment_references
        self.aggregate = aggregate
        self.preprocess = dict(preprocess or {})
        self.device = device
        self.check_hardware = check_hardware
        self.allow_remote = allow_remote
        self.missing = missing
        self.seed = seed
        self.output_col = output_col
        self.verbose = verbose
        self.method_ = make_method(method, method_params)
        self.method_name = self.method_.name
        # fitted state
        self.language_: str | None = None
        self.preprocessor_: Preprocessor | None = None
        self.segmenter_: Segmenter | None = None
        self.pipeline_: ScoringPipeline | None = None
        self.runtime_: Runtime | None = None
        self.label_col_: str | None = None
        self.label_info_: LabelInfo | None = None
        self.splits_: Splits | None = None
        self.baseline_: Baseline | None = None
        self.threshold_: Threshold | None = None
        self.scaler_: GradientScaler | None = None
        self.report_: EvaluationReport | None = None
        self.references_: pd.DataFrame | None = None
        self.provenance_: dict = {}
        self.fit_seconds_: float | None = None
        self._fitted = False
        self._inference_checked = False

    # ------------------------------------------------------------------ setup helpers
    def _resolve_language(self, texts: Sequence[str]) -> str:
        if self.language != "auto":
            return self.language
        lang = detect_language(texts)
        if lang is None:
            note("Could not detect the language of the texts; assuming English. "
                 "Pass language='de' (etc.) to set it.", self.verbose)
            return "en"
        note(f"Detected language: {lang!r} (set language=... to override).", self.verbose)
        return lang

    def _setup(self, sample_texts: Sequence[str]) -> None:
        from .hardware.device import resolve_device

        self.language_ = self._resolve_language(sample_texts)
        info = self.method_.info
        self.preprocessor_ = Preprocessor(
            PreprocessConfig.from_options(self.language_, self.preprocess), verbose=self.verbose
        )
        device = resolve_device(self.device)
        if self.segment == "auto":
            kind = "auto" if info.segment_default else "none"
        else:
            kind = self.segment
        self.segmenter_ = Segmenter(
            kind, self.language_, spacy_model=self.preprocess.get("spacy_model"),
            device=device, verbose=self.verbose,
        )
        self.runtime_ = Runtime(
            device=device, seed=self.seed, check_hardware=self.check_hardware,
            allow_remote=self.allow_remote, verbose=self.verbose, language=self.language_,
            output=self.output, label_info=self.label_info_,
        )
        self.method_.bind(self.preprocessor_, self.runtime_)
        seg_left = None if self.segmenter_.kind == "none" else self.segmenter_.split
        seg_right = self.segmenter_.split if (self.segment_references and seg_left) else None
        self.pipeline_ = ScoringPipeline(
            self.method_, segment_left=seg_left, segment_right=seg_right, aggregate=self.aggregate,
            light_normalize_inputs=info.preprocessing == "light",
            show_progress=self.verbose and info.cost_tier != "light",
        )
        if seg_left is not None:
            self.segmenter_.hint_if_unreliable(list(dict.fromkeys(sample_texts))[:2000])

    def _standardize(self, df: pd.DataFrame, *, labels: bool, what: str) -> tuple[pd.DataFrame, np.ndarray]:
        validate_columns(df, self.columns, what=what)
        left, right, mask = coerce_text(df, self.columns, missing=self.missing, what=what, warn=False)
        std = pd.DataFrame({LEFT: left, RIGHT: right}, index=df.index)
        if self.group_col and self.group_col in df:
            std[GROUP] = df[self.group_col].astype(str).to_numpy()
        if labels:
            std[LABEL] = to_numeric_labels(df[self.label_col_], self.positive_label)
            mask = mask & ~np.isnan(std[LABEL].to_numpy())
        return std, mask

    def _check_hardware(self, task: str, n_items: int) -> None:
        self.method_.check_hardware(task, n_items)

    def _reference_pipeline(self) -> ScoringPipeline:
        return ScoringPipeline(self.method_, aggregate="max",
                               light_normalize_inputs=self.method_.info.preprocessing == "light")

    # ------------------------------------------------------------------------- fit
    def fit(
        self,
        coded: pd.DataFrame | None = None,
        *,
        label_col: str | None = None,
        unlabeled: pd.DataFrame | None = None,
        corpus: Sequence[str] | pd.DataFrame | None = None,
        references: pd.DataFrame | Sequence[str] | None = None,
        split_by: str | None = "auto",
        test_size: float = 0.2,
        dev_size: float = 0.1,
        augment: bool | str = "auto",
        splits: Splits | None = None,
        refit_on_all_coded: bool = False,
    ) -> SimilarityAssessor:
        """Fit the method, its baseline/threshold and evaluate on held-out coded pairs.

        Parameters
        ----------
        coded:
            Coded pairs (with a label column). Split into training (and development) and
            test pairs; the test pairs are used only for evaluation.
        unlabeled:
            Uncoded pairs. Used without labels: as corpus for vocabulary-based methods,
            as source of references, and for baselines when no coded data are given.
        corpus:
            Additional texts (list or DataFrame with the response column) for fitting
            vocabularies, idf weights and topic models.
        references:
            Additional references (list, or DataFrame with the reference column and
            optionally ``group_col``) for reference-based thresholds.
        splits:
            Pre-computed :class:`~simchology.data.splitting.Splits` (as used by
            :func:`compare_methods` so that every method is tested on the same pairs).
        """
        t0 = time.perf_counter()
        set_seed(self.seed)
        info = self.method_.info
        supervised = info.requires_labels
        if coded is None and splits is None and supervised:
            raise ConfigurationError(
                f"The method '{self.method_name}' is trained on coded data: pass coded=... "
                "with human similarity judgments."
            )
        sample = []
        for df in (coded, unlabeled, splits.train if splits else None):
            if df is not None:
                validate_columns(df, self.columns, what="the data")
                sample += df[self.columns[0]].dropna().astype(str).tolist()[:500]

        # labels and splits
        train_df = dev_df = test_df = None
        if coded is not None or splits is not None:
            source = coded if coded is not None else pd.concat(
                [d for d in (splits.train, splits.dev, splits.test) if d is not None])
            self.label_col_ = label_col or guess_label_col(source)
            if self.label_col_ is None:
                raise ConfigurationError(
                    "Could not find the label column in the coded data. Pass label_col=... "
                    f"(columns are: {', '.join(map(str, source.columns))})."
                )
            validate_columns(source, [self.label_col_], what="the coded data")
            self.label_info_ = detect_label_type(source[self.label_col_], positive_label=self.positive_label)
            if splits is None:
                if test_size > 0:
                    splits = split_coded(
                        coded, label_col=self.label_col_, columns=self.columns,
                        group_col=self.group_col if self.group_col in coded else None,
                        split_by=split_by, test_size=test_size,
                        dev_size=dev_size if supervised else 0.0, seed=self.seed,
                        positive_label=self.positive_label, verbose=self.verbose,
                    )
                else:
                    labelled = coded[coded[self.label_col_].notna()]
                    splits = Splits(labelled, None, labelled.iloc[:0], None, self.seed, pd.DataFrame())
            self.splits_ = splits
            train_df, dev_df, test_df = splits.train, splits.dev, splits.test
            if not supervised and dev_df is not None and len(dev_df):
                train_df, dev_df = pd.concat([train_df, dev_df]), None
            if supervised and (dev_df is None or not len(dev_df)):
                dev_df = None

        self._setup(sample or [""])
        self.runtime_.label_info = self.label_info_
        self._warn_column_order(coded if coded is not None else unlabeled)

        # unlabeled corpus / references
        std_train = std_dev = None
        if train_df is not None:
            std_train, m = self._standardize(train_df, labels=True, what="the coded data")
            std_train = std_train[m]
            if dev_df is not None:
                std_dev, m = self._standardize(dev_df, labels=True, what="the coded data")
                std_dev = std_dev[m]
        std_unlab = None
        if unlabeled is not None:
            std_unlab, m = self._standardize(unlabeled, labels=False, what="the data")
            std_unlab = std_unlab[m]

        # fit the method
        if info.requires_corpus_fit:
            texts_l, texts_r = [], []
            for f in (std_train, std_dev, std_unlab):
                if f is not None:
                    texts_l += f[LEFT].tolist()
                    texts_r += f[RIGHT].tolist()
            if test_df is not None and len(test_df):
                # test texts are used without labels, as in the original transductive setup
                std_test, m = self._standardize(test_df, labels=False, what="the coded data")
                texts_l += std_test[LEFT][m].tolist()
                texts_r += std_test[RIGHT][m].tolist()
            texts_l += _corpus_texts(corpus, self.columns[0])
            texts_r += _reference_texts(references, self.columns[1])
            units = self.pipeline_.units(texts_l, texts_r)
            if not units:
                raise ConfigurationError(f"The method '{self.method_name}' needs texts to learn its vocabulary.")
            self.method_.fit(corpus=units)
        elif supervised or info.uses_training_pairs:
            train_for_method, dev_for_method = std_train, std_dev
            if supervised and self._use_augmentation(augment, std_train):
                train_for_method = _augment_std(std_train, derive_seed(self.seed, 0))
                if std_dev is not None:
                    dev_for_method = _augment_std(std_dev, derive_seed(self.seed, 1))
                note(f"Augmenting the training data with perturbed response endings "
                     f"({len(std_train):,} -> {len(train_for_method):,} pairs).", self.verbose)
            self.method_.fit(train=train_for_method, dev=dev_for_method)
        else:
            self.method_.fit()

        self._fit_decision_rule(std_train, std_dev, std_unlab, coded, unlabeled, references)
        self._fitted = True

        if test_df is not None and len(test_df):
            self.report_ = self._evaluate(test_df)
        self.fit_seconds_ = time.perf_counter() - t0
        self.provenance_ = collect_provenance(
            method=self.method_name, method_params=_jsonable(self.method_.get_params()),
            models=self._model_details(), device=self.runtime_.device, language=self.language_,
            preprocessing=self.preprocessor_.describe() if info.preprocessing == "full" else "whitespace only",
            segmentation=self.segmenter_.describe(), seed=self.seed,
        )
        if refit_on_all_coded and coded is not None and test_size > 0:
            report, old_splits = self.report_, self.splits_
            note("Refitting on all coded pairs (the reported metrics are from the held-out run).", self.verbose)
            self.fit(coded, label_col=self.label_col_, unlabeled=unlabeled, corpus=corpus,
                     references=references, split_by=split_by, test_size=0.0, dev_size=dev_size,
                     augment=augment)
            self.report_, self.splits_ = report, old_splits
        return self

    def _model_details(self) -> dict[str, dict]:
        """The method's models plus the pipelines used for preprocessing and sentence splitting."""
        models = dict(self.method_.model_details())
        pp = self.preprocessor_
        if self.method_.info.preprocessing == "full" and pp.backend == "spacy":
            models["preprocessing"] = spacy_model_record(pp.nlp(), pp.spacy_model_name)
        segmentation = self.segmenter_.model_details()
        if segmentation is not None:
            models["segmentation"] = segmentation
        return models

    def _use_augmentation(self, augment: bool | str, std_train: pd.DataFrame) -> bool:
        if augment == "auto":
            share = share_lacking_terminal_punct(std_train[LEFT].unique().tolist())
            return share >= 0.01
        return bool(augment)

    def _warn_column_order(self, df: pd.DataFrame | None) -> None:
        if df is None or self.method_.info.symmetric:
            return
        a = df[self.columns[0]].dropna().astype(str).str.len().mean()
        b = df[self.columns[1]].dropna().astype(str).str.len().mean()
        if b > 1.5 * a:
            warn(
                f"The reference column {self.columns[1]!r} is on average much longer than the "
                f"response column {self.columns[0]!r}. '{self.method_name}' is asymmetric and "
                "expects columns=(response, reference); check that the order is intended.",
                SimchologyDataWarning,
            )

    def _fit_decision_rule(self, std_train, std_dev, std_unlab, coded, unlabeled, references) -> None:
        native = self.method_.native_output()
        supervised = self.method_.info.requires_labels
        self.threshold_ = None
        self.scaler_ = None
        needs: set[str] = set()
        if self.output == "binary":
            self.threshold_ = make_threshold(self.threshold, native_output=native,
                                             baseline_quantile=self.baseline_quantile)
            if self.threshold_.needs:
                needs.add(self.threshold_.needs)
        else:
            kind = self.label_info_.kind if self.label_info_ else None
            self.scaler_ = make_scaler(self.gradient_scale, label_kind=kind)
            if self.scaler_.needs:
                needs.add(self.scaler_.needs)
        # pairs used to fit the decision rule: held-out dev pairs for supervised methods
        # (training pairs would make the model look too confident), else training pairs
        rule_pairs = std_dev if (supervised and std_dev is not None) else std_train

        if "labels" in needs and rule_pairs is None:
            raise ConfigurationError("This threshold/scaling needs coded data (coded=...).")
        if "baseline" in needs:
            pool = rule_pairs if rule_pairs is not None else std_unlab
            if pool is None:
                raise ConfigurationError("Building a baseline needs data (coded=... or data to assess).")
            self.baseline_ = self._compute_baseline(pool)
        fit_kwargs: dict[str, Any] = {"baseline": self.baseline_, "seed": self.seed}
        if "labels" in needs:
            scores = self.pipeline_.score(rule_pairs[LEFT].tolist(), rule_pairs[RIGHT].tolist())
            labels = rule_pairs[LABEL].to_numpy(dtype=float)
            if self.output == "binary":
                labels = binarize_labels(labels, self.label_info_, self.positive_threshold)
            fit_kwargs.update(scores=scores, labels=labels,
                              groups=rule_pairs[GROUP].to_numpy() if GROUP in rule_pairs else None)
        if "references" in needs:
            self.references_ = self._collect_references(coded, unlabeled, references)
            ref_pipe = self._reference_pipeline()
            fit_kwargs.update(references=self.references_,
                              reference_scorer=ref_pipe.score)
        if self.threshold_ is not None:
            self.threshold_.fit(**fit_kwargs)
        if self.scaler_ is not None:
            self.scaler_.fit(**fit_kwargs, upper=self.method_.score_range()[1])

    def _compute_baseline(self, pairs: pd.DataFrame) -> Baseline:
        negative_threshold = None
        if self.label_info_ is not None and self.label_info_.kind == "gradient":
            negative_threshold = self.label_info_.min
        bpairs = build_baseline_pairs(
            pairs, strategy=self.baseline, n_pairs=self.baseline_pairs,
            negative_threshold=negative_threshold, seed=derive_seed(self.seed, 3),
        )
        scores = self.pipeline_.score(bpairs[LEFT].tolist(), bpairs[RIGHT].tolist())
        baseline = Baseline(scores, bpairs, self.method_name, bpairs.attrs.get("strategy", self.baseline),
                            self.seed)
        note(f"Baseline: {len(baseline):,} dissimilar pairs ({baseline.strategy}); typical "
             f"unrelated score {baseline.median:.3f}, 95th percentile {baseline.quantile(0.95):.3f}.",
             self.verbose)
        return baseline

    def _collect_references(self, coded, unlabeled, references) -> pd.DataFrame:
        frames = []
        for df in (coded, unlabeled):
            if df is not None and self.columns[1] in df:
                cols = [self.columns[1]] + ([self.group_col] if self.group_col and self.group_col in df else [])
                frames.append(df[cols].dropna(subset=[self.columns[1]]))
        if references is not None:
            if isinstance(references, pd.DataFrame):
                col = self.columns[1] if self.columns[1] in references else references.columns[0]
                cols = [col] + ([self.group_col] if self.group_col and self.group_col in references else [])
                frames.append(references[cols].rename(columns={col: self.columns[1]}))
            else:
                frames.append(pd.DataFrame({self.columns[1]: list(references)}))
        refs = pd.concat(frames, ignore_index=True)
        out = pd.DataFrame({RIGHT: refs[self.columns[1]].astype(str).map(str.strip)})
        if self.group_col and self.group_col in refs:
            out[GROUP] = refs[self.group_col].astype(str).to_numpy()
        out = out[out[RIGHT] != ""].drop_duplicates(RIGHT).reset_index(drop=True)
        return out

    # ---------------------------------------------------------------- scoring/predict
    def _require_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError("Call fit() first (or use simchology.assess()).")

    def score(self, data: pd.DataFrame) -> pd.Series:
        """Raw similarity scores (higher = more similar), aligned with ``data.index``."""
        self._require_fitted()
        left, right, mask = coerce_text(data, self.columns, missing=self.missing)
        scores = np.full(len(data), np.nan)
        if self.method_.info.cost_tier != "light" and not self._inference_checked:
            self._inference_checked = True
            self._check_hardware("inference", int(mask.sum()))
        if mask.any():
            idx = np.flatnonzero(mask)
            scores[idx] = self.pipeline_.score([left[i] for i in idx], [right[i] for i in idx])
        return pd.Series(scores, index=data.index, name=SCORE_COL)

    def _decide(self, scores: np.ndarray, data: pd.DataFrame, output: str) -> np.ndarray:
        if output == "binary":
            if self.threshold_ is None:
                raise ConfigurationError("This assessor was fitted for gradient output; refit with output='binary'.")
            groups = data[self.group_col].to_numpy() if self.group_col and self.group_col in data else None
            refs = data[self.columns[1]].astype(str).str.strip().to_numpy()
            return self.threshold_.apply(scores, groups, refs)
        scaler = self.scaler_ or RawScaler()
        return scaler.transform(scores)

    def predict(self, data: pd.DataFrame, *, keep_scores: bool = False) -> pd.DataFrame:
        """Return a copy of ``data`` with the ``predicted_similarity`` column added."""
        self._require_fitted()
        validate_columns(data, self.columns, what="the data")
        if self.output_col in data.columns:
            warn(f"Overwriting the existing column {self.output_col!r}.", SimchologyDataWarning)
        scores = self.score(data).to_numpy()
        values = self._decide(scores, data, self.output)
        out = data.copy()
        if self.output == "binary":
            out[self.output_col] = pd.array(
                [pd.NA if np.isnan(v) else int(v) for v in values], dtype="Int8"
            )
        else:
            out[self.output_col] = values.astype(float)
        if keep_scores:
            out[SCORE_COL] = scores
            if self.output == "binary":
                groups = data[self.group_col].to_numpy() if self.group_col and self.group_col in data else None
                out[THRESHOLD_COL] = self.threshold_.values(
                    len(data), groups, data[self.columns[1]].astype(str).str.strip().to_numpy())
        return out

    # ---------------------------------------------------------------------- evaluate
    def _evaluate(self, coded: pd.DataFrame, *, robustness: bool = False, ci: float | None = None,
                  n_boot: int = 1000) -> EvaluationReport:
        std, mask = self._standardize(coded, labels=True, what="the coded data")
        std = std[mask]
        frame = coded.loc[mask] if coded.index.is_unique else coded[mask]
        scores = self.pipeline_.score(std[LEFT].tolist(), std[RIGHT].tolist())
        y = std[LABEL].to_numpy(dtype=float)
        info = self.label_info_
        notes: list[str] = []
        predictions = pd.DataFrame({LEFT: std[LEFT].to_numpy(), RIGHT: std[RIGHT].to_numpy(),
                                    "label": y, "score": scores}, index=std.index)
        per_group = confusion = None
        cis = None
        groups = std[GROUP].to_numpy() if GROUP in std else None
        cluster = std[LEFT].to_numpy()
        if self.output == "binary":
            if info.kind == "gradient" and self.positive_threshold is None:
                notes.append("labels are gradient and positive_threshold is not set, so binary "
                             "agreement cannot be computed")
                return EvaluationReport(self.method_name, self.output, info.kind, len(std), {},
                                        self.threshold_.describe(), predictions=predictions, notes=notes)
            yb = binarize_labels(y, info, self.positive_threshold)
            pred = self._decide(scores, frame, "binary")
            predictions["prediction"] = pred
            metrics = binary_metrics(yb, pred, y_score=scores)
            confusion = pd.crosstab(pd.Series(yb, name="coded"), pd.Series(pred, name="predicted"))
            if groups is not None:
                per_group = pd.DataFrame([
                    {"group": g, **binary_metrics(yb[groups == g], pred[groups == g])}
                    for g in pd.unique(groups)
                ])
            if ci:
                from sklearn.metrics import f1_score

                cis = {"f1": bootstrap_ci(lambda a, b: f1_score(a, b, zero_division=0), yb, pred,
                                          groups=cluster, n_boot=n_boot, ci=ci, seed=self.seed)}
            threshold_text = self.threshold_.describe()
        else:
            g = self._decide(scores, frame, "gradient")
            predictions["prediction"] = g
            on_scale = (self.scaler_ is not None and self.scaler_.kind == "calibrated_isotonic") or (
                self.method_.info.requires_labels and info.kind == "gradient")
            metrics = gradient_metrics(y, g, on_label_scale=on_scale)
            if info.kind == "binary":
                extra = binary_metrics(y, (g > np.nanmedian(g)).astype(float), y_score=g)
                for k in ("roc_auc", "average_precision", "point_biserial_r"):
                    if k in extra:
                        metrics[k] = extra[k]
            if groups is not None:
                per_group = pd.DataFrame([
                    {"group": grp, **gradient_metrics(y[groups == grp], g[groups == grp], on_label_scale=on_scale)}
                    for grp in pd.unique(groups)
                ])
            if ci:
                from scipy.stats import spearmanr

                cis = {"spearman_rho": bootstrap_ci(lambda a, b: spearmanr(a, b)[0], y, g, groups=cluster,
                                                    n_boot=n_boot, ci=ci, seed=self.seed)}
            threshold_text = self.scaler_.describe() if self.scaler_ else ""
        report = EvaluationReport(
            self.method_name, self.output, info.kind, len(std), metrics, threshold_text, cis,
            confusion, per_group, predictions=predictions, notes=notes,
        )
        if robustness:
            self._add_robustness(report, coded)
        return report

    def _add_robustness(self, report: EvaluationReport, coded: pd.DataFrame) -> None:
        aug = augment_frame(coded, text_col=self.columns[0], seed=derive_seed(self.seed, 2))
        std, mask = self._standardize(aug, labels=True, what="the coded data")
        aug, std = aug[mask], std[mask]
        scores = self.pipeline_.score(std[LEFT].tolist(), std[RIGHT].tolist())
        y = std[LABEL].to_numpy(dtype=float)
        if self.output == "binary":
            y = binarize_labels(y, self.label_info_, self.positive_threshold)
            pred = self._decide(scores, aug, "binary")
        else:
            pred = self._decide(scores, aug, "gradient")
        report.robustness, report.consistency = robustness_report(
            aug, label=y, prediction=pred, output=self.output)

    def evaluate(
        self, coded: pd.DataFrame | None = None, *, robustness: bool = False,
        ci: float | None = None, n_boot: int = 1000,
    ) -> EvaluationReport:
        """Agreement with coded pairs (default: the held-out test pairs from :meth:`fit`).

        ``robustness=True`` additionally scores perturbed copies of every pair (final
        period removed / last word cut) and reports how often the judgment changes.
        ``ci=0.95`` adds bootstrap confidence intervals, resampling whole responses.
        """
        self._require_fitted()
        if coded is None:
            if self.splits_ is None or not len(self.splits_.test):
                raise ConfigurationError("No held-out coded pairs are available; pass coded=...")
            coded = self.splits_.test
        elif self.label_col_ is None:
            self.label_col_ = guess_label_col(coded)
            self.label_info_ = detect_label_type(coded[self.label_col_], positive_label=self.positive_label)
        return self._evaluate(coded, robustness=robustness, ci=ci, n_boot=n_boot)

    # ----------------------------------------------------------------------- explain
    def explain(self, response: str | pd.DataFrame, reference: str | None = None, *,
                top_k: int = 10) -> Explanation | list[Explanation]:
        """Explain the score of a (response, reference) pair, or of every row of a DataFrame."""
        self._require_fitted()
        if isinstance(response, pd.DataFrame):
            left, right, _ = coerce_text(response, self.columns, missing="empty", warn=False)
            return [self.explain(a, b, top_k=top_k) for a, b in zip(left, right)]
        if reference is None:
            raise ConfigurationError("explain() needs a response and a reference text.")
        scores, best = self.pipeline_.score([response], [reference], return_details=True)
        segment = best[0] if best[0] is not None else response
        if self.method_.info.preprocessing == "light":
            from .text.normalize import light_normalize

            ref = light_normalize(reference)
        else:
            ref = reference
        exp = self.method_.explain_pair(segment, ref, top_k=top_k)
        exp.winning_segment = segment if segment != response else None
        exp.response = response
        exp.reference = reference
        exp.score = float(scores[0])
        return exp

    # ---------------------------------------------------------------------- describe
    def describe(self) -> str:
        """A paragraph describing the procedure, e.g. for the methods section of a paper."""
        self._require_fitted()
        p = self.provenance_
        parts = [
            f"Similarity was assessed with Simchology {p.get('simchology')} using "
            f"{self.method_.describe()}."
        ]
        # the preprocessing and segmentation pipelines are named in the sentences below
        models = [format_model(m) for role, m in (p.get("models") or {}).items()
                  if role not in ("preprocessing", "segmentation")]
        if models:
            parts.append("Models: " + ", ".join(models) + ".")
        if self.method_.info.preprocessing == "full" and self.method_.info.family in (
                "string", "lexical", "topic", "static", "alignment"):
            pp = self.preprocessor_.describe()
            done = "normalised and " + ("lemmatised" if pp["unit"] == "lemma" else "tokenised")
            tool = f" with spaCy ({pp['spacy_model']})" if pp["backend"] == "spacy" else ""
            removed = [w for w, on in (("stop words", pp["remove_stopwords"]),
                                       ("punctuation", pp["remove_punct"])) if on]
            tail = f", and {' and '.join(removed)} {'was' if len(removed) == 1 else 'were'} removed" if removed else ""
            parts.append(f"Texts were {done}{tool}{tail}.")
        if self.segmenter_.kind != "none":
            parts.append(
                f"Responses were split into sentences ({self.segmenter_.describe()}), and the "
                f"{'highest' if self.aggregate == 'max' else 'mean'} sentence-level similarity "
                "was used for each pair."
            )
        if self.splits_ is not None and len(self.splits_.test):
            by = f", keeping all pairs of a response together (split by {self.splits_.split_by!r})" if self.splits_.split_by else ""
            parts.append(
                f"Of {sum(len(d) for d in (self.splits_.train, self.splits_.dev, self.splits_.test) if d is not None):,} "
                f"coded pairs, {len(self.splits_.test):,} were held out for evaluation{by}."
            )
        if self.baseline_ is not None:
            sources = {"coded_negatives": "pairs coded as dissimilar",
                       "cross_group": "responses re-paired with references of other groups",
                       "random": "random re-pairings of responses and references"}
            what = " and ".join(sources.get(s, s) for s in self.baseline_.strategy.split("+"))
            parts.append(
                f"A baseline of {len(self.baseline_):,} dissimilar pairs ({what}) was scored "
                f"with the same method (median {self.baseline_.median:.3f})."
            )
        if self.output == "binary" and self.threshold_ is not None:
            parts.append(f"Pairs were classified as {self.threshold_.describe()}.")
        elif self.scaler_ is not None and self.scaler_.kind != "raw":
            parts.append(f"Scores are reported as {self.scaler_.describe()}.")
        if self.report_ is not None and self.report_.metrics:
            m = self.report_.metrics
            if self.output == "binary":
                parts.append(f"On the held-out pairs, precision was {m['precision']:.3f}, recall "
                             f"{m['recall']:.3f}, F1 {m['f1']:.3f} and Cohen's kappa {m['cohen_kappa']:.3f}.")
            elif not np.isnan(m.get("spearman_rho", np.nan)):
                parts.append(f"On the held-out pairs, scores correlated with the human codes at "
                             f"Spearman's rho = {m['spearman_rho']:.3f}.")
        return " ".join(parts)

    # ----------------------------------------------------------------------- persist
    def save(self, path: str | Path) -> Path:
        """Save the fitted assessor to a directory (see :func:`simchology.load`)."""
        from .io import save_assessor

        return save_assessor(self, path)

    @classmethod
    def load(cls, path: str | Path, *, trust: bool = False, device: str = "auto") -> SimilarityAssessor:
        """Load a saved assessor; ``device`` selects where neural models run."""
        from .io import load_assessor

        return load_assessor(path, trust=trust, device=device)

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "not fitted"
        return (f"SimilarityAssessor(method={self.method_name!r}, output={self.output!r}, "
                f"columns={self.columns}, {state})")


# --------------------------------------------------------------------------- helpers
def _corpus_texts(corpus, column: str) -> list[str]:
    if corpus is None:
        return []
    if isinstance(corpus, pd.DataFrame):
        col = column if column in corpus else corpus.columns[0]
        return corpus[col].dropna().astype(str).tolist()
    return [str(t) for t in corpus if isinstance(t, str)]


def _reference_texts(references, column: str) -> list[str]:
    return _corpus_texts(references, column)


def _augment_std(std: pd.DataFrame, seed: int) -> pd.DataFrame:
    aug = augment_frame(std, text_col=LEFT, seed=seed)
    return aug.reset_index(drop=True)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    return repr(obj)
