"""Saving and loading fitted assessors.

An assessor is saved as a directory::

    my_assessor/
      simchology.json   configuration, fitted thresholds/scaling, label info, provenance
      baseline.npz      baseline scores and pairs (if a baseline was used)
      report.json       evaluation on the held-out coded pairs (if any)
      method/           fitted method state (vocabularies, LSA/LDA arrays, model weights)

No pickle files are written for the built-in methods, so loading is safe. Custom
methods that store pickles require ``trust=True``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._version import __version__
from .baseline.baseline import Baseline
from .baseline.scaling import GradientScaler
from .baseline.thresholds import Threshold
from .data.labels import LabelInfo
from .evaluation.report import EvaluationReport
from .exceptions import ConfigurationError, NotFittedError

SCHEMA_VERSION = 1


def save_assessor(assessor, path) -> Path:
    from .assessor import _jsonable

    if not assessor._fitted:
        raise NotFittedError("Only fitted assessors can be saved.")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    a = assessor
    preprocess = dict(a.preprocess)
    preprocess.update(backend=a.preprocessor_.backend, unit=a.preprocessor_.unit)
    config = {
        "schema_version": SCHEMA_VERSION,
        "simchology_version": __version__,
        "method": a.method_name,
        "method_params": _jsonable(a.method_.get_params()),
        "columns": list(a.columns),
        "output": a.output,
        "group_col": a.group_col,
        "threshold": a.threshold if isinstance(a.threshold, (str, int, float)) or a.threshold is None else None,
        "baseline_quantile": a.baseline_quantile,
        "baseline": a.baseline,
        "baseline_pairs": a.baseline_pairs,
        "gradient_scale": a.gradient_scale,
        "positive_threshold": a.positive_threshold,
        "positive_label": a.positive_label,
        "language": a.language_,
        "segment": a.segmenter_.kind,
        "segment_references": a.segment_references,
        "aggregate": a.aggregate,
        "preprocess": preprocess,
        "check_hardware": a.check_hardware,
        "allow_remote": a.allow_remote,
        "missing": a.missing,
        "seed": a.seed,
        "output_col": a.output_col,
        "label_col": a.label_col_,
        "label_info": a.label_info_.to_dict() if a.label_info_ else None,
        "threshold_state": a.threshold_.to_dict() if a.threshold_ else None,
        "scaler_state": a.scaler_.to_dict() if a.scaler_ else None,
        "baseline_meta": ({"strategy": a.baseline_.strategy, "seed": a.baseline_.seed}
                          if a.baseline_ is not None else None),
        "splits": (json.loads(a.splits_.summary.to_json(orient="records"))
                   if a.splits_ is not None and len(a.splits_.summary) else None),
        "provenance": a.provenance_,
    }
    (path / "simchology.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if a.baseline_ is not None:
        np.savez(path / "baseline.npz", **a.baseline_.to_arrays())
    if a.report_ is not None:
        (path / "report.json").write_text(json.dumps(a.report_.to_dict(), indent=2, default=float),
                                          encoding="utf-8")
    a.method_.save(path / "method")
    return path


def load_assessor(path, *, trust: bool = False, device: str = "auto"):
    from .assessor import SimilarityAssessor

    path = Path(path)
    cfg_file = path / "simchology.json"
    if not cfg_file.exists():
        raise ConfigurationError(f"{path} is not a saved Simchology assessor (no simchology.json).")
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    if cfg.get("schema_version", 0) > SCHEMA_VERSION:
        raise ConfigurationError(
            f"This assessor was saved with a newer Simchology ({cfg.get('simchology_version')}); "
            "upgrade Simchology to load it."
        )
    params = {k: v for k, v in cfg["method_params"].items()}
    a = SimilarityAssessor(
        cfg["method"], cfg["output"], columns=tuple(cfg["columns"]), group_col=cfg["group_col"],
        threshold=cfg["threshold"] if cfg["threshold"] is not None else "auto",
        baseline_quantile=cfg["baseline_quantile"], baseline=cfg["baseline"],
        baseline_pairs=cfg["baseline_pairs"], gradient_scale=cfg["gradient_scale"],
        positive_threshold=cfg["positive_threshold"], positive_label=cfg["positive_label"],
        language=cfg["language"], segment=cfg["segment"], segment_references=cfg["segment_references"],
        aggregate=cfg["aggregate"], preprocess=cfg["preprocess"], method_params=params,
        check_hardware=cfg["check_hardware"], allow_remote=cfg["allow_remote"], missing=cfg["missing"],
        seed=cfg["seed"], output_col=cfg["output_col"], device=device, verbose=False,
    )
    a.label_col_ = cfg["label_col"]
    a.label_info_ = LabelInfo.from_dict(cfg["label_info"]) if cfg["label_info"] else None
    a._setup([""])
    a.runtime_.label_info = a.label_info_
    a.method_.load(path / "method", trust=trust)
    a.threshold_ = Threshold.from_dict(cfg["threshold_state"]) if cfg["threshold_state"] else None
    a.scaler_ = GradientScaler.from_dict(cfg["scaler_state"]) if cfg["scaler_state"] else None
    if cfg.get("baseline_meta") and (path / "baseline.npz").exists():
        with np.load(path / "baseline.npz", allow_pickle=False) as z:
            a.baseline_ = Baseline.from_arrays(dict(z), method=a.method_name, **cfg["baseline_meta"])
    if (path / "report.json").exists():
        a.report_ = EvaluationReport.from_dict(json.loads((path / "report.json").read_text(encoding="utf-8")))
    a.provenance_ = cfg.get("provenance", {})
    a._fitted = True
    a.verbose = True
    return a
