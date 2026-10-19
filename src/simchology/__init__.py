"""Simchology: text similarity assessment for psychologists.

Quick start::

    import simchology as sim

    df = sim.load_demo("en")
    result = sim.assess(df, columns=("text", "info"), method="tfidf", output="binary",
                        coded=df, label_col="label", group_col="topic")

See :func:`assess` for all options and :func:`list_methods` for the available methods.
"""

from .hardware.runtime import configure_environment as _configure_environment

_configure_environment()

from ._version import __version__  # noqa: E402
from .api import assess, baseline, compare_methods, download_resources, explain  # noqa: E402
from .assessor import SimilarityAssessor  # noqa: E402
from .baseline import (  # noqa: E402
    Baseline,
    BaselineQuantileThreshold,
    FixedThreshold,
    OptimizedThreshold,
    ProbabilityThreshold,
    ReferenceThreshold,
    Threshold,
)
from .data.augmentation import augment  # noqa: E402
from .data.splitting import Splits, split_coded  # noqa: E402
from .datasets.demo import load_demo  # noqa: E402
from .evaluation.report import EvaluationReport  # noqa: E402
from .exceptions import (  # noqa: E402
    BackendUnavailableError,
    ConfigurationError,
    HardwareInsufficientError,
    MissingDependencyError,
    ModelNotAvailableError,
    NotFittedError,
    RemoteNotAllowedError,
    SimchologyDataWarning,
    SimchologyError,
    SimchologyHardwareWarning,
    SimchologyPrivacyWarning,
    SimchologyReproducibilityWarning,
    SimchologyWarning,
)
from .explain.explanation import Explanation  # noqa: E402
from .hardware import hardware_report  # noqa: E402
from .methods.registry import list_methods  # noqa: E402


def load(path, *, trust: bool = False, device: str = "auto") -> SimilarityAssessor:
    """Load an assessor saved with :meth:`SimilarityAssessor.save`."""
    return SimilarityAssessor.load(path, trust=trust, device=device)


__all__ = [
    "BackendUnavailableError", "Baseline", "BaselineQuantileThreshold", "ConfigurationError",
    "EvaluationReport", "Explanation", "FixedThreshold", "HardwareInsufficientError",
    "MissingDependencyError", "ModelNotAvailableError", "NotFittedError", "OptimizedThreshold",
    "ProbabilityThreshold", "ReferenceThreshold", "RemoteNotAllowedError", "SimchologyDataWarning",
    "SimchologyError", "SimchologyHardwareWarning", "SimchologyPrivacyWarning",
    "SimchologyReproducibilityWarning", "SimchologyWarning", "SimilarityAssessor", "Splits",
    "Threshold", "__version__", "assess", "augment", "baseline", "compare_methods",
    "download_resources", "explain", "hardware_report", "list_methods", "load", "load_demo",
    "split_coded",
]
