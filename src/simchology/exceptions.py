"""Errors and warnings raised by Simchology.

All errors derive from :class:`SimchologyError` and all warnings from
:class:`SimchologyWarning`, so users can catch or filter them in one place::

    import warnings, simchology as sim
    warnings.simplefilter("ignore", sim.SimchologyHardwareWarning)
"""

from __future__ import annotations


class SimchologyError(Exception):
    """Base class for all Simchology errors."""


class MissingDependencyError(SimchologyError, ImportError):
    """An optional package needed for the requested feature is not installed."""


class ConfigurationError(SimchologyError, ValueError):
    """An argument or combination of arguments is invalid."""


class NotFittedError(SimchologyError, AttributeError):
    """A method or assessor was used before it was fitted."""


class BackendUnavailableError(SimchologyError, ConnectionError):
    """A local server (Ollama, a System One server such as Kev) could not be reached."""


class ModelNotAvailableError(SimchologyError):
    """A model is not downloaded/pulled or does not exist."""


class RemoteNotAllowedError(SimchologyError, PermissionError):
    """A non-local endpoint was requested without ``allow_remote=True``."""


class HardwareInsufficientError(SimchologyError, RuntimeError):
    """The machine is not sufficient for a task and ``check_hardware="error"`` was set."""


class SimchologyWarning(UserWarning):
    """Base class for all Simchology warnings."""


class SimchologyHardwareWarning(SimchologyWarning):
    """The machine may be too small or too slow for the requested task."""


class SimchologyDataWarning(SimchologyWarning):
    """Something about the data may affect the validity of the results."""


class SimchologyPrivacyWarning(SimchologyWarning):
    """Participant data is about to leave the local machine."""


class SimchologyReproducibilityWarning(SimchologyWarning):
    """Results may not be exactly reproducible on other hardware or software versions."""
