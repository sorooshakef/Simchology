"""Hardware detection, resource estimation and cross-platform runtime settings."""

from .checks import (
    ResourceRequest,
    Verdict,
    check_disk_space,
    check_ollama_model,
    check_resources,
    emit,
    recommend_kev_model,
)
from .device import DeviceInfo, HardwareReport, detect_hardware, device_info, resolve_device
from .estimate import MemoryEstimate, estimate_memory, estimate_runtime
from .memory import available_memory, total_memory
from .models import ModelShape, ParamInfo, model_parameter_count, model_shape
from .runtime import configure_environment, training_runtime_kwargs


def hardware_report() -> HardwareReport:
    """Describe this machine's compute devices (print it for a readable summary)."""
    return detect_hardware()


__all__ = [
    "DeviceInfo", "HardwareReport", "MemoryEstimate", "ModelShape", "ParamInfo",
    "ResourceRequest", "Verdict", "available_memory", "check_disk_space", "check_ollama_model",
    "check_resources", "configure_environment", "detect_hardware", "device_info", "emit",
    "estimate_memory", "estimate_runtime", "hardware_report", "model_parameter_count",
    "model_shape", "recommend_kev_model", "resolve_device", "total_memory",
    "training_runtime_kwargs",
]
