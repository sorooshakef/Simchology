"""Detect the compute devices of the current machine.

Works without torch installed (then only the CPU and RAM are reported).
"""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass, field

import psutil

from .._optional import is_available

GB = 1024**3


@dataclass
class DeviceInfo:
    kind: str  # "cuda" | "mps" | "xpu" | "cpu"
    index: int | None
    name: str
    total_memory: int | None  # bytes usable by this device
    available_memory: int | None  # bytes currently free for this device
    bf16: bool = False
    fp16: bool = False
    unified_memory: bool = False

    @property
    def torch_device(self) -> str:
        return self.kind if self.index is None else f"{self.kind}:{self.index}"


@dataclass
class HardwareReport:
    os: str
    python: str
    cpu_count: int
    ram_total: int
    ram_available: int
    torch_version: str | None
    torch_cuda_build: bool
    devices: list[DeviceInfo]
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> DeviceInfo:
        order = {"cuda": 0, "mps": 1, "xpu": 2, "cpu": 3}
        return sorted(self.devices, key=lambda d: order.get(d.kind, 9))[0]

    def __str__(self) -> str:
        lines = [
            "Simchology hardware report",
            f"  OS:      {self.os}",
            f"  Python:  {self.python}",
            f"  CPU:     {self.cpu_count} logical cores",
            f"  RAM:     {self.ram_total / GB:.1f} GB total, {self.ram_available / GB:.1f} GB available",
            f"  torch:   {self.torch_version or 'not installed'}",
            "  Devices:",
        ]
        for d in self.devices:
            mem = (
                f"{d.available_memory / GB:.1f} of {d.total_memory / GB:.1f} GB available"
                if d.total_memory
                else "memory unknown"
            )
            prec = ", ".join(p for p, ok in (("bf16", d.bf16), ("fp16", d.fp16)) if ok) or "fp32"
            shared = ", shared with system RAM" if d.unified_memory else ""
            lines.append(f"    - {d.torch_device}: {d.name} ({mem}{shared}; {prec})")
        lines.append(f"  Default device: {self.best.torch_device}")
        for n in self.notes:
            lines.append(f"  Note: {n}")
        return "\n".join(lines)

    __repr__ = __str__


def _cpu_info() -> DeviceInfo:
    vm = psutil.virtual_memory()
    return DeviceInfo(
        kind="cpu",
        index=None,
        name=platform.processor() or platform.machine() or "CPU",
        total_memory=vm.total,
        available_memory=vm.available,
        bf16=False,
        fp16=False,
    )


def _torch_devices(notes: list[str]) -> list[DeviceInfo]:
    import torch

    devices: list[DeviceInfo] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
            except Exception:
                total = torch.cuda.get_device_properties(i).total_memory
                free = total
            try:
                bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                bf16 = False
            devices.append(
                DeviceInfo("cuda", i, torch.cuda.get_device_name(i), total, free, bf16=bf16, fp16=True)
            )
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        vm = psutil.virtual_memory()
        total = vm.total
        try:
            total = int(torch.mps.recommended_max_memory())
        except Exception:
            total = int(vm.total * 0.7)
        try:
            used = int(torch.mps.driver_allocated_memory())
        except Exception:
            used = 0
        available = max(0, min(total - used, vm.available + used))
        try:
            torch.ones(1, dtype=torch.bfloat16, device="mps")
            bf16 = True
        except Exception:
            bf16 = False
        devices.append(
            DeviceInfo("mps", None, "Apple GPU (Metal)", total, available, bf16=bf16,
                       fp16=False, unified_memory=True)
        )
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and hasattr(xpu, "is_available") and xpu.is_available():
        for i in range(xpu.device_count()):
            props = xpu.get_device_properties(i)
            total = getattr(props, "total_memory", None)
            try:
                free, total = xpu.mem_get_info(i)
            except Exception:
                free = total
            devices.append(DeviceInfo("xpu", i, getattr(props, "name", "Intel GPU"), total, free,
                                      bf16=True, fp16=True))
    if not devices and torch.version.cuda is None and shutil.which("nvidia-smi"):
        notes.append(
            "An NVIDIA GPU seems to be present, but the installed torch build has no CUDA "
            "support (common on Windows, where pip installs the CPU-only build). Reinstall "
            "torch with the command for your CUDA version from https://pytorch.org/get-started/locally/"
        )
    return devices


def detect_hardware() -> HardwareReport:
    """Describe the machine: OS, RAM, and every compute device torch can use."""
    notes: list[str] = []
    vm = psutil.virtual_memory()
    torch_version = None
    cuda_build = False
    devices: list[DeviceInfo] = []
    if is_available("torch"):
        import torch

        torch_version = torch.__version__
        cuda_build = torch.version.cuda is not None
        devices = _torch_devices(notes)
    else:
        notes.append(
            "torch is not installed; neural methods are unavailable. "
            'Install them with: pip install "simchology[embeddings]"'
        )
    devices.append(_cpu_info())
    return HardwareReport(
        os=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python=sys.version.split()[0],
        cpu_count=psutil.cpu_count() or 1,
        ram_total=vm.total,
        ram_available=vm.available,
        torch_version=torch_version,
        torch_cuda_build=cuda_build,
        devices=devices,
        notes=notes,
    )


def resolve_device(device: str = "auto") -> str:
    """Turn ``"auto"`` into the best available torch device string (cuda > mps > xpu > cpu)."""
    if device and device != "auto":
        return device
    if not is_available("torch"):
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and hasattr(xpu, "is_available") and xpu.is_available():
        return "xpu"
    return "cpu"


def device_info(device: str = "auto") -> DeviceInfo:
    """Return the :class:`DeviceInfo` for a torch device string such as ``"cuda:0"``."""
    resolved = resolve_device(device)
    kind, _, idx = resolved.partition(":")
    if kind == "cpu" or not is_available("torch"):
        return _cpu_info()
    for d in _torch_devices([]):
        if d.kind == kind and (not idx or d.index == int(idx) or d.index is None):
            return d
    return _cpu_info()
