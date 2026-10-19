import warnings

import pytest

from simchology.exceptions import HardwareInsufficientError, SimchologyHardwareWarning
from simchology.hardware import (
    ResourceRequest,
    check_resources,
    detect_hardware,
    emit,
    estimate_memory,
    estimate_runtime,
    resolve_device,
    training_runtime_kwargs,
)
from simchology.hardware import checks as checks_mod
from simchology.hardware.device import DeviceInfo
from simchology.hardware.estimate import GB
from simchology.hardware.models import parse_size_from_name


def test_report_and_device_resolution():
    report = detect_hardware()
    assert report.devices[-1].kind == "cpu"
    assert "Simchology hardware report" in str(report)
    assert resolve_device("cpu") == "cpu"
    assert resolve_device() in ("cuda", "mps", "xpu", "cpu")


def test_estimates_scale_sensibly():
    inf = estimate_memory("inference", 300_000_000, batch_size=8, seq_len=512, hidden_size=768, n_layers=12)
    ft = estimate_memory("full_finetune", 300_000_000, batch_size=8, seq_len=512, hidden_size=768, n_layers=12)
    lora = estimate_memory("lora", 300_000_000, batch_size=8, seq_len=512, hidden_size=768, n_layers=12)
    ckpt = estimate_memory("full_finetune", 300_000_000, batch_size=8, seq_len=512, hidden_size=768,
                           n_layers=12, gradient_checkpointing=True)
    assert inf.total_bytes < lora.total_bytes < ft.total_bytes
    assert ckpt.total_bytes < ft.total_bytes
    assert ft.breakdown["weights+gradients+optimizer"] == 16 * 300_000_000
    # the calibration anchor: mmBERT-base, 2,328 pairs x 7 epochs x 1024 tokens on MPS ~ 2.3 h
    hours = estimate_runtime("full_finetune", 307_000_000, 2328 * 7 * 1024, "mps") / 3600
    assert 1.5 < hours < 3.5


def test_parse_size_from_name():
    assert parse_size_from_name("qwen3:4b") == 4_000_000_000
    assert parse_size_from_name("jaredpalmer/kev-0.8b") == 800_000_000
    assert parse_size_from_name("bert-base") is None


def _fake_device(kind, avail_gb, total_gb=None):
    return DeviceInfo(kind, None, f"fake {kind}", int((total_gb or avail_gb) * GB), int(avail_gb * GB))


@pytest.mark.parametrize("kind,avail,level", [("cuda", 2, "insufficient"), ("cuda", 80, "ok"),
                                              ("cpu", 64, "slow")])
def test_verdicts(monkeypatch, kind, avail, level):
    monkeypatch.setattr(checks_mod, "device_info", lambda device="auto": _fake_device(kind, avail))
    req = ResourceRequest("full_finetune", n_params=300_000_000, hidden_size=768, n_layers=12,
                          seq_len=512, batch_size=2, n_items=10_000, description="fine-tuning a test model")
    v = check_resources(req, device=kind)
    assert v.level == level
    if level == "insufficient":
        assert v.suggestions and "GB" in v.message
        with pytest.raises(HardwareInsufficientError):
            emit(v, "error")
        with pytest.warns(SimchologyHardwareWarning):
            emit(v, "warn")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        emit(v, "off")


def test_unknown_model_size_is_not_an_error():
    v = check_resources(ResourceRequest("inference", model_id="not/a-real-model"), device="cpu")
    assert v.level == "unknown"


def test_training_kwargs_cpu():
    kw = training_runtime_kwargs("cpu")
    assert kw["use_cpu"] and not kw["dataloader_pin_memory"] and kw["dataloader_num_workers"] == 0
    assert not kw["bf16"] and not kw["fp16"]
