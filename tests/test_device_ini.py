"""Tests for device and dtype selection."""

from __future__ import annotations

import pytest
import torch

import edg_acoustics.device_ini as device_ini


def test_resolve_device_accepts_cpu(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_DEVICE", "cpu")
    assert device_ini._resolve_device() == torch.device("cpu")


def test_resolve_device_accepts_auto(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_DEVICE", "auto")
    resolved = device_ini._resolve_device()
    assert resolved.type in {"cpu", "cuda"}


def test_resolve_device_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_DEVICE", "tpu")
    with pytest.raises(ValueError, match="EDG_ACOUSTICS_DEVICE"):
        device_ini._resolve_device()


def test_resolve_device_rejects_unavailable_cuda(monkeypatch):
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this runner")
    monkeypatch.setenv("EDG_ACOUSTICS_DEVICE", "cuda")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        device_ini._resolve_device()


def test_global_dtype_is_fp64():
    assert device_ini.dtype == torch.float64
