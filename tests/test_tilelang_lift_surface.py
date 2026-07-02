"""Tests for the optional TileLang lift surface backend."""

from __future__ import annotations

import pytest
import torch

from scenario1_utils import (
    assert_simulation_state_close,
    build_scenario1_simulation,
    clone_bcvar,
)


def _tilelang_runtime_status() -> tuple[bool, str]:
    try:
        import tilelang  # noqa: F401
        import tilelang.language  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _is_maca_cuda_runtime() -> bool:
    if not torch.cuda.is_available():
        return False
    if getattr(torch.version, "maca", None) is not None:
        return True
    name = torch.cuda.get_device_name(0).lower()
    return "metax" in name or "maca" in name or "muxi" in name


TILELANG_AVAILABLE, TILELANG_UNAVAILABLE_REASON = _tilelang_runtime_status()


def test_tilelang_lift_disable_flag_is_reported_on_cpu(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_LIFT", "0")
    sim = build_scenario1_simulation(device="cpu")

    assert sim._tilelang_lift_mode == "0"
    assert not sim._use_tilelang_lift_surface
    assert sim._tilelang_lift_config == "bm48_bn64_bk16_s0_t256_fullcol"
    assert "disabled" in sim._tilelang_lift_fallback_reason


def test_tilelang_lift_rejects_invalid_mode(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_LIFT", "invalid")

    with pytest.raises(ValueError, match="EDG_ACOUSTICS_TILELANG_LIFT"):
        build_scenario1_simulation(device="cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TileLang lift runtime test requires CUDA",
)
def test_forced_tilelang_lift_falls_back_or_validates(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_LIFT", "1")
    sim = build_scenario1_simulation(device="cuda")
    sim.RHS_operator(
        sim.P,
        sim.Vx,
        sim.Vy,
        sim.Vz,
        clone_bcvar(sim.BC.BCvar),
    )
    torch.cuda.synchronize()

    if sim._use_tilelang_lift_surface:
        assert sim._tilelang_lift_correctness_checked
        assert sim._tilelang_lift_fallback_reason == ""
    else:
        assert sim._tilelang_lift_fallback_reason


@pytest.mark.skipif(
    not (torch.cuda.is_available() and TILELANG_AVAILABLE and _is_maca_cuda_runtime()),
    reason=f"requires CUDA + MACA/MetaX + TileLang: {TILELANG_UNAVAILABLE_REASON}",
)
def test_tilelang_lift_cuda_graph_path_matches_eager(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_LIFT", "1")
    eager = build_scenario1_simulation(device="cuda")
    graphed = build_scenario1_simulation(device="cuda")

    eager.time_integration(
        n_time_steps=1,
        progress=False,
        use_cuda_graph=False,
        record_receivers=False,
    )
    graphed.time_integration(
        n_time_steps=1,
        progress=False,
        use_cuda_graph=True,
        record_receivers=False,
    )
    torch.cuda.synchronize()

    assert_simulation_state_close(graphed, eager, rtol=1.0e-10, atol=1.0e-10)
    assert graphed._tilelang_lift_correctness_checked or graphed._tilelang_lift_fallback_reason
    assert graphed._tilelang_lift_graph_capture_supported in {True, False}
    if graphed._tilelang_lift_graph_capture_supported:
        assert graphed._use_tilelang_lift_surface
    else:
        assert not graphed._use_tilelang_lift_surface
        assert graphed._tilelang_lift_fallback_reason
