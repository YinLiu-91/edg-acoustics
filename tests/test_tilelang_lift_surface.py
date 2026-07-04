"""Tests for the optional TileLang lift surface backend."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from scenario1_utils import (
    assert_rhs_close,
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


def test_tilelang_derivative_volume_aos_disable_flag_is_reported_on_cpu(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_DERIVATIVE_VOLUME_AOS", "0")
    sim = build_scenario1_simulation(device="cpu")

    assert sim._tilelang_derivative_volume_aos_mode == "0"
    assert not sim._use_tilelang_derivative_volume_aos
    assert "disabled" in sim._tilelang_derivative_volume_aos_fallback_reason


def test_tilelang_derivative_volume_aos_rejects_invalid_mode(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_DERIVATIVE_VOLUME_AOS", "invalid")

    with pytest.raises(ValueError, match="EDG_ACOUSTICS_TILELANG_DERIVATIVE_VOLUME_AOS"):
        build_scenario1_simulation(device="cpu")


def test_tilelang_derivative_gemm_disable_flag_is_reported_on_cpu(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_DERIVATIVE_GEMM", "0")
    sim = build_scenario1_simulation(device="cpu")

    assert sim._tilelang_derivative_gemm_mode == "0"
    assert not sim._use_tilelang_derivative_gemm
    assert sim._tilelang_derivative_gemm_config == "bm112_bn64_bk12_s1_t256_fullcol"
    assert "disabled" in sim._tilelang_derivative_gemm_fallback_reason


def test_tilelang_derivative_gemm_rejects_invalid_mode(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_DERIVATIVE_GEMM", "invalid")

    with pytest.raises(ValueError, match="EDG_ACOUSTICS_TILELANG_DERIVATIVE_GEMM"):
        build_scenario1_simulation(device="cpu")


def test_tilelang_derivative_volume_aos_variant_configs_are_exposed():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "edg_acoustics"
        / "tilelang_derivative_volume_aos.py"
    )
    spec = importlib.util.spec_from_file_location(
        "local_tilelang_derivative_volume_aos",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    default_config = module.get_config(module.TILELANG_DERIVATIVE_VOLUME_AOS_CONFIG_NAME)
    direct_config = module.get_config("bp16_be8_bn32_bk16_s0_t128_fullcol_direct")
    fieldfrag_config = module.get_config("bp16_be8_bn32_bk16_s0_t128_fullcol_fieldfrag")
    fieldpair_config = module.get_config("bp16_be8_bn32_bk16_s0_t128_fullcol_fieldpair")
    merged3_config = module.get_config("bp16_be8_bn32_bk16_s0_t128_fullcol_merged3")

    assert default_config.variant == "copy_shared"
    assert direct_config.variant == "direct_epilogue"
    assert fieldfrag_config.variant == "field_fragments"
    assert fieldpair_config.variant == "field_pairs"
    assert merged3_config.variant == "merged3"
    assert direct_config.explicit_shared_memory_bytes < default_config.explicit_shared_memory_bytes
    assert fieldfrag_config.explicit_shared_memory_bytes < default_config.explicit_shared_memory_bytes
    assert merged3_config.explicit_shared_memory_bytes == default_config.explicit_shared_memory_bytes
    assert "bp16_be8_bn32_bk16_s0_t128_fullcol_fieldpair" in module.available_config_names()
    assert "bp16_be8_bn32_bk16_s0_t128_fullcol_merged3" in module.available_config_names()


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
    not torch.cuda.is_available(),
    reason="TileLang derivative-volume AoS runtime test requires CUDA",
)
def test_forced_tilelang_derivative_volume_aos_falls_back_or_validates(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_AOS_STATE_LAYOUT", "1")
    monkeypatch.setenv("EDG_ACOUSTICS_AFFINE_METRIC_RHS", "1")
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_DERIVATIVE_VOLUME_AOS", "1")

    baseline = build_scenario1_simulation(device="cuda")
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_DERIVATIVE_VOLUME_AOS", "0")
    reference = build_scenario1_simulation(device="cuda")

    baseline_rhs = baseline.RHS_operator(
        baseline.P,
        baseline.Vx,
        baseline.Vy,
        baseline.Vz,
        clone_bcvar(baseline.BC.BCvar),
    )
    reference_rhs = reference.RHS_operator(
        reference.P,
        reference.Vx,
        reference.Vy,
        reference.Vz,
        clone_bcvar(reference.BC.BCvar),
    )
    torch.cuda.synchronize()

    assert_rhs_close(baseline_rhs, reference_rhs, rtol=1.0e-10, atol=1.0e-10)
    if baseline._use_tilelang_derivative_volume_aos:
        assert baseline._tilelang_derivative_volume_aos_correctness_checked
        assert baseline._tilelang_derivative_volume_aos_fallback_reason == ""
    else:
        assert baseline._tilelang_derivative_volume_aos_fallback_reason


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TileLang derivative GEMM runtime test requires CUDA",
)
def test_forced_tilelang_derivative_gemm_falls_back_or_validates(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_MERGED_DERIVATIVES", "1")
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_DERIVATIVE_GEMM", "1")
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_DERIVATIVE_VOLUME_AOS", "0")

    baseline = build_scenario1_simulation(device="cuda")
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_DERIVATIVE_GEMM", "0")
    reference = build_scenario1_simulation(device="cuda")

    baseline_rhs = baseline.RHS_operator(
        baseline.P,
        baseline.Vx,
        baseline.Vy,
        baseline.Vz,
        clone_bcvar(baseline.BC.BCvar),
    )
    reference_rhs = reference.RHS_operator(
        reference.P,
        reference.Vx,
        reference.Vy,
        reference.Vz,
        clone_bcvar(reference.BC.BCvar),
    )
    torch.cuda.synchronize()

    assert_rhs_close(baseline_rhs, reference_rhs, rtol=1.0e-10, atol=1.0e-10)
    if baseline._use_tilelang_derivative_gemm:
        assert baseline._tilelang_derivative_gemm_correctness_checked
        assert baseline._tilelang_derivative_gemm_fallback_reason == ""
    else:
        assert baseline._tilelang_derivative_gemm_fallback_reason


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
    assert graphed._tilelang_lift_segmented_graph_supported in {True, False, None}
    if (
        graphed._tilelang_lift_graph_capture_supported
        or graphed._tilelang_lift_segmented_graph_supported
    ):
        assert graphed._use_tilelang_lift_surface
    else:
        assert not graphed._use_tilelang_lift_surface
        assert graphed._tilelang_lift_fallback_reason


@pytest.mark.skipif(
    not (torch.cuda.is_available() and TILELANG_AVAILABLE and _is_maca_cuda_runtime()),
    reason=f"requires CUDA + MACA/MetaX + TileLang: {TILELANG_UNAVAILABLE_REASON}",
)
def test_tilelang_derivative_volume_aos_cuda_graph_path_matches_eager(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_AOS_STATE_LAYOUT", "1")
    monkeypatch.setenv("EDG_ACOUSTICS_AFFINE_METRIC_RHS", "1")
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_DERIVATIVE_VOLUME_AOS", "1")
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
    assert (
        graphed._tilelang_derivative_volume_aos_correctness_checked
        or graphed._tilelang_derivative_volume_aos_fallback_reason
    )


@pytest.mark.skipif(
    not (torch.cuda.is_available() and TILELANG_AVAILABLE and _is_maca_cuda_runtime()),
    reason=f"requires CUDA + MACA/MetaX + TileLang: {TILELANG_UNAVAILABLE_REASON}",
)
def test_tilelang_derivative_gemm_cuda_graph_path_matches_eager(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_MERGED_DERIVATIVES", "1")
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_DERIVATIVE_GEMM", "1")
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_DERIVATIVE_VOLUME_AOS", "0")
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
    assert (
        graphed._tilelang_derivative_gemm_correctness_checked
        or graphed._tilelang_derivative_gemm_fallback_reason
    )
    assert graphed._tilelang_derivative_gemm_graph_capture_supported in {True, False, None}


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="segmented CUDA graph regression requires CUDA",
)
def test_forced_segmented_cuda_graph_matches_eager_with_eager_lift(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_LIFT", "0")
    eager = build_scenario1_simulation(device="cuda")
    graphed = build_scenario1_simulation(device="cuda")

    def invoke_eager_lift():
        torch.mm(
            graphed.lift,
            graphed._flux_by_face,
            out=graphed._surface_by_node,
        )

    graphed._use_tilelang_lift_surface = True
    graphed._tilelang_lift_correctness_checked = True
    graphed._tilelang_lift_fallback_reason = ""
    graphed._tilelang_segmented_graph_mode = "1"
    graphed._validate_tilelang_lift_kernel = lambda: True
    graphed._invoke_tilelang_lift_kernel = invoke_eager_lift

    eager.time_integration(
        n_time_steps=2,
        progress=False,
        use_cuda_graph=False,
        record_receivers=False,
    )
    graphed.time_integration(
        n_time_steps=2,
        progress=False,
        use_cuda_graph=True,
        record_receivers=False,
    )
    torch.cuda.synchronize()

    assert_simulation_state_close(graphed, eager, rtol=1.0e-10, atol=1.0e-10)
    assert graphed.last_time_integration_cuda_graph_mode == "segmented_tilelang_lift"
    assert graphed._tilelang_lift_segmented_graph_supported is True


@pytest.mark.skipif(
    not (torch.cuda.is_available() and TILELANG_AVAILABLE and _is_maca_cuda_runtime()),
    reason=f"requires CUDA + MACA/MetaX + TileLang: {TILELANG_UNAVAILABLE_REASON}",
)
def test_tilelang_lift_forced_segmented_cuda_graph_matches_eager(monkeypatch):
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_LIFT", "1")
    monkeypatch.setenv("EDG_ACOUSTICS_TILELANG_SEGMENTED_CUDA_GRAPH", "1")
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
    assert graphed.last_time_integration_cuda_graph_mode in {
        "segmented_tilelang_lift",
        "full",
    }
