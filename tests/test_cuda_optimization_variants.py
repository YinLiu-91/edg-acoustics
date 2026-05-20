"""Fast CUDA equivalence tests for recent optimization switches."""

from __future__ import annotations

import pytest
import torch

from scenario1_utils import (
    assert_rhs_close,
    assert_simulation_state_close,
    build_scenario1_simulation,
    clone_bcvar,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA optimization variant tests require CUDA"
)

RTOL = 1e-10
ATOL = 1e-10


def build_cuda_variant(
    monkeypatch: pytest.MonkeyPatch,
    *,
    compact_flux: bool = False,
    merged_derivatives: bool = False,
    fused_state_accumulation: bool = False,
    paired_interior_flux: bool = False,
):
    monkeypatch.setenv(
        "EDG_ACOUSTICS_COMPACT_FLUX_COEFFICIENTS", "1" if compact_flux else "0"
    )
    monkeypatch.setenv(
        "EDG_ACOUSTICS_MERGED_DERIVATIVES", "1" if merged_derivatives else "0"
    )
    monkeypatch.setenv(
        "EDG_ACOUSTICS_FUSED_STATE_ACCUMULATION",
        "1" if fused_state_accumulation else "0",
    )
    monkeypatch.setenv(
        "EDG_ACOUSTICS_PAIRED_INTERIOR_FLUX", "1" if paired_interior_flux else "0"
    )
    return build_scenario1_simulation(device="cuda")


@pytest.mark.parametrize(
    ("name", "options"),
    (
        pytest.param(
            "compact_flux",
            {"compact_flux": True},
            id="compact-flux",
        ),
        pytest.param(
            "merged_derivatives",
            {"merged_derivatives": True},
            id="merged-derivatives",
        ),
        pytest.param(
            "compact_flux_and_merged_derivatives",
            {"compact_flux": True, "merged_derivatives": True},
            id="compact-flux-and-merged-derivatives",
        ),
        pytest.param(
            "paired_interior_flux",
            {"compact_flux": True, "paired_interior_flux": True},
            id="paired-interior-flux",
        ),
    ),
)
def test_cuda_optimization_variant_rhs_matches_baseline(monkeypatch, name, options):
    baseline = build_cuda_variant(monkeypatch)
    baseline_rhs = baseline.RHS_operator(
        baseline.P,
        baseline.Vx,
        baseline.Vy,
        baseline.Vz,
        clone_bcvar(baseline.BC.BCvar),
    )

    optimized = build_cuda_variant(monkeypatch, **options)
    optimized_rhs = optimized.RHS_operator(
        optimized.P,
        optimized.Vx,
        optimized.Vy,
        optimized.Vz,
        clone_bcvar(optimized.BC.BCvar),
    )
    torch.cuda.synchronize()

    try:
        assert_rhs_close(optimized_rhs, baseline_rhs, rtol=RTOL, atol=ATOL)
    except AssertionError as exc:
        raise AssertionError(name) from exc


def test_cuda_fused_state_accumulation_matches_baseline(monkeypatch):
    baseline = build_cuda_variant(
        monkeypatch, compact_flux=True, merged_derivatives=True
    )
    baseline.time_integration(n_time_steps=2, progress=False, use_cuda_graph=True)

    optimized = build_cuda_variant(
        monkeypatch,
        compact_flux=True,
        merged_derivatives=True,
        fused_state_accumulation=True,
    )
    optimized.time_integration(n_time_steps=2, progress=False, use_cuda_graph=True)
    torch.cuda.synchronize()

    assert optimized._use_fused_state_accumulation
    assert_simulation_state_close(optimized, baseline, rtol=RTOL, atol=ATOL)
