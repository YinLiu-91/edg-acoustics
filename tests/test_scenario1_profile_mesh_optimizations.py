"""Profile-mesh optimization equivalence tests without large golden arrays."""

from __future__ import annotations

import os

import pytest
import torch

from scenario1_utils import (
    assert_rhs_close,
    assert_simulation_state_close,
    build_scenario1_simulation,
    clone_bcvar,
)


PROFILE_MESH = "scenario1_profile_lc0p20.msh"
RTOL = 1e-10
ATOL = 1e-10


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="profile-mesh optimization tests require CUDA"
)


def build_profile_mesh_sim(
    monkeypatch: pytest.MonkeyPatch,
    *,
    compact_flux: bool,
    merged_derivatives: bool,
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
    return build_scenario1_simulation(mesh_name=PROFILE_MESH, device="cuda")


def test_profile_mesh_compact_flux_and_merged_derivatives_match_baseline_rhs(
    monkeypatch,
):
    baseline = build_profile_mesh_sim(
        monkeypatch, compact_flux=False, merged_derivatives=False
    )
    baseline_bcvar = clone_bcvar(baseline.BC.BCvar)
    baseline_rhs = baseline.RHS_operator(
        baseline.P, baseline.Vx, baseline.Vy, baseline.Vz, baseline_bcvar
    )

    optimized = build_profile_mesh_sim(
        monkeypatch, compact_flux=True, merged_derivatives=True
    )
    assert optimized._use_compact_flux_coefficients
    assert optimized._use_merged_derivatives
    optimized_bcvar = clone_bcvar(optimized.BC.BCvar)
    optimized_rhs = optimized.RHS_operator(
        optimized.P, optimized.Vx, optimized.Vy, optimized.Vz, optimized_bcvar
    )

    assert_rhs_close(optimized_rhs, baseline_rhs, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(
    os.environ.get("EDG_ACOUSTICS_RUN_SLOW_TESTS") != "1",
    reason="lc0p20 short integration is a slow opt-in regression test",
)
def test_profile_mesh_compact_flux_matches_baseline_short_integration(monkeypatch):
    baseline = build_profile_mesh_sim(
        monkeypatch, compact_flux=False, merged_derivatives=False
    )
    baseline.time_integration(n_time_steps=2, progress=False, use_cuda_graph=True)

    optimized = build_profile_mesh_sim(
        monkeypatch, compact_flux=True, merged_derivatives=True
    )
    optimized.time_integration(n_time_steps=2, progress=False, use_cuda_graph=True)

    assert_simulation_state_close(optimized, baseline, rtol=RTOL, atol=ATOL)
