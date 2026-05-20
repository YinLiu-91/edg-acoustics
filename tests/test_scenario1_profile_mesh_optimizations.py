"""Profile-mesh optimization equivalence tests without large golden arrays."""

from __future__ import annotations

import numpy
import pytest
import torch

from scenario1_utils import build_scenario1_simulation, clone_bcvar, tensor_to_numpy


PROFILE_MESH = "scenario1_profile_lc0p20.msh"
RTOL = 1e-10
ATOL = 1e-10


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="profile-mesh optimization tests require CUDA"
)


def assert_close(name: str, actual, expected):
    numpy.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, err_msg=name)


def build_profile_mesh_sim(
    monkeypatch: pytest.MonkeyPatch,
    *,
    compact_flux: bool,
    merged_derivatives: bool,
):
    monkeypatch.setenv(
        "EDG_ACOUSTICS_COMPACT_FLUX_COEFFICIENTS", "1" if compact_flux else "0"
    )
    monkeypatch.setenv(
        "EDG_ACOUSTICS_MERGED_DERIVATIVES", "1" if merged_derivatives else "0"
    )
    monkeypatch.setenv("EDG_ACOUSTICS_PAIRED_INTERIOR_FLUX", "0")
    return build_scenario1_simulation(mesh_name=PROFILE_MESH)


def test_profile_mesh_compact_flux_matches_baseline_rhs(monkeypatch):
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

    for name, actual, expected in zip(
        ("rhs_p", "rhs_vx", "rhs_vy", "rhs_vz"),
        optimized_rhs[:4],
        baseline_rhs[:4],
    ):
        assert actual.dtype == torch.float64
        assert_close(name, tensor_to_numpy(actual), tensor_to_numpy(expected))

    for index, (actual_state, expected_state) in enumerate(
        zip(optimized_rhs[4], baseline_rhs[4])
    ):
        for key, actual in actual_state.items():
            expected = expected_state[key]
            if torch.is_tensor(actual):
                assert_close(
                    f"rhs_bc{index}_{key}",
                    tensor_to_numpy(actual),
                    tensor_to_numpy(expected),
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

    for name in ("P", "Vx", "Vy", "Vz", "prec"):
        assert_close(
            name,
            tensor_to_numpy(getattr(optimized, name)),
            tensor_to_numpy(getattr(baseline, name)),
        )
