"""Compact fp64 golden tests for scenario 1."""

from __future__ import annotations

import numpy
import pytest
import torch

from scenario1_utils import GOLDEN_DIR, build_scenario1_simulation, clone_bcvar, tensor_to_numpy


GOLDEN_FILE = GOLDEN_DIR / "scenario1_fp64_short.npz"
RTOL = 1e-10
ATOL = 1e-10


pytestmark = pytest.mark.skipif(
    not GOLDEN_FILE.exists(), reason=f"Missing generated golden file: {GOLDEN_FILE}"
)


def assert_close(name: str, actual, expected):
    numpy.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, err_msg=name)


def test_scenario1_rhs_matches_fp64_golden():
    golden = numpy.load(GOLDEN_FILE)
    sim = build_scenario1_simulation()
    bcvar = clone_bcvar(sim.BC.BCvar)

    rhs_p, rhs_vx, rhs_vy, rhs_vz, bcvar = sim.RHS_operator(
        sim.P,
        sim.Vx,
        sim.Vy,
        sim.Vz,
        bcvar,
    )

    assert rhs_p.dtype == torch.float64
    assert_close("rhs_p", tensor_to_numpy(rhs_p), golden["rhs_p"])
    assert_close("rhs_vx", tensor_to_numpy(rhs_vx), golden["rhs_vx"])
    assert_close("rhs_vy", tensor_to_numpy(rhs_vy), golden["rhs_vy"])
    assert_close("rhs_vz", tensor_to_numpy(rhs_vz), golden["rhs_vz"])

    for index, state in enumerate(bcvar):
        for key, value in state.items():
            if torch.is_tensor(value):
                assert_close(
                    f"rhs_bc{index}_{key}",
                    tensor_to_numpy(value),
                    golden[f"rhs_bc{index}_{key}"],
                )


def test_scenario1_short_time_integration_matches_fp64_golden():
    assert_short_time_integration_matches_fp64_golden(use_cuda_graph=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires CUDA")
def test_scenario1_cuda_graph_matches_fp64_golden():
    assert_short_time_integration_matches_fp64_golden(use_cuda_graph=True)


def assert_short_time_integration_matches_fp64_golden(use_cuda_graph: bool):
    golden = numpy.load(GOLDEN_FILE)
    sim = build_scenario1_simulation()
    if use_cuda_graph and sim.P.device.type != "cuda":
        pytest.skip("Scenario is not running on CUDA")
    sim.time_integration(
        n_time_steps=int(golden["n_time_steps"]),
        use_cuda_graph=use_cuda_graph,
    )

    assert sim.P.dtype == torch.float64
    assert sim.prec.dtype == torch.float64
    assert_close("prec", tensor_to_numpy(sim.prec), golden["prec"])
    assert_close("final_p", tensor_to_numpy(sim.P), golden["final_p"])
    assert_close("final_vx", tensor_to_numpy(sim.Vx), golden["final_vx"])
    assert_close("final_vy", tensor_to_numpy(sim.Vy), golden["final_vy"])
    assert_close("final_vz", tensor_to_numpy(sim.Vz), golden["final_vz"])

    for index, state in enumerate(sim.BC.BCvar):
        for key, value in state.items():
            if torch.is_tensor(value):
                assert_close(
                    f"final_bc{index}_{key}",
                    tensor_to_numpy(value),
                    golden[f"final_bc{index}_{key}"],
                )
