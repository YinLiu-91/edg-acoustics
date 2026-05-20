"""CPU/CUDA fp64 parity tests for scenario 1."""

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
    not torch.cuda.is_available(), reason="CPU/GPU parity tests require CUDA"
)

RTOL = 1e-9
ATOL = 1e-9


def synchronize():
    torch.cuda.synchronize()


def test_scenario1_rhs_matches_between_cpu_and_cuda_eager():
    cpu_sim = build_scenario1_simulation(device="cpu")
    cuda_sim = build_scenario1_simulation(device="cuda")

    assert cpu_sim.P.device.type == "cpu"
    assert cuda_sim.P.device.type == "cuda"

    cpu_rhs = cpu_sim.RHS_operator(
        cpu_sim.P,
        cpu_sim.Vx,
        cpu_sim.Vy,
        cpu_sim.Vz,
        clone_bcvar(cpu_sim.BC.BCvar),
    )
    cuda_rhs = cuda_sim.RHS_operator(
        cuda_sim.P,
        cuda_sim.Vx,
        cuda_sim.Vy,
        cuda_sim.Vz,
        clone_bcvar(cuda_sim.BC.BCvar),
    )
    synchronize()

    assert_rhs_close(cuda_rhs, cpu_rhs, rtol=RTOL, atol=ATOL)


def test_scenario1_short_integration_matches_between_cpu_and_cuda_eager():
    cpu_sim = build_scenario1_simulation(device="cpu")
    cuda_sim = build_scenario1_simulation(device="cuda")

    cpu_sim.time_integration(n_time_steps=2, progress=False)
    cuda_sim.time_integration(n_time_steps=2, progress=False)
    synchronize()

    assert_simulation_state_close(cuda_sim, cpu_sim, rtol=RTOL, atol=ATOL)


def test_scenario1_cuda_graph_matches_cuda_eager():
    eager = build_scenario1_simulation(device="cuda")
    graphed = build_scenario1_simulation(device="cuda")

    eager.time_integration(n_time_steps=2, progress=False, use_cuda_graph=False)
    graphed.time_integration(n_time_steps=2, progress=False, use_cuda_graph=True)
    synchronize()

    assert_simulation_state_close(graphed, eager, rtol=1e-10, atol=1e-10)
