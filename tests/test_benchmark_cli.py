"""Smoke tests for the scenario1 benchmark CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "benchmarks" / "scenario1_benchmark.py"


def run_benchmark(*args: str, timeout: int = 120):
    env = os.environ.copy()
    env["EDG_ACOUSTICS_DEVICE"] = "cpu"
    return subprocess.run(
        [sys.executable, str(BENCHMARK), *args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    ).stdout


def assert_common_output(output: str):
    for field in (
        "mode=",
        "steps=",
        "mesh_name=scenario1_coarser.msh",
        "N_tets=",
        "Np=",
        "Nfp=",
        "dtype=torch.float64",
        "device=",
        "optimizations=",
        "interior_face_order=",
        "ordered_aos_variant=",
        "ordered_aos_state_load_mode=",
    ):
        assert field in output


def test_scenario1_benchmark_cli_reports_cpu_metadata():
    output = run_benchmark(
        "--device",
        "cpu",
        "--steps",
        "1",
        "--cpu-threads",
        "1",
        "--no-record-receivers",
    )

    assert_common_output(output)
    assert "device=cpu" in output
    assert "cuda_graph=False" in output
    assert "cpu_threads=" in output
    assert "interior_face_order=tile_plus_packed" in output
    assert "face_order_enabled=0" in output
    assert "ordered_aos_variant=base" in output
    assert "ordered_aos_state_load_mode=scalar" in output
    assert "aos_state_layout:0" in output
    assert "aos_volume_vector_loads:0" in output


def test_scenario1_benchmark_cli_reports_real_case_metadata():
    output = run_benchmark(
        "--device",
        "cpu",
        "--mesh-name",
        "scenario1_coarser.msh",
        "--real-case-total-time",
        "1e-5",
        "--cpu-threads",
        "1",
        "--no-record-receivers",
    )

    assert_common_output(output)
    assert "mode=real_case" in output
    assert "total_time=" in output
    assert "elapsed_s=" in output
    assert "ms_per_step=" in output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph smoke requires CUDA")
def test_scenario1_benchmark_cli_reports_cuda_graph_metadata():
    output = run_benchmark(
        "--device",
        "cuda",
        "--steps",
        "1",
        "--cuda-graph",
        "--no-record-receivers",
    )

    assert_common_output(output)
    assert "device=cuda" in output
    assert "cuda_graph=True" in output
    assert "cuda_name=" in output


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="default AoS metadata smoke requires CUDA",
)
def test_scenario1_benchmark_cli_reports_default_aos_metadata():
    output = run_benchmark(
        "--device",
        "cuda",
        "--mesh-name",
        "scenario1_profile_lc0p20.msh",
        "--steps",
        "1",
        "--cuda-graph",
        "--no-record-receivers",
        timeout=300,
    )

    assert "mesh_name=scenario1_profile_lc0p20.msh" in output
    assert "interior_face_order=tile_plus_packed" in output
    assert "face_order_tile_size=128" in output
    assert "face_order_block_size=128" in output
    assert "face_order_enabled=1" in output
    assert "face_order_storage=tile_local_u8" in output
    assert "ordered_aos_variant=vec4_scheduled" in output
    assert "ordered_aos_state_load_mode=vec4_scheduled" in output
    assert "aos_state_layout:1" in output
    assert "aos_volume_vector_loads:1" in output
    assert "ordered_aos_state_vec4:1" in output
    assert "affine_metric_rhs:1" in output
