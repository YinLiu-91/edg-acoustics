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


def run_benchmark(*args: str):
    env = os.environ.copy()
    env["EDG_ACOUSTICS_DEVICE"] = "cpu"
    return subprocess.run(
        [sys.executable, str(BENCHMARK), *args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=120,
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
