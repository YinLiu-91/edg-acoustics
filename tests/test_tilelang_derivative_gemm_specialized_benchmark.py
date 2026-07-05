from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_specialized_benchmark_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "tilelang_derivative_gemm_specialized_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "local_tilelang_derivative_gemm_specialized_benchmark",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_specialized_configs_expose_expected_candidates():
    module = load_specialized_benchmark_module()
    names = [config.name for config in module.available_configs()]

    assert "single_bm128_bn64_bk4_s0_t256_fullcol" in names
    assert "fullk_bm112_bn64_bk36_s0_t256_fullcol" in names
    assert "fulla_staged_bm112_bn64_bk4_s0_t256_fullcol" in names
    assert "fullk_group2_bm112_bn64_bk36_s0_t256_fullcol" in names
    assert "tri35_bm48_bn64_bk36_s0_t256_fullcol_3launch" in names


def test_fullk_112_reduces_padding_against_current_single():
    module = load_specialized_benchmark_module()
    configs = {config.name: config for config in module.available_configs()}

    baseline = configs["single_bm128_bn64_bk4_s0_t256_fullcol"]
    fullk_112 = configs["fullk_bm112_bn64_bk36_s0_t256_fullcol"]
    n_columns = 181140

    assert module.shared_memory_bytes(fullk_112) <= 65536
    assert module.work_inflation(fullk_112, n_columns) < module.work_inflation(
        baseline,
        n_columns,
    )


def test_tri35_reports_three_b_reads_and_higher_padding():
    module = load_specialized_benchmark_module()
    configs = {config.name: config for config in module.available_configs()}

    baseline = configs["single_bm128_bn64_bk4_s0_t256_fullcol"]
    tri35 = configs["tri35_bm48_bn64_bk36_s0_t256_fullcol_3launch"]
    n_columns = 181140

    assert tri35.kernel_launches_per_call == 3
    assert tri35.b_read_multiplier == 3
    assert module.work_inflation(tri35, n_columns) > module.work_inflation(
        baseline,
        n_columns,
    )


def test_group2_reduces_estimated_a_reload_bytes():
    module = load_specialized_benchmark_module()
    configs = {config.name: config for config in module.available_configs()}

    fullk_112 = configs["fullk_bm112_bn64_bk36_s0_t256_fullcol"]
    group2 = configs["fullk_group2_bm112_bn64_bk36_s0_t256_fullcol"]
    n_columns = 1377572

    assert module.shared_memory_bytes(group2) <= 65536
    assert module.estimated_tile_global_bytes(group2, n_columns) < module.estimated_tile_global_bytes(
        fullk_112,
        n_columns,
    )
