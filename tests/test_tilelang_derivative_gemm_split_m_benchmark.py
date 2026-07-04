from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_split_m_benchmark_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "tilelang_derivative_gemm_split_m_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "local_tilelang_derivative_gemm_split_m_benchmark",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_m_configs_cover_derivative_rows_and_reduce_padding():
    module = load_split_m_benchmark_module()
    configs = {config.name: config for config in module.available_configs()}

    single = configs["single_bm128_bn64_bk4_s0_t256_fullcol"]
    split = configs["splitm96_9_bm96_16_bn64_bk4_s0_t256_fullcol"]

    assert single.total_rows == module.DERIVATIVE_M
    assert split.total_rows == module.DERIVATIVE_M

    n_columns = 181140
    assert module.padded_flops(split, n_columns) < module.padded_flops(single, n_columns)
    assert module.config_work_inflation(split, n_columns) < module.config_work_inflation(single, n_columns)


def test_resize_q_by_node_can_expand_and_truncate():
    module = load_split_m_benchmark_module()
    import torch

    base = torch.arange(35 * 8, dtype=torch.float64).reshape(35, 8)
    smaller = module.resize_q_by_node(base, 5)
    larger = module.resize_q_by_node(base, 19)

    assert tuple(smaller.shape) == (35, 5)
    assert torch.equal(smaller, base[:, :5])

    assert tuple(larger.shape) == (35, 19)
    assert torch.equal(larger[:, :8], base)
    assert torch.equal(larger[:, 8:16], base)
    assert torch.equal(larger[:, 16:], base[:, :3])


def test_unknown_config_error_lists_known_configs():
    module = load_split_m_benchmark_module()

    with pytest.raises(ValueError, match="unknown split-M config"):
        module.config_by_name("missing")


def test_column_resize_mode_labels_scenario_truncated_and_repeated():
    module = load_split_m_benchmark_module()

    assert module.column_resize_mode(8, 8) == "scenario"
    assert module.column_resize_mode(8, 5) == "truncated"
    assert module.column_resize_mode(8, 19) == "repeated"
