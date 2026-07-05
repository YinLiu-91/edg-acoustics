from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_mma_tilelang_module():
    pytest.importorskip("tilelang")
    module_path = Path(__file__).resolve().parents[1] / "benchmarks" / "mma_tilelang_v6.py"
    spec = importlib.util.spec_from_file_location("local_mma_tilelang_v6", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_c500_deep_configs_target_large_register_derivative_tiles():
    module = load_mma_tilelang_module()

    configs = module.get_configs(
        105,
        35,
        shared_memory_limit=64 * 1024,
        sweep_level="c500-deep",
        warp_size=64,
    )
    names = {config.name for config in configs}

    assert "bm32_bn64_bk16_s1_t256_fullcol" in names
    assert any(name.startswith("bm112_") for name in names)
    assert any(name.startswith("bm112_") and name.endswith("_ss") for name in names)
    assert any("_bk12_" in name for name in names)
    assert not any(config.persistent for config in configs)


def test_c500_deep_persistent_configs_are_explicit_opt_in():
    module = load_mma_tilelang_module()

    configs = module.get_configs(
        105,
        35,
        shared_memory_limit=64 * 1024,
        sweep_level="c500-deep",
        warp_size=64,
        include_persistent=True,
    )

    assert any(config.persistent for config in configs)
    assert all(config.name.endswith("_persistent") for config in configs if config.persistent)


def test_c500_next_configs_focus_on_fullcol_derivative_winner():
    module = load_mma_tilelang_module()

    configs = module.get_configs(
        105,
        35,
        shared_memory_limit=64 * 1024,
        sweep_level="c500-next",
        warp_size=64,
    )
    names = {config.name for config in configs}

    assert "bm128_bn64_bk4_s0_t256_fullcol" in names
    assert "bm128_bn64_bk4_s1_t256_fullcol" in names
    assert "bm112_bn64_bk12_s1_t256_fullcol" in names
    assert "bm112_bn64_bk12_s0_t256_fullcol" in names
    assert all(config.policy == "fullcol" for config in configs)
    assert all(not config.use_shared_store for config in configs)
    assert all(not config.persistent for config in configs)
    assert any(config.block_N == 80 for config in configs)
    assert any(config.threads == 384 for config in configs)


def test_c500_bm128_configs_broaden_current_runtime_winner_search():
    module = load_mma_tilelang_module()

    configs = module.get_configs(
        105,
        35,
        shared_memory_limit=64 * 1024,
        sweep_level="c500-bm128",
        warp_size=64,
    )
    names = {config.name for config in configs}

    assert "bm128_bn64_bk4_s0_t256_fullcol" in names
    assert "bm128_bn64_bk4_s1_t256_fullcol" in names
    assert "bm112_bn64_bk4_s0_t256_fullcol" in names
    assert "bm96_bn64_bk4_s0_t256_fullcol" in names
    assert "bm128_bn64_bk4_s0_t256" in names
    assert all(config.block_K % 4 == 0 for config in configs)
    assert all(config.policy in ("fullcol", "square") for config in configs)
    assert all(not config.use_shared_store for config in configs)
    assert all(not config.persistent for config in configs)
    assert any(config.block_M == 64 for config in configs)
    assert any(config.block_M == 176 for config in configs)
    assert any(config.block_N == 160 for config in configs)
    assert any(config.block_K == 16 for config in configs)
    assert any(config.threads == 384 for config in configs)


def test_roofline_metrics_report_padding_and_peak_percentages():
    module = load_mma_tilelang_module()
    args = SimpleNamespace(peak_fp64_tflops=4.0, peak_bandwidth_tbps=1.8)
    config = module.named_config(112, 32, 16, 0, 128, "fullcol")

    row = module.make_row(
        105,
        35,
        1377572,
        torch_ms=5.316096,
        torch_q20=5.0,
        torch_q80=6.0,
        config=config,
        args=args,
    )

    assert row["logical_flops"] == 2 * 105 * 35 * 1377572
    assert row["padded_flops"] > row["logical_flops"]
    assert row["work_inflation"] > 1.0
    assert row["arithmetic_intensity"] > 0.0
    assert row["roofline_bound_tflops"] == pytest.approx(4.0)
    assert row["torch_fp64_peak_pct"] == pytest.approx(100.0 * row["torch_tflops"] / 4.0)
