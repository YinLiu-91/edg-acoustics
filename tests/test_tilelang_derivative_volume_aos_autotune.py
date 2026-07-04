from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_autotune_candidate_filtering_targets_supported_variants():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "tilelang_derivative_volume_aos_autotune.py"
    )
    spec = importlib.util.spec_from_file_location(
        "local_tilelang_derivative_volume_aos_autotune",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    candidates = module.available_autotune_candidates(
        variant_names=module._TARGET_VARIANTS,
        policy_names=module._TARGET_POLICIES,
        shared_memory_limit=65536,
        warp_size=64,
    )

    assert candidates
    assert module._TARGET_VARIANTS == ("copy_shared",)
    assert "merged3" not in module._TARGET_VARIANTS
    assert all(candidate.variant in module._TARGET_VARIANTS for candidate in candidates)
    assert all(candidate.policy in module._TARGET_POLICIES for candidate in candidates)
    assert "field_fragments" not in {candidate.variant for candidate in candidates}
    assert "field_pairs" not in {candidate.variant for candidate in candidates}
    assert "merged3" not in {candidate.variant for candidate in candidates}
    assert any(
        candidate.name == "bp16_be8_bn32_bk16_s0_t128_fullcol_copy_shared"
        for candidate in candidates
    )


def test_autotune_candidate_filtering_can_request_experimental_variants():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "tilelang_derivative_volume_aos_autotune.py"
    )
    spec = importlib.util.spec_from_file_location(
        "local_tilelang_derivative_volume_aos_autotune_experimental",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    candidates = module.available_autotune_candidates(
        variant_names=module._EXPERIMENTAL_VARIANTS,
        policy_names=module._TARGET_POLICIES,
        shared_memory_limit=65536,
        warp_size=64,
    )

    assert candidates
    assert {candidate.variant for candidate in candidates} == {
        "direct_epilogue",
        "merged3",
    }
