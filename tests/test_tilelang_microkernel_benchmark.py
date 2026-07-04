from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_tilelang_microkernel_configs_are_exposed():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "tilelang_derivative_volume_microkernel_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "local_tilelang_derivative_volume_microkernel_benchmark",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    default_config = module.get_config("bp16_be8_t128_qkn_dshared_unroll")
    qnk_config = module.get_config("bp16_be8_t128_qnk_dshared_unroll")
    global_config = module.get_config("bp16_be8_t128_qkn_dglobal_unroll")

    assert module.MANUAL_MICROKERNEL_STATUS == "experimental_retired"
    assert default_config.q_layout == "qkn"
    assert qnk_config.q_layout == "qnk"
    assert global_config.d_source == "global"
    assert (
        global_config.explicit_shared_memory_bytes
        < default_config.explicit_shared_memory_bytes
    )
    assert "bp16_be16_t256_qkn_dshared_unroll" in module.available_config_names()
