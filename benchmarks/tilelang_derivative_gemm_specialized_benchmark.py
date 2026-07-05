"""Benchmark specialized derivative GEMM candidates on scenario1 inputs.

This isolates the EDG merged derivative hot path:

    D_merged[105, 35] @ q_by_node[35, N]

The goal is to test C500-specific ideas that are too experimental for the
runtime path:

- pad K to 36 and issue a single full-K TileLang GEMM
- load the full A matrix into shared once, then stage K tiles from shared
- amortize one full-A load across two N tiles
- split D_merged into Dr/Ds/Dt and launch three 35x35 GEMMs
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BENCHMARKS_DIR))

import tilelang_derivative_gemm_split_m_benchmark as split_bench  # noqa: E402


DERIVATIVE_M = split_bench.DERIVATIVE_M
DERIVATIVE_K = split_bench.DERIVATIVE_K
PADDED_K = 36
TRI_ROWS = 35

T = None
_JITTED_FP64_GENERIC_MATMUL = None
_JITTED_FP64_FULLA_STAGED = None
_JITTED_FP64_FULLK_GROUP = None
_GENERIC_KERNEL_CACHE: dict[tuple[int, int, int, "SpecializedConfig"], object] = {}
_FULLA_STAGED_KERNEL_CACHE: dict[tuple[int, "SpecializedConfig"], object] = {}
_FULLK_GROUP_KERNEL_CACHE: dict[tuple[int, "SpecializedConfig"], object] = {}
_ACCEPTS_SKIP_VALIDATION: dict[int, bool] = {}


@dataclass(frozen=True)
class SpecializedConfig:
    name: str
    kind: str
    block_M: int
    block_N: int
    block_K: int
    num_stages: int
    threads: int
    policy: str = "fullcol"
    enable_swizzle: bool = True
    pad_k_to: int = DERIVATIVE_K
    group_n: int = 1
    row_splits: tuple[int, ...] = (DERIVATIVE_M,)

    @property
    def kernel_launches_per_call(self) -> int:
        if self.kind == "tri35":
            return len(self.row_splits)
        return 1

    @property
    def b_read_multiplier(self) -> int:
        if self.kind == "tri35":
            return len(self.row_splits)
        return 1


@dataclass
class CandidateInputs:
    base: split_bench.KernelInputs
    d_merged_k36: torch.Tensor
    q_by_node_k36: torch.Tensor
    tri35: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    tri35_k36: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def n_columns(self) -> int:
        return self.base.n_columns


def _fp64_generic_matmul_tn(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_stages,
    threads,
    enable_swizzle,
    policy,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float64),
        B: T.Tensor((K, N), T.float64),
        C: T.Tensor((M, N), T.float64),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), T.float64)
            B_shared = T.alloc_shared((block_N, block_K), T.float64)
            C_local = T.alloc_fragment((block_M, block_N), T.float64)

            T.use_swizzle(panel_size=10, enable=enable_swizzle)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                for j, kk in T.Parallel(block_N, block_K):
                    k_idx = ko * block_K + kk
                    n_idx = bx * block_N + j
                    B_shared[j, kk] = T.if_then_else(
                        (k_idx < K) & (n_idx < N),
                        B[k_idx, n_idx],
                        T.float64(0.0),
                    )
                T.gemm(A_shared, B_shared, C_local, transpose_B=True, policy=policy)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _fp64_fulla_staged_matmul_tn(
    M,
    N,
    full_K,
    block_M,
    block_N,
    block_K,
    threads,
    enable_swizzle,
    policy,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, full_K), T.float64),
        B: T.Tensor((full_K, N), T.float64),
        C: T.Tensor((M, N), T.float64),
    ):
        with T.Kernel(T.ceildiv(N, block_N), 1, threads=threads) as (bx, _):
            A_full_shared = T.alloc_shared((block_M, full_K), T.float64)
            A_tile_shared = T.alloc_shared((block_M, block_K), T.float64)
            B_shared = T.alloc_shared((block_N, block_K), T.float64)
            C_local = T.alloc_fragment((block_M, block_N), T.float64)

            T.use_swizzle(panel_size=10, enable=enable_swizzle)
            for i, kk in T.Parallel(block_M, full_K):
                A_full_shared[i, kk] = T.if_then_else(
                    (i < M) & (kk < full_K),
                    A[i, kk],
                    T.float64(0.0),
                )
            T.clear(C_local)
            for ko in T.serial(0, T.ceildiv(full_K, block_K)):
                for i, kk in T.Parallel(block_M, block_K):
                    k_idx = ko * block_K + kk
                    A_tile_shared[i, kk] = T.if_then_else(
                        k_idx < full_K,
                        A_full_shared[i, k_idx],
                        T.float64(0.0),
                    )
                for j, kk in T.Parallel(block_N, block_K):
                    k_idx = ko * block_K + kk
                    n_idx = bx * block_N + j
                    B_shared[j, kk] = T.if_then_else(
                        (k_idx < full_K) & (n_idx < N),
                        B[k_idx, n_idx],
                        T.float64(0.0),
                    )
                T.gemm(
                    A_tile_shared,
                    B_shared,
                    C_local,
                    transpose_B=True,
                    policy=policy,
                )
            T.copy(C_local, C[0, bx * block_N])

    return main


def _fp64_fullk_group_matmul_tn(
    M,
    N,
    full_K,
    block_M,
    block_N,
    threads,
    enable_swizzle,
    policy,
    group_n,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, full_K), T.float64),
        B: T.Tensor((full_K, N), T.float64),
        C: T.Tensor((M, N), T.float64),
    ):
        with T.Kernel(T.ceildiv(T.ceildiv(N, block_N), group_n), 1, threads=threads) as (
            bx,
            _,
        ):
            A_shared = T.alloc_shared((block_M, full_K), T.float64)
            B_shared = T.alloc_shared((block_N, full_K), T.float64)
            C_local = T.alloc_fragment((block_M, block_N), T.float64)

            T.use_swizzle(panel_size=10, enable=enable_swizzle)
            for i, kk in T.Parallel(block_M, full_K):
                A_shared[i, kk] = T.if_then_else(
                    (i < M) & (kk < full_K),
                    A[i, kk],
                    T.float64(0.0),
                )
            for group_index in T.serial(0, group_n):
                T.clear(C_local)
                for j, kk in T.Parallel(block_N, full_K):
                    n_idx = (bx * group_n + group_index) * block_N + j
                    B_shared[j, kk] = T.if_then_else(
                        n_idx < N,
                        B[kk, n_idx],
                        T.float64(0.0),
                    )
                T.gemm(A_shared, B_shared, C_local, transpose_B=True, policy=policy)
                T.copy(
                    C_local,
                    C[0, (bx * group_n + group_index) * block_N],
                )

    return main


def _ensure_tilelang():
    global T
    if T is not None:
        return
    import tilelang.language as tilelang_language

    T = tilelang_language


def _jitted_fp64_generic_matmul():
    global _JITTED_FP64_GENERIC_MATMUL
    if _JITTED_FP64_GENERIC_MATMUL is not None:
        return _JITTED_FP64_GENERIC_MATMUL

    import tilelang

    _ensure_tilelang()
    _JITTED_FP64_GENERIC_MATMUL = tilelang.jit(_fp64_generic_matmul_tn)
    return _JITTED_FP64_GENERIC_MATMUL


def _jitted_fp64_fulla_staged():
    global _JITTED_FP64_FULLA_STAGED
    if _JITTED_FP64_FULLA_STAGED is not None:
        return _JITTED_FP64_FULLA_STAGED

    import tilelang

    _ensure_tilelang()
    _JITTED_FP64_FULLA_STAGED = tilelang.jit(_fp64_fulla_staged_matmul_tn)
    return _JITTED_FP64_FULLA_STAGED


def _jitted_fp64_fullk_group():
    global _JITTED_FP64_FULLK_GROUP
    if _JITTED_FP64_FULLK_GROUP is not None:
        return _JITTED_FP64_FULLK_GROUP

    import tilelang

    _ensure_tilelang()
    _JITTED_FP64_FULLK_GROUP = tilelang.jit(_fp64_fullk_group_matmul_tn)
    return _JITTED_FP64_FULLK_GROUP


def policy_value(policy: str):
    _ensure_tilelang()
    if policy == "square":
        return T.GemmWarpPolicy.Square
    if policy == "fullrow":
        return T.GemmWarpPolicy.FullRow
    if policy == "fullcol":
        return T.GemmWarpPolicy.FullCol
    raise ValueError(f"unknown GEMM warp policy: {policy}")


def build_generic_kernel(m: int, n: int, k: int, config: SpecializedConfig):
    cache_key = (m, n, k, config)
    kernel = _GENERIC_KERNEL_CACHE.get(cache_key)
    if kernel is not None:
        return kernel

    jitted = _jitted_fp64_generic_matmul()
    kernel = jitted(
        m,
        n,
        k,
        config.block_M,
        config.block_N,
        config.block_K,
        config.num_stages,
        config.threads,
        config.enable_swizzle,
        policy_value(config.policy),
    )
    _GENERIC_KERNEL_CACHE[cache_key] = kernel
    return kernel


def build_fulla_staged_kernel(n: int, config: SpecializedConfig):
    cache_key = (n, config)
    kernel = _FULLA_STAGED_KERNEL_CACHE.get(cache_key)
    if kernel is not None:
        return kernel

    jitted = _jitted_fp64_fulla_staged()
    kernel = jitted(
        DERIVATIVE_M,
        n,
        config.pad_k_to,
        config.block_M,
        config.block_N,
        config.block_K,
        config.threads,
        config.enable_swizzle,
        policy_value(config.policy),
    )
    _FULLA_STAGED_KERNEL_CACHE[cache_key] = kernel
    return kernel


def build_fullk_group_kernel(n: int, config: SpecializedConfig):
    cache_key = (n, config)
    kernel = _FULLK_GROUP_KERNEL_CACHE.get(cache_key)
    if kernel is not None:
        return kernel

    jitted = _jitted_fp64_fullk_group()
    kernel = jitted(
        DERIVATIVE_M,
        n,
        config.pad_k_to,
        config.block_M,
        config.block_N,
        config.threads,
        config.enable_swizzle,
        policy_value(config.policy),
        config.group_n,
    )
    _FULLK_GROUP_KERNEL_CACHE[cache_key] = kernel
    return kernel


def call_tilelang_kernel(kernel, args) -> None:
    kernel_id = id(kernel)
    accepts = _ACCEPTS_SKIP_VALIDATION.get(kernel_id)
    if accepts is not False:
        try:
            kernel(*args, skip_tensor_validation=True)
            _ACCEPTS_SKIP_VALIDATION[kernel_id] = True
            return
        except TypeError as exc:
            if "skip_tensor_validation" not in str(exc):
                raise
            _ACCEPTS_SKIP_VALIDATION[kernel_id] = False
    kernel(*args)


def current_config() -> SpecializedConfig:
    return SpecializedConfig(
        "single_bm128_bn64_bk4_s0_t256_fullcol",
        "generic",
        block_M=128,
        block_N=64,
        block_K=4,
        num_stages=0,
        threads=256,
    )


def available_configs() -> tuple[SpecializedConfig, ...]:
    configs = (
        current_config(),
        SpecializedConfig(
            "fullk_bm112_bn64_bk36_s0_t256_fullcol",
            "generic",
            block_M=112,
            block_N=64,
            block_K=36,
            num_stages=0,
            threads=256,
            pad_k_to=PADDED_K,
        ),
        SpecializedConfig(
            "fullk_bm128_bn64_bk36_s0_t256_fullcol",
            "generic",
            block_M=128,
            block_N=64,
            block_K=36,
            num_stages=0,
            threads=256,
            pad_k_to=PADDED_K,
        ),
        SpecializedConfig(
            "fulla_staged_bm112_bn64_bk4_s0_t256_fullcol",
            "fulla_staged",
            block_M=112,
            block_N=64,
            block_K=4,
            num_stages=0,
            threads=256,
            pad_k_to=PADDED_K,
        ),
        SpecializedConfig(
            "fullk_group2_bm112_bn64_bk36_s0_t256_fullcol",
            "fullk_group",
            block_M=112,
            block_N=64,
            block_K=36,
            num_stages=0,
            threads=256,
            pad_k_to=PADDED_K,
            group_n=2,
        ),
        SpecializedConfig(
            "tri35_bm48_bn64_bk36_s0_t256_fullcol_3launch",
            "tri35",
            block_M=48,
            block_N=64,
            block_K=36,
            num_stages=0,
            threads=256,
            pad_k_to=PADDED_K,
            row_splits=(TRI_ROWS, TRI_ROWS, TRI_ROWS),
        ),
        SpecializedConfig(
            "tri35_bm48_bn64_bk4_s0_t256_fullcol_3launch",
            "tri35",
            block_M=48,
            block_N=64,
            block_K=4,
            num_stages=0,
            threads=256,
            row_splits=(TRI_ROWS, TRI_ROWS, TRI_ROWS),
        ),
    )
    for config in configs:
        validate_config(config)
    return configs


def config_by_name(name: str) -> SpecializedConfig:
    for config in available_configs():
        if config.name == name:
            return config
    known = ", ".join(config.name for config in available_configs())
    raise ValueError(f"unknown specialized config {name!r}; known configs: {known}")


def validate_config(config: SpecializedConfig) -> None:
    if sum(config.row_splits) != DERIVATIVE_M:
        raise ValueError(
            f"{config.name} covers {sum(config.row_splits)} rows, expected {DERIVATIVE_M}"
        )
    if config.block_M % 16 != 0:
        raise ValueError(
            f"{config.name} requires block_M multiple of 16, got {config.block_M}"
        )
    if config.block_N % 16 != 0:
        raise ValueError(
            f"{config.name} requires block_N multiple of 16, got {config.block_N}"
        )
    if config.block_K % 4 != 0:
        raise ValueError(
            f"{config.name} requires block_K multiple of 4, got {config.block_K}"
        )
    if config.threads % 64 != 0:
        raise ValueError(
            f"{config.name} requires threads multiple of 64, got {config.threads}"
        )
    if config.pad_k_to not in {DERIVATIVE_K, PADDED_K}:
        raise ValueError(
            f"{config.name} requires K padding target 35 or 36, got {config.pad_k_to}"
        )
    if config.kind not in {"generic", "fulla_staged", "fullk_group", "tri35"}:
        raise ValueError(f"{config.name} uses unknown kind {config.kind!r}")
    if config.kind == "fullk_group" and config.group_n < 2:
        raise ValueError(f"{config.name} requires group_n >= 2")


def pad_k_for_a(a: torch.Tensor, target_k: int) -> torch.Tensor:
    if a.shape[1] == target_k:
        return a.contiguous()
    if a.shape[1] > target_k:
        raise ValueError(f"cannot shrink A from K={a.shape[1]} to {target_k}")
    zeros = torch.zeros(
        (a.shape[0], target_k - a.shape[1]),
        dtype=a.dtype,
        device=a.device,
    )
    return torch.cat((a, zeros), dim=1).contiguous()


def pad_k_for_b(b: torch.Tensor, target_k: int) -> torch.Tensor:
    if b.shape[0] == target_k:
        return b.contiguous()
    if b.shape[0] > target_k:
        raise ValueError(f"cannot shrink B from K={b.shape[0]} to {target_k}")
    zeros = torch.zeros(
        (target_k - b.shape[0], b.shape[1]),
        dtype=b.dtype,
        device=b.device,
    )
    return torch.cat((b, zeros), dim=0).contiguous()


def make_candidate_inputs(
    base_inputs: split_bench.KernelInputs,
    n_columns: int,
) -> CandidateInputs:
    q_by_node = split_bench.resize_q_by_node(base_inputs.q_by_node, n_columns)
    base = split_bench.KernelInputs(
        d_merged=base_inputs.d_merged,
        q_by_node=q_by_node,
    )
    d_merged_k36 = pad_k_for_a(base.d_merged, PADDED_K)
    q_by_node_k36 = pad_k_for_b(base.q_by_node, PADDED_K)
    tri35 = (
        base.d_merged[0:TRI_ROWS].contiguous(),
        base.d_merged[TRI_ROWS : 2 * TRI_ROWS].contiguous(),
        base.d_merged[2 * TRI_ROWS : 3 * TRI_ROWS].contiguous(),
    )
    tri35_k36 = tuple(pad_k_for_a(part, PADDED_K) for part in tri35)
    return CandidateInputs(
        base=base,
        d_merged_k36=d_merged_k36,
        q_by_node_k36=q_by_node_k36,
        tri35=tri35,
        tri35_k36=tri35_k36,
    )


def effective_k(config: SpecializedConfig) -> int:
    return config.pad_k_to


def logical_flops(n_columns: int) -> int:
    return 2 * DERIVATIVE_M * DERIVATIVE_K * n_columns


def padded_flops(config: SpecializedConfig, n_columns: int) -> int:
    total = 0
    k_total = effective_k(config)
    for rows in config.row_splits:
        total += (
            2
            * split_bench.ceildiv_int(rows, config.block_M)
            * config.block_M
            * split_bench.ceildiv_int(n_columns, config.block_N)
            * config.block_N
            * split_bench.ceildiv_int(k_total, config.block_K)
            * config.block_K
        )
    return total


def generic_estimated_tile_bytes(
    rows: int,
    k_total: int,
    n_columns: int,
    config: SpecializedConfig,
) -> int:
    m_tiles = split_bench.ceildiv_int(rows, config.block_M)
    n_tiles = split_bench.ceildiv_int(n_columns, config.block_N)
    k_tiles = split_bench.ceildiv_int(k_total, config.block_K)
    a_bytes = m_tiles * n_tiles * k_tiles * config.block_M * config.block_K * 8
    b_bytes = m_tiles * n_tiles * k_tiles * config.block_N * config.block_K * 8
    c_store_bytes = m_tiles * n_tiles * config.block_M * config.block_N * 8
    return a_bytes + b_bytes + c_store_bytes


def estimated_tile_global_bytes(config: SpecializedConfig, n_columns: int) -> int:
    if config.kind == "generic":
        return generic_estimated_tile_bytes(
            DERIVATIVE_M,
            effective_k(config),
            n_columns,
            config,
        )
    if config.kind == "tri35":
        return sum(
            generic_estimated_tile_bytes(rows, effective_k(config), n_columns, config)
            for rows in config.row_splits
        )
    if config.kind == "fulla_staged":
        n_tiles = split_bench.ceildiv_int(n_columns, config.block_N)
        k_tiles = split_bench.ceildiv_int(effective_k(config), config.block_K)
        a_bytes = n_tiles * config.block_M * effective_k(config) * 8
        b_bytes = n_tiles * k_tiles * config.block_N * config.block_K * 8
        c_store_bytes = n_tiles * config.block_M * config.block_N * 8
        return a_bytes + b_bytes + c_store_bytes
    if config.kind == "fullk_group":
        n_tiles = split_bench.ceildiv_int(n_columns, config.block_N)
        n_groups = split_bench.ceildiv_int(n_tiles, config.group_n)
        a_bytes = n_groups * config.block_M * effective_k(config) * 8
        b_bytes = n_tiles * config.block_N * effective_k(config) * 8
        c_store_bytes = n_tiles * config.block_M * config.block_N * 8
        return a_bytes + b_bytes + c_store_bytes
    raise ValueError(f"unsupported config kind {config.kind!r}")


def shared_memory_bytes(config: SpecializedConfig) -> int:
    stage_factor = max(config.num_stages, 1)
    if config.kind in {"generic", "tri35"}:
        elements = config.block_M * config.block_K + config.block_N * config.block_K
        return elements * 8 * stage_factor
    if config.kind == "fulla_staged":
        elements = (
            config.block_M * effective_k(config)
            + config.block_M * config.block_K
            + config.block_N * config.block_K
        )
        return elements * 8
    if config.kind == "fullk_group":
        elements = (
            config.block_M * effective_k(config)
            + config.block_N * effective_k(config)
        )
        return elements * 8
    raise ValueError(f"unsupported config kind {config.kind!r}")


def work_inflation(config: SpecializedConfig, n_columns: int) -> float:
    return padded_flops(config, n_columns) / logical_flops(n_columns)


def print_config_summary(config: SpecializedConfig, n_columns: int) -> None:
    print(f"config={config.name}")
    print(f"kind={config.kind}")
    print(f"kernel_launches_per_call={config.kernel_launches_per_call}")
    print(f"b_read_multiplier={config.b_read_multiplier}")
    print(f"pad_k_to={config.pad_k_to}")
    print(f"block_M={config.block_M}")
    print(f"block_N={config.block_N}")
    print(f"block_K={config.block_K}")
    print(f"num_stages={config.num_stages}")
    print(f"threads={config.threads}")
    print(f"policy={config.policy}")
    print(f"group_n={config.group_n}")
    print(f"row_splits={','.join(str(row) for row in config.row_splits)}")
    print(f"shared_kib={shared_memory_bytes(config) / 1024:.1f}")
    print(f"logical_flops={logical_flops(n_columns)}")
    print(f"padded_flops={padded_flops(config, n_columns)}")
    print(f"work_inflation={work_inflation(config, n_columns):.6f}")
    print(
        f"estimated_tile_global_bytes="
        f"{estimated_tile_global_bytes(config, n_columns)}"
    )


def report_perf(
    prefix: str,
    ms: float,
    config: SpecializedConfig | None,
    n_columns: int,
    *,
    peak_fp64_tflops: float,
    peak_bandwidth_tbps: float,
) -> None:
    flops = logical_flops(n_columns)
    logical_bytes = split_bench.logical_global_bytes(n_columns)
    print(f"{prefix}_ms={ms:.6f}")
    print(f"{prefix}_us={ms * 1000.0:.3f}")
    print(f"{prefix}_tflops={split_bench.tflops(ms, flops):.6f}")
    print(
        f"{prefix}_logical_bandwidth_tbps="
        f"{split_bench.bandwidth_tbps(ms, logical_bytes):.6f}"
    )
    print(
        f"{prefix}_fp64_peak_pct="
        f"{100.0 * split_bench.tflops(ms, flops) / peak_fp64_tflops:.6f}"
    )
    print(
        f"{prefix}_bandwidth_peak_pct="
        f"{100.0 * split_bench.bandwidth_tbps(ms, logical_bytes) / peak_bandwidth_tbps:.6f}"
    )
    if config is not None:
        padded = padded_flops(config, n_columns)
        estimated_bytes = estimated_tile_global_bytes(config, n_columns)
        print(f"{prefix}_padded_tflops={split_bench.tflops(ms, padded):.6f}")
        print(
            f"{prefix}_estimated_tile_bandwidth_tbps="
            f"{split_bench.bandwidth_tbps(ms, estimated_bytes):.6f}"
        )
        print(f"{prefix}_work_inflation={work_inflation(config, n_columns):.6f}")


def launch_baseline(inputs: CandidateInputs, out: torch.Tensor) -> None:
    config = current_config()
    kernel = build_generic_kernel(
        DERIVATIVE_M,
        inputs.n_columns,
        DERIVATIVE_K,
        config,
    )
    call_tilelang_kernel(kernel, (inputs.base.d_merged, inputs.base.q_by_node, out))


def launch_candidate(
    inputs: CandidateInputs,
    config: SpecializedConfig,
    out: torch.Tensor,
) -> None:
    if config.kind == "generic":
        kernel = build_generic_kernel(
            DERIVATIVE_M,
            inputs.n_columns,
            effective_k(config),
            config,
        )
        a = inputs.base.d_merged if config.pad_k_to == DERIVATIVE_K else inputs.d_merged_k36
        b = inputs.base.q_by_node if config.pad_k_to == DERIVATIVE_K else inputs.q_by_node_k36
        call_tilelang_kernel(kernel, (a, b, out))
        return

    if config.kind == "fulla_staged":
        kernel = build_fulla_staged_kernel(inputs.n_columns, config)
        call_tilelang_kernel(kernel, (inputs.d_merged_k36, inputs.q_by_node_k36, out))
        return

    if config.kind == "fullk_group":
        kernel = build_fullk_group_kernel(inputs.n_columns, config)
        call_tilelang_kernel(kernel, (inputs.d_merged_k36, inputs.q_by_node_k36, out))
        return

    if config.kind == "tri35":
        row_start = 0
        tri_inputs = inputs.tri35 if config.pad_k_to == DERIVATIVE_K else inputs.tri35_k36
        tri_b = inputs.base.q_by_node if config.pad_k_to == DERIVATIVE_K else inputs.q_by_node_k36
        for rows, tri_a in zip(config.row_splits, tri_inputs, strict=True):
            row_stop = row_start + rows
            kernel = build_generic_kernel(
                rows,
                inputs.n_columns,
                effective_k(config),
                config,
            )
            call_tilelang_kernel(
                kernel,
                (tri_a, tri_b, out[row_start:row_stop]),
            )
            row_start = row_stop
        return

    raise ValueError(f"unsupported config kind {config.kind!r}")


def run_config(
    args,
    inputs: CandidateInputs,
    config: SpecializedConfig,
    reference: torch.Tensor,
    torch_ms: float,
    single_ms: float,
) -> tuple[float, bool]:
    print_config_summary(config, inputs.n_columns)
    actual = torch.empty_like(reference)
    try:
        launch_candidate(inputs, config, actual)
        split_bench.synchronize()
    except Exception as exc:
        print(f"candidate_build_error={type(exc).__name__}: {exc}")
        return float("inf"), False

    if not split_bench.print_compare("candidate", actual, reference):
        return float("inf"), False

    candidate_ms = split_bench.time_callable(
        lambda: launch_candidate(inputs, config, actual),
        backend=args.profile_backend,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    report_perf(
        "candidate",
        candidate_ms,
        config,
        inputs.n_columns,
        peak_fp64_tflops=args.peak_fp64_tflops,
        peak_bandwidth_tbps=args.peak_bandwidth_tbps,
    )
    print(f"speedup_vs_torch={torch_ms / candidate_ms:.6f}")
    print(f"speedup_vs_single={single_ms / candidate_ms:.6f}")
    return candidate_ms, True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark specialized TileLang derivative GEMM candidates."
    )
    parser.add_argument(
        "--mesh-name",
        default="scenario1_profile_lc0p20.msh",
        help="Scenario1 mesh used to build real derivative inputs.",
    )
    parser.add_argument(
        "--n-columns",
        type=int,
        nargs="+",
        default=None,
        help="Target N columns. Defaults to the scenario1 AoS column count.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Config name to benchmark. Repeat to select multiple configs.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configs and exit.",
    )
    parser.add_argument(
        "--profile-backend",
        choices=("event", "cudagraph", "cupti"),
        default="event",
        help="Timing backend.",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--iterations", type=int, default=100, help="Timed iterations.")
    parser.add_argument(
        "--peak-fp64-tflops",
        type=float,
        default=4.0,
        help="Peak FP64 throughput used for reporting.",
    )
    parser.add_argument(
        "--peak-bandwidth-tbps",
        type=float,
        default=1.8,
        help="Peak DRAM bandwidth used for reporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = available_configs()
    if args.list_configs:
        for config in configs:
            print(config.name)
        return

    baseline = current_config()
    if args.config:
        selected = [config_by_name(name) for name in args.config]
    else:
        selected = [config for config in configs if config.name != baseline.name]

    base_inputs, sim = split_bench.prepare_inputs(args.mesh_name)
    target_columns = args.n_columns or [base_inputs.n_columns]

    props = torch.cuda.get_device_properties(0)
    print(f"device={props.name}")
    print(f"shared_memory_per_block={props.shared_memory_per_block}")
    print(f"warp_size={getattr(props, 'warp_size', None)}")
    print(f"L2_cache_size={getattr(props, 'L2_cache_size', None)}")
    print(f"mesh_name={args.mesh_name}")
    print(f"N_tets={sim.N_tets}")
    print(f"Np={sim.Np}")
    print(f"N_columns_scenario={base_inputs.n_columns}")
    print(f"profile_backend={args.profile_backend}")
    print(f"warmup={args.warmup}")
    print(f"iterations={args.iterations}")

    any_candidate_faster = False
    for n_columns in target_columns:
        print("---")
        print(f"N_columns={n_columns}")
        print(
            f"input_columns_mode="
            f"{split_bench.column_resize_mode(base_inputs.n_columns, n_columns)}"
        )
        inputs = make_candidate_inputs(base_inputs, n_columns)
        reference = torch.empty(
            (DERIVATIVE_M, n_columns),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        torch.mm(inputs.base.d_merged, inputs.base.q_by_node, out=reference)
        split_bench.synchronize()

        torch_ms = split_bench.time_callable(
            lambda: torch.mm(
                inputs.base.d_merged,
                inputs.base.q_by_node,
                out=reference,
            ),
            backend=args.profile_backend,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        report_perf(
            "torch",
            torch_ms,
            None,
            n_columns,
            peak_fp64_tflops=args.peak_fp64_tflops,
            peak_bandwidth_tbps=args.peak_bandwidth_tbps,
        )

        single_out = torch.empty_like(reference)
        launch_baseline(inputs, single_out)
        split_bench.synchronize()
        if not split_bench.print_compare("single_current", single_out, reference):
            raise RuntimeError("current single-kernel baseline failed correctness")
        single_ms = split_bench.time_callable(
            lambda: launch_baseline(inputs, single_out),
            backend=args.profile_backend,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        report_perf(
            "single_current",
            single_ms,
            baseline,
            n_columns,
            peak_fp64_tflops=args.peak_fp64_tflops,
            peak_bandwidth_tbps=args.peak_bandwidth_tbps,
        )
        print(f"single_current_speedup_vs_torch={torch_ms / single_ms:.6f}")

        best_name = baseline.name
        best_ms = single_ms
        for config in selected:
            if config.name == baseline.name:
                continue
            print("---")
            candidate_ms, ok = run_config(
                args,
                inputs,
                config,
                reference,
                torch_ms,
                single_ms,
            )
            if ok and candidate_ms < best_ms:
                best_ms = candidate_ms
                best_name = config.name
            if ok and candidate_ms < single_ms:
                any_candidate_faster = True
        print(f"best_config_for_N={best_name}")
        print(f"best_ms_for_N={best_ms:.6f}")
    print(f"any_candidate_faster_than_single={int(any_candidate_faster)}")


if __name__ == "__main__":
    main()
