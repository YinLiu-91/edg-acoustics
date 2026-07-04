from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass

import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench


SHAPE_PRESETS = {
    # EDG scenario1 derivative: D_merged[105,35] @ q_by_node[35,N]
    "derivative": (105, 35),
    # EDG scenario1 surface lift: lift[35,60] @ flux_by_face[60,N]
    "lift": (35, 60),
}


K_M_PER_WARP = 16
K_N_PER_WARP = 16


@dataclass(frozen=True)
class KernelConfig:
    name: str
    block_M: int
    block_N: int
    block_K: int
    num_stages: int
    threads: int = 128
    enable_swizzle: bool = True
    policy: str = "square"
    use_shared_store: bool = False
    persistent: bool = False


@tilelang.jit
def fp64_matmul_tn(
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
    use_shared_store,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float64),
        B: T.Tensor((K, N), T.float64),
        C: T.Tensor((M, N), T.float64),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float64)
            B_shared = T.alloc_shared((block_N, block_K), T.float64)
            C_local = T.alloc_fragment((block_M, block_N), T.float64)
            if use_shared_store:
                C_shared = T.alloc_shared((block_M, block_N), T.float64)

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
            if use_shared_store:
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])
            else:
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.jit
def fp64_matmul_tn_persistent(
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
    use_shared_store,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float64),
        B: T.Tensor((K, N), T.float64),
        C: T.Tensor((M, N), T.float64),
    ):
        sm_num = driver.get_num_sms()
        with T.Kernel(sm_num, threads=threads) as block_id:
            A_shared = T.alloc_shared((block_M, block_K), T.float64)
            B_shared = T.alloc_shared((block_N, block_K), T.float64)
            C_local = T.alloc_fragment((block_M, block_N), T.float64)
            if use_shared_store:
                C_shared = T.alloc_shared((block_M, block_N), T.float64)

            T.use_swizzle(panel_size=10, enable=enable_swizzle)
            for by, bx in T.Persistent(
                [T.ceildiv(M, block_M), T.ceildiv(N, block_N)], sm_num, block_id
            ):
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
                if use_shared_store:
                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[by * block_M, bx * block_N])
                else:
                    T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def shared_memory_bytes(config: KernelConfig) -> int:
    stage_factor = max(config.num_stages, 1)
    elements = config.block_M * config.block_K + config.block_N * config.block_K
    staged_bytes = elements * 8 * stage_factor
    if config.use_shared_store:
        staged_bytes += config.block_M * config.block_N * 8
    return staged_bytes


def warp_partition_supported(config: KernelConfig, warp_size: int) -> bool:
    if config.threads % warp_size != 0:
        return False
    num_warps = config.threads // warp_size
    if num_warps < 1:
        return False

    if config.policy == "fullrow":
        m_warp = num_warps
        n_warp = 1
        if config.block_M % (m_warp * K_M_PER_WARP) != 0:
            m_warp = config.block_M // K_M_PER_WARP
            if m_warp < 1:
                return False
            n_warp = num_warps // m_warp
            if n_warp == 0:
                n_warp = 1
        return m_warp * n_warp == num_warps

    if config.policy == "fullcol":
        m_warp = 1
        n_warp = num_warps
        if config.block_N % (n_warp * K_N_PER_WARP) != 0:
            n_warp = config.block_N // K_N_PER_WARP
            if n_warp < 1:
                return False
            m_warp = num_warps // n_warp
            if m_warp == 0:
                m_warp = 1
        return m_warp * n_warp == num_warps

    if config.policy != "square":
        return False

    max_m_warps = config.block_M // K_M_PER_WARP
    for m_warp in range(1, min(max_m_warps, num_warps) + 1):
        n_warp = num_warps // m_warp
        if m_warp * n_warp != num_warps:
            continue
        if config.block_M < m_warp * K_M_PER_WARP:
            continue
        if config.block_N < n_warp * K_N_PER_WARP:
            continue
        return True
    return False


def supported_config(config: KernelConfig, shared_memory_limit: int, warp_size: int) -> bool:
    if config.block_M % K_M_PER_WARP != 0:
        return False
    if config.block_N % K_N_PER_WARP != 0 or config.block_K % 4 != 0:
        return False
    if shared_memory_bytes(config) > shared_memory_limit:
        return False
    return warp_partition_supported(config, warp_size)


def policy_value(policy: str):
    if policy == "square":
        return T.GemmWarpPolicy.Square
    if policy == "fullrow":
        return T.GemmWarpPolicy.FullRow
    if policy == "fullcol":
        return T.GemmWarpPolicy.FullCol
    raise ValueError(f"unknown GEMM warp policy: {policy}")


def dedupe_configs(configs: list[KernelConfig]) -> list[KernelConfig]:
    seen = set()
    result = []
    for config in configs:
        key = (
            config.block_M,
            config.block_N,
            config.block_K,
            config.num_stages,
            config.threads,
            config.enable_swizzle,
            config.policy,
            config.use_shared_store,
            config.persistent,
        )
        if key not in seen:
            seen.add(key)
            result.append(config)
    return result


def with_policy(config: KernelConfig, policy: str) -> KernelConfig:
    if policy == "square":
        return config
    return KernelConfig(
        f"{config.name}_{policy}",
        config.block_M,
        config.block_N,
        config.block_K,
        config.num_stages,
        config.threads,
        config.enable_swizzle,
        policy,
        config.use_shared_store,
        config.persistent,
    )


def with_shared_store(config: KernelConfig) -> KernelConfig:
    suffix = "_ss" if not config.name.endswith("_ss") else ""
    return KernelConfig(
        f"{config.name}{suffix}",
        config.block_M,
        config.block_N,
        config.block_K,
        config.num_stages,
        config.threads,
        config.enable_swizzle,
        config.policy,
        True,
        config.persistent,
    )


def with_persistent(config: KernelConfig) -> KernelConfig:
    suffix = "_persistent" if not config.name.endswith("_persistent") else ""
    return KernelConfig(
        f"{config.name}{suffix}",
        config.block_M,
        config.block_N,
        config.block_K,
        config.num_stages,
        config.threads,
        config.enable_swizzle,
        config.policy,
        config.use_shared_store,
        True,
    )


def named_config(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    policy: str,
    *,
    use_shared_store: bool = False,
) -> KernelConfig:
    name = f"bm{block_m}_bn{block_n}_bk{block_k}_s{num_stages}_t{threads}"
    config = KernelConfig(
        name,
        block_m,
        block_n,
        block_k,
        num_stages,
        threads=threads,
        policy=policy,
        use_shared_store=use_shared_store,
    )
    if policy != "square":
        config = with_policy(config, policy)
    if use_shared_store:
        config = with_shared_store(config)
    return config


def c500_derivative_configs(include_persistent: bool) -> list[KernelConfig]:
    candidates = [
        KernelConfig("bm32_bn64_bk16_s1_t128", 32, 64, 16, 1),
        KernelConfig("bm32_bn64_bk16_s0_t256", 32, 64, 16, 0, threads=256),
        KernelConfig("bm32_bn64_bk16_s1_t256", 32, 64, 16, 1, threads=256),
        with_policy(KernelConfig("bm32_bn64_bk16_s0_t256", 32, 64, 16, 0, threads=256), "fullcol"),
        with_policy(KernelConfig("bm32_bn64_bk16_s1_t256", 32, 64, 16, 1, threads=256), "fullcol"),
    ]

    # C500 has a large register file. Covering M=105 in one CTA should reduce
    # repeated B tile loads and expose whether extra accumulator registers pay off.
    for block_n in (32, 64, 96):
        for block_k in (8, 12, 16):
            for num_stages in (0, 1):
                for threads in (128, 256):
                    for policy in ("fullcol", "square"):
                        candidates.append(
                            named_config(112, block_n, block_k, num_stages, threads, policy)
                        )

    # Two-M-tile candidates trade some repeated B traffic for lower accumulator
    # pressure. These are useful when bm112 occupancy is too low.
    for block_m in (64, 80, 96):
        for block_n in (64, 96):
            for block_k in (12, 16):
                for num_stages in (0, 1):
                    for policy in ("fullcol", "square"):
                        candidates.append(
                            named_config(block_m, block_n, block_k, num_stages, 256, policy)
                        )

    # Lift-style configs validate whether the successful lift tile shape transfers.
    for block_m in (48, 64):
        for block_k in (12, 16):
            for num_stages in (0, 1):
                for policy in ("fullcol", "square"):
                    candidates.append(named_config(block_m, 64, block_k, num_stages, 256, policy))

    # C500 can benefit from register -> shared -> global epilogues. Keep this
    # focused on configs that can fit C_shared under the 64 KiB block limit.
    shared_store_bases = [
        named_config(112, 32, block_k, num_stages, threads, policy)
        for block_k in (12, 16)
        for num_stages in (0, 1)
        for threads in (128, 256)
        for policy in ("fullcol", "square")
    ]
    shared_store_bases += [
        named_config(80, 64, block_k, 0, 256, policy)
        for block_k in (12, 16)
        for policy in ("fullcol", "square")
    ]
    candidates += [with_shared_store(config) for config in shared_store_bases]

    if include_persistent:
        persistent_bases = [
            named_config(112, 32, 16, 0, 128, "fullcol"),
            named_config(112, 64, 16, 0, 256, "fullcol"),
            named_config(80, 64, 16, 0, 256, "fullcol"),
        ]
        candidates += [with_persistent(config) for config in persistent_bases]

    return candidates


def c500_next_derivative_configs() -> list[KernelConfig]:
    candidates = [
        named_config(112, 64, 12, 1, 256, "fullcol"),
        named_config(112, 64, 12, 0, 256, "fullcol"),
    ]

    # Narrow search around the accepted C500 derivative winner. Square policy
    # often fails fragment layout normalization for this M=105, K=35, warp64 shape.
    for block_m in (96, 112, 128):
        for block_n in (48, 64, 80, 96):
            for block_k in (4, 8, 12):
                for num_stages in (0, 1):
                    for threads in (192, 256, 320, 384):
                        candidates.append(
                            named_config(
                                block_m,
                                block_n,
                                block_k,
                                num_stages,
                                threads,
                                "fullcol",
                            )
                        )

    return candidates


def get_configs(
    m: int,
    k: int,
    shared_memory_limit: int,
    sweep_level: str,
    warp_size: int,
    include_persistent: bool = False,
    config_names: tuple[str, ...] = (),
) -> list[KernelConfig]:
    if m == 105 and k == 35:
        if sweep_level == "c500-deep":
            candidates = c500_derivative_configs(include_persistent=include_persistent)
        elif sweep_level == "c500-next":
            candidates = c500_next_derivative_configs()
        else:
            derivative_winners = [
                with_policy(KernelConfig("bm32_bn64_bk16_s0_t256", 32, 64, 16, 0, threads=256), "fullcol"),
                with_policy(KernelConfig("bm32_bn64_bk16_s1_t256", 32, 64, 16, 1, threads=256), "fullcol"),
            ]
            candidates = [
                KernelConfig("bm32_bn64_bk16_s1_t128", 32, 64, 16, 1),
                KernelConfig("bm32_bn64_bk16_s0_t256", 32, 64, 16, 0, threads=256),
                KernelConfig("bm32_bn64_bk16_s1_t256", 32, 64, 16, 1, threads=256),
                *derivative_winners,
            ]
        if sweep_level in ("reg", "wide"):
            square_reg = [
                KernelConfig("bm32_bn96_bk16_s0_t256", 32, 96, 16, 0, threads=256),
                KernelConfig("bm32_bn96_bk16_s1_t256", 32, 96, 16, 1, threads=256),
                KernelConfig("bm32_bn128_bk16_s0_t256", 32, 128, 16, 0, threads=256),
                KernelConfig("bm32_bn128_bk16_s1_t256", 32, 128, 16, 1, threads=256),
                KernelConfig("bm96_bn64_bk16_s0_t256", 96, 64, 16, 0, threads=256),
                KernelConfig("bm96_bn64_bk16_s1_t256", 96, 64, 16, 1, threads=256),
            ]
            candidates += square_reg
            policy_base = [
                KernelConfig("bm32_bn96_bk16_s0_t256", 32, 96, 16, 0, threads=256),
                KernelConfig("bm32_bn96_bk16_s1_t256", 32, 96, 16, 1, threads=256),
                KernelConfig("bm32_bn128_bk16_s0_t256", 32, 128, 16, 0, threads=256),
                KernelConfig("bm32_bn128_bk16_s1_t256", 32, 128, 16, 1, threads=256),
                KernelConfig("bm96_bn64_bk16_s0_t256", 96, 64, 16, 0, threads=256),
                KernelConfig("bm96_bn64_bk16_s1_t256", 96, 64, 16, 1, threads=256),
            ]
            candidates += [with_policy(config, "fullcol") for config in policy_base]
        if sweep_level == "wide":
            candidates += [
                KernelConfig("bm64_bn64_bk16_s0_t128", 64, 64, 16, 0),
                KernelConfig("bm64_bn64_bk16_s1_t128", 64, 64, 16, 1),
                KernelConfig("bm64_bn64_bk16_s2_t128", 64, 64, 16, 2),
                KernelConfig("bm64_bn128_bk16_s0_t128", 64, 128, 16, 0),
                KernelConfig("bm64_bn128_bk16_s1_t128", 64, 128, 16, 1),
                KernelConfig("bm128_bn64_bk16_s0_t128", 128, 64, 16, 0),
                KernelConfig("bm128_bn64_bk16_s1_t128", 128, 64, 16, 1),
                KernelConfig("bm128_bn128_bk16_s0_t128", 128, 128, 16, 0),
                KernelConfig("bm32_bn64_bk16_s0_t128", 32, 64, 16, 0),
                KernelConfig("bm32_bn64_bk16_s3_t128", 32, 64, 16, 3),
                KernelConfig("bm64_bn32_bk16_s3_t128", 64, 32, 16, 3),
                KernelConfig("bm32_bn128_bk16_s2_t128", 32, 128, 16, 2),
                KernelConfig("bm64_bn64_bk16_s3_t128", 64, 64, 16, 3),
                KernelConfig("bm128_bn32_bk16_s2_t128", 128, 32, 16, 2),
                KernelConfig("bm64_bn32_bk64_s0_t128", 64, 32, 64, 0),
            ]
            policy_base = [
                KernelConfig("bm128_bn64_bk16_s0_t128", 128, 64, 16, 0),
                KernelConfig("bm128_bn64_bk16_s1_t128", 128, 64, 16, 1),
                KernelConfig("bm128_bn128_bk16_s0_t128", 128, 128, 16, 0),
                KernelConfig("bm64_bn128_bk16_s0_t128", 64, 128, 16, 0),
                KernelConfig("bm64_bn128_bk16_s1_t128", 64, 128, 16, 1),
            ]
            candidates += [with_policy(config, "fullcol") for config in policy_base]
            candidates += [with_policy(config, "fullrow") for config in policy_base]
    elif m == 35 and k == 60:
        lift_winners = [
            with_policy(KernelConfig("bm48_bn64_bk16_s0_t256", 48, 64, 16, 0, threads=256), "fullcol"),
            with_policy(KernelConfig("bm48_bn64_bk16_s1_t256", 48, 64, 16, 1, threads=256), "fullcol"),
            with_policy(KernelConfig("bm64_bn64_bk16_s0_t256", 64, 64, 16, 0, threads=256), "fullcol"),
            with_policy(KernelConfig("bm64_bn64_bk16_s1_t256", 64, 64, 16, 1, threads=256), "fullcol"),
            with_policy(KernelConfig("bm64_bn64_bk16_s0_t128", 64, 64, 16, 0), "fullcol"),
            with_policy(KernelConfig("bm64_bn64_bk16_s1_t128", 64, 64, 16, 1), "fullcol"),
        ]
        candidates = [
            KernelConfig("bm64_bn64_bk16_s1_t128", 64, 64, 16, 1),
            KernelConfig("bm64_bn64_bk16_s0_t256", 64, 64, 16, 0, threads=256),
            KernelConfig("bm64_bn64_bk16_s1_t256", 64, 64, 16, 1, threads=256),
            *lift_winners,
        ]
        if sweep_level in ("reg", "wide"):
            square_reg = [
                KernelConfig("bm64_bn96_bk16_s0_t256", 64, 96, 16, 0, threads=256),
                KernelConfig("bm64_bn96_bk16_s1_t256", 64, 96, 16, 1, threads=256),
                KernelConfig("bm64_bn128_bk16_s0_t256", 64, 128, 16, 0, threads=256),
                KernelConfig("bm64_bn128_bk16_s1_t256", 64, 128, 16, 1, threads=256),
                KernelConfig("bm32_bn128_bk16_s0_t256", 32, 128, 16, 0, threads=256),
                KernelConfig("bm32_bn128_bk16_s1_t256", 32, 128, 16, 1, threads=256),
            ]
            candidates += square_reg
            policy_base = [
                KernelConfig("bm64_bn96_bk16_s0_t256", 64, 96, 16, 0, threads=256),
                KernelConfig("bm64_bn96_bk16_s1_t256", 64, 96, 16, 1, threads=256),
                KernelConfig("bm32_bn128_bk16_s0_t256", 32, 128, 16, 0, threads=256),
                KernelConfig("bm32_bn128_bk16_s1_t256", 32, 128, 16, 1, threads=256),
            ]
            candidates += [with_policy(config, "fullcol") for config in policy_base]
        if sweep_level == "wide":
            candidates += [
                KernelConfig("bm64_bn64_bk16_s0_t128", 64, 64, 16, 0),
                KernelConfig("bm64_bn64_bk16_s2_t128", 64, 64, 16, 2),
                KernelConfig("bm64_bn96_bk16_s0_t128", 64, 96, 16, 0),
                KernelConfig("bm64_bn96_bk16_s1_t128", 64, 96, 16, 1),
                KernelConfig("bm64_bn128_bk16_s0_t128", 64, 128, 16, 0),
                KernelConfig("bm64_bn128_bk16_s1_t128", 64, 128, 16, 1),
                KernelConfig("bm32_bn128_bk16_s0_t128", 32, 128, 16, 0),
                KernelConfig("bm32_bn128_bk16_s1_t128", 32, 128, 16, 1),
                KernelConfig("bm16_bn128_bk16_s0_t128", 16, 128, 16, 0),
                KernelConfig("bm16_bn128_bk16_s1_t128", 16, 128, 16, 1),
                KernelConfig("bm16_bn128_bk16_s2_t128", 16, 128, 16, 2),
                KernelConfig("bm64_bn32_bk64_s0_t128", 64, 32, 64, 0),
                KernelConfig("bm64_bn32_bk32_s2_t128", 64, 32, 32, 2),
                KernelConfig("bm32_bn64_bk32_s2_t128", 32, 64, 32, 2),
                KernelConfig("bm32_bn128_bk16_s2_t128", 32, 128, 16, 2),
                KernelConfig("bm64_bn96_bk16_s2_t128", 64, 96, 16, 2),
                KernelConfig("bm64_bn128_bk16_s2_t128", 64, 128, 16, 2),
            ]
            policy_base = [
                KernelConfig("bm64_bn64_bk16_s0_t128", 64, 64, 16, 0),
                KernelConfig("bm64_bn64_bk16_s1_t128", 64, 64, 16, 1),
                KernelConfig("bm64_bn96_bk16_s0_t128", 64, 96, 16, 0),
                KernelConfig("bm64_bn96_bk16_s1_t128", 64, 96, 16, 1),
                KernelConfig("bm64_bn128_bk16_s0_t128", 64, 128, 16, 0),
                KernelConfig("bm64_bn128_bk16_s1_t128", 64, 128, 16, 1),
                KernelConfig("bm16_bn128_bk16_s1_t128", 16, 128, 16, 1),
                KernelConfig("bm64_bn64_bk16_s0_t256", 64, 64, 16, 0, threads=256),
                KernelConfig("bm64_bn64_bk16_s1_t256", 64, 64, 16, 1, threads=256),
                KernelConfig("bm64_bn96_bk16_s0_t256", 64, 96, 16, 0, threads=256),
                KernelConfig("bm64_bn96_bk16_s1_t256", 64, 96, 16, 1, threads=256),
                KernelConfig("bm64_bn128_bk16_s0_t256", 64, 128, 16, 0, threads=256),
                KernelConfig("bm64_bn128_bk16_s1_t256", 64, 128, 16, 1, threads=256),
                KernelConfig("bm32_bn128_bk16_s0_t256", 32, 128, 16, 0, threads=256),
                KernelConfig("bm32_bn128_bk16_s1_t256", 32, 128, 16, 1, threads=256),
            ]
            candidates += [with_policy(config, "fullcol") for config in policy_base]
            candidates += [with_policy(config, "fullrow") for config in policy_base]
    else:
        candidates = [
            KernelConfig("bm64_bn64_bk16_s0_t128", 64, 64, 16, 0),
            KernelConfig("bm64_bn64_bk16_s1_t128", 64, 64, 16, 1),
            KernelConfig("bm64_bn64_bk16_s2_t128", 64, 64, 16, 2),
            KernelConfig("bm64_bn128_bk16_s0_t128", 64, 128, 16, 0),
            KernelConfig("bm64_bn128_bk16_s1_t128", 64, 128, 16, 1),
            KernelConfig("bm128_bn32_bk16_s2_t128", 128, 32, 16, 2),
        ]

    configs = [
        config
        for config in dedupe_configs(candidates)
        if supported_config(config, shared_memory_limit, warp_size)
    ]
    if config_names:
        requested = set(config_names)
        configs = [config for config in configs if config.name in requested]
        missing = sorted(requested.difference(config.name for config in configs))
        if missing:
            known = ", ".join(config.name for config in dedupe_configs(candidates))
            raise ValueError(f"unknown or unsupported --config entries: {missing}; known candidates: {known}")
    return configs


def tilelang_matmul_out(kernel, a, b, out):
    kernel(a, b, out)
    return out


def torch_matmul_out(a, b, out):
    torch.mm(a, b, out=out)
    return out


def tflops(ms: float, m: int, k: int, n: int) -> float:
    return 2.0 * m * k * n * 1.0e-12 / (ms * 1.0e-3)


def throughput_tflops(ms: float, flops: int) -> float:
    return flops * 1.0e-12 / (ms * 1.0e-3)


def bandwidth_tbps(ms: float, num_bytes: int) -> float:
    return num_bytes * 1.0e-12 / (ms * 1.0e-3)


def ceildiv_int(a: int, b: int) -> int:
    return (a + b - 1) // b


def logical_flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def logical_global_bytes(m: int, k: int, n: int) -> int:
    return (m * k + k * n + m * n) * 8


def padded_flops(m: int, k: int, n: int, config: KernelConfig) -> int:
    return (
        2
        * ceildiv_int(m, config.block_M)
        * config.block_M
        * ceildiv_int(n, config.block_N)
        * config.block_N
        * ceildiv_int(k, config.block_K)
        * config.block_K
    )


def estimated_tile_global_bytes(m: int, k: int, n: int, config: KernelConfig) -> int:
    m_tiles = ceildiv_int(m, config.block_M)
    n_tiles = ceildiv_int(n, config.block_N)
    k_tiles = ceildiv_int(k, config.block_K)
    a_bytes = m_tiles * n_tiles * k_tiles * config.block_M * config.block_K * 8
    b_bytes = m_tiles * n_tiles * k_tiles * config.block_N * config.block_K * 8
    c_store_bytes = m_tiles * n_tiles * config.block_M * config.block_N * 8
    return a_bytes + b_bytes + c_store_bytes


def enrich_metrics(row, args) -> None:
    m = row["M"]
    k = row["K"]
    n = row["N"]
    config = row["config"]
    logical_ops = logical_flops(m, k, n)
    padded_ops = padded_flops(m, k, n, config)
    logical_bytes = logical_global_bytes(m, k, n)
    tile_bytes = estimated_tile_global_bytes(m, k, n, config)
    arithmetic_intensity = logical_ops / logical_bytes
    row.update(
        {
            "logical_flops": logical_ops,
            "padded_flops": padded_ops,
            "work_inflation": padded_ops / logical_ops,
            "logical_global_bytes": logical_bytes,
            "estimated_tile_global_bytes": tile_bytes,
            "arithmetic_intensity": arithmetic_intensity,
            "roofline_bound_tflops": min(
                args.peak_fp64_tflops,
                args.peak_bandwidth_tbps * arithmetic_intensity,
            ),
        }
    )


def gib(num_bytes: int) -> float:
    return num_bytes / 1024**3


def print_device_info() -> None:
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print("device:", device)
    print("name:", props.name)
    print("capability:", torch.cuda.get_device_capability(device))
    print("SM/CU:", props.multi_processor_count)
    print("warp_size:", getattr(props, "warp_size", None))
    print("total_memory GiB:", f"{gib(props.total_memory):.2f}")
    print("L2 bytes:", getattr(props, "L2_cache_size", None))
    print("shared_memory_per_block:", getattr(props, "shared_memory_per_block", None))


def validate_sample(ref, actual, n: int, columns: int = 256) -> tuple[bool, torch.Tensor]:
    if n <= columns:
        idx = torch.arange(n, device=ref.device)
    else:
        first = torch.arange(columns // 2, device=ref.device)
        last = n - columns // 2 + torch.arange(columns // 2, device=ref.device)
        idx = torch.cat([first, last])
    return torch.allclose(actual[:, idx], ref[:, idx], rtol=1.0e-10, atol=1.0e-10), idx


def normalize_bench_result(result):
    if isinstance(result, (list, tuple)):
        if len(result) == 3:
            return result
        if len(result) == 1:
            return result[0], result[0], result[0]
    return result, result, result


def format_exception(exc: Exception, trace: bool = False) -> str:
    if trace:
        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
    return str(exc)


def compile_kernel(m: int, n: int, k: int, config: KernelConfig):
    kernel_builder = fp64_matmul_tn_persistent if config.persistent else fp64_matmul_tn
    return kernel_builder(
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
        config.use_shared_store,
    )


def make_row(m: int, k: int, n: int, torch_ms: float, torch_q20: float, torch_q80: float, config: KernelConfig, args):
    row = {
        "M": m,
        "K": k,
        "N": n,
        "config": config,
        "shared_memory_bytes": shared_memory_bytes(config),
        "torch_ms": torch_ms,
        "torch_tflops": tflops(torch_ms, m, k, n),
        "torch_min_tflops": tflops(torch_q80, m, k, n),
        "torch_max_tflops": tflops(torch_q20, m, k, n),
        "tilelang_ms": None,
        "tilelang_tflops": None,
        "tilelang_min_tflops": None,
        "tilelang_max_tflops": None,
        "tilelang_padded_tflops": None,
        "tilelang_logical_bandwidth_tbps": None,
        "tilelang_estimated_tile_bandwidth_tbps": None,
        "tilelang_fp64_peak_pct": None,
        "tilelang_bandwidth_peak_pct": None,
        "error": None,
    }
    row["torch_logical_bandwidth_tbps"] = bandwidth_tbps(torch_ms, logical_global_bytes(m, k, n))
    row["torch_fp64_peak_pct"] = 100.0 * row["torch_tflops"] / args.peak_fp64_tflops
    row["torch_bandwidth_peak_pct"] = 100.0 * row["torch_logical_bandwidth_tbps"] / args.peak_bandwidth_tbps
    enrich_metrics(row, args)
    return row


def do_bench_repeated(fn, args):
    best = None
    for _ in range(args.repeat):
        result = normalize_bench_result(
            do_bench(
                fn,
                warmup=args.warmup,
                rep=args.rep,
                quantiles=[0.5, 0.2, 0.8],
                backend=args.profile_backend,
            )
        )
        if best is None or result[0] < best[0]:
            best = result
    return best


def bench_config(
    m: int,
    k: int,
    n: int,
    config: KernelConfig,
    a,
    b,
    out_torch,
    out_tilelang,
    torch_ms: float,
    torch_q20: float,
    torch_q80: float,
    args,
):
    row = make_row(m, k, n, torch_ms, torch_q20, torch_q80, config, args)
    try:
        kernel = compile_kernel(m, n, k, config)
        tilelang_matmul_out(kernel, a, b, out_tilelang)
        torch.cuda.synchronize()

        if not args.no_validate:
            ok, idx = validate_sample(out_torch, out_tilelang, n)
            if not ok:
                ref_sample = out_torch[:, idx]
                actual_sample = out_tilelang[:, idx]
                diff = (ref_sample - actual_sample).abs()
                rel = diff / ref_sample.abs().clamp_min(1.0e-300)
                raise RuntimeError(
                    f"validation failed: max_abs={diff.max().item()}, max_rel={rel.max().item()}"
                )

        tilelang_ms, tilelang_q20, tilelang_q80 = do_bench_repeated(
            lambda: tilelang_matmul_out(kernel, a, b, out_tilelang),
            args,
        )
        row.update(
            {
                "tilelang_ms": tilelang_ms,
                "tilelang_tflops": tflops(tilelang_ms, m, k, n),
                "tilelang_min_tflops": tflops(tilelang_q80, m, k, n),
                "tilelang_max_tflops": tflops(tilelang_q20, m, k, n),
                "tilelang_padded_tflops": throughput_tflops(tilelang_ms, row["padded_flops"]),
                "tilelang_logical_bandwidth_tbps": bandwidth_tbps(
                    tilelang_ms, row["logical_global_bytes"]
                ),
                "tilelang_estimated_tile_bandwidth_tbps": bandwidth_tbps(
                    tilelang_ms, row["estimated_tile_global_bytes"]
                ),
            }
        )
        row["tilelang_fp64_peak_pct"] = 100.0 * row["tilelang_tflops"] / args.peak_fp64_tflops
        row["tilelang_bandwidth_peak_pct"] = (
            100.0 * row["tilelang_logical_bandwidth_tbps"] / args.peak_bandwidth_tbps
        )
    except Exception as exc:
        row["error"] = format_exception(exc, trace=args.error_trace)
    return row


def bench_one(m: int, k: int, n: int, args):
    shared_memory_limit = args.shared_memory_kb * 1024
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    warp_size = getattr(props, "warp_size", 32) or 32
    configs = get_configs(
        m,
        k,
        shared_memory_limit,
        args.sweep_level,
        warp_size,
        include_persistent=args.include_persistent,
        config_names=tuple(args.config),
    )
    if not configs:
        raise RuntimeError(f"no candidate configs fit {args.shared_memory_kb} KiB shared memory limit")

    a = torch.randn((m, k), device="cuda", dtype=torch.float64)
    b = torch.randn((k, n), device="cuda", dtype=torch.float64)
    out_torch = torch.empty((m, n), device="cuda", dtype=torch.float64)
    out_tilelang = torch.empty_like(out_torch)

    torch_matmul_out(a, b, out_torch)
    torch.cuda.synchronize()

    torch_ms, torch_q20, torch_q80 = do_bench_repeated(
        lambda: torch_matmul_out(a, b, out_torch),
        args,
    )

    rows = []
    for config in configs:
        rows.append(
            bench_config(
                m,
                k,
                n,
                config,
                a,
                b,
                out_torch,
                out_tilelang,
                torch_ms,
                torch_q20,
                torch_q80,
                args,
            )
        )
    return rows


def print_config_row(row) -> None:
    config = row["config"]
    name_width = 42
    if row["tilelang_tflops"] is None:
        print(
            f"  {config.name:<{name_width}} shared={row['shared_memory_bytes'] // 1024:>2} KiB "
            f"policy={config.policy:<7} persistent={int(config.persistent)} "
            f"tilelang=failed"
        )
        if row["error"] is not None:
            for line in row["error"].splitlines():
                print("    " + line)
        return

    speedup = row["tilelang_tflops"] / row["torch_tflops"]
    print(
        f"  {config.name:<{name_width}} shared={row['shared_memory_bytes'] // 1024:>2} KiB "
        f"policy={config.policy:<7} persistent={int(config.persistent)} "
        f"tilelang={row['tilelang_tflops']:8.4f} TFLOPS ({row['tilelang_ms']:8.4f} ms) "
        f"padded={row['tilelang_padded_tflops']:8.4f} TFLOPS "
        f"infl={row['work_inflation']:5.2f}x "
        f"bw={row['tilelang_logical_bandwidth_tbps']:6.3f}/{row['tilelang_estimated_tile_bandwidth_tbps']:6.3f} TB/s "
        f"fp64={row['tilelang_fp64_peak_pct']:5.1f}% "
        f"speedup={speedup:6.3f}x"
    )


def print_csv(rows) -> None:
    print()
    print("csv:")
    print(
        "M,K,N,config,policy,persistent,shared_kib,work_inflation,arithmetic_intensity,"
        "roofline_bound_tflops,torch_ms,torch_tflops,torch_logical_bandwidth_tbps,"
        "torch_fp64_peak_pct,tilelang_ms,tilelang_tflops,tilelang_padded_tflops,"
        "tilelang_logical_bandwidth_tbps,tilelang_estimated_tile_bandwidth_tbps,"
        "tilelang_fp64_peak_pct,tilelang_speedup,error"
    )
    for row in rows:
        config = row["config"]
        tilelang_ms = "" if row["tilelang_ms"] is None else f"{row['tilelang_ms']:.6f}"
        tilelang_tflops = "" if row["tilelang_tflops"] is None else f"{row['tilelang_tflops']:.6f}"
        tilelang_padded_tflops = "" if row["tilelang_padded_tflops"] is None else f"{row['tilelang_padded_tflops']:.6f}"
        tilelang_logical_bandwidth = (
            "" if row["tilelang_logical_bandwidth_tbps"] is None else f"{row['tilelang_logical_bandwidth_tbps']:.6f}"
        )
        tilelang_estimated_tile_bandwidth = (
            ""
            if row["tilelang_estimated_tile_bandwidth_tbps"] is None
            else f"{row['tilelang_estimated_tile_bandwidth_tbps']:.6f}"
        )
        tilelang_fp64_peak_pct = (
            "" if row["tilelang_fp64_peak_pct"] is None else f"{row['tilelang_fp64_peak_pct']:.6f}"
        )
        tilelang_speedup = "" if row["tilelang_tflops"] is None else f"{row['tilelang_tflops'] / row['torch_tflops']:.6f}"
        error = "" if row["error"] is None else row["error"].splitlines()[0].replace(",", ";")
        print(
            f"{row['M']},{row['K']},{row['N']},{config.name},{config.policy},{int(config.persistent)},"
            f"{row['shared_memory_bytes'] // 1024},{row['work_inflation']:.6f},"
            f"{row['arithmetic_intensity']:.6f},{row['roofline_bound_tflops']:.6f},"
            f"{row['torch_ms']:.6f},{row['torch_tflops']:.6f},"
            f"{row['torch_logical_bandwidth_tbps']:.6f},{row['torch_fp64_peak_pct']:.6f},"
            f"{tilelang_ms},{tilelang_tflops},{tilelang_padded_tflops},"
            f"{tilelang_logical_bandwidth},{tilelang_estimated_tile_bandwidth},"
            f"{tilelang_fp64_peak_pct},{tilelang_speedup},{error}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        choices=["all", *SHAPE_PRESETS.keys()],
        default="all",
        help="Shape preset to benchmark. all runs derivative and lift.",
    )
    parser.add_argument(
        "--M",
        "--m",
        dest="m",
        type=int,
        default=None,
        help="Override preset M for A[M,K] @ B[K,N]. Requires --K.",
    )
    parser.add_argument(
        "--K",
        "--k",
        dest="k",
        type=int,
        default=None,
        help="Override preset K for A[M,K] @ B[K,N]. Requires --M.",
    )
    parser.add_argument(
        "--N",
        "--n",
        "--n-values",
        dest="n_values",
        type=int,
        nargs="+",
        default=[1377572],
        help="N values to benchmark for A[M,K] @ B[K,N].",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--error-trace", action="store_true")
    parser.add_argument("--shared-memory-kb", type=int, default=64)
    parser.add_argument(
        "--sweep-level",
        choices=["core", "reg", "wide", "c500-deep", "c500-next"],
        default="core",
        help=(
            "core runs stable configs; reg adds larger accumulator and t256 configs; "
            "wide adds legacy variants; c500-deep adds register-aware derivative GEMM candidates; "
            "c500-next narrows the derivative search around the accepted C500 winner."
        ),
    )
    parser.add_argument(
        "--config",
        nargs="*",
        default=[],
        help="Run only the named candidate config(s) from the selected sweep level.",
    )
    parser.add_argument(
        "--include-persistent",
        action="store_true",
        help="Include experimental persistent derivative GEMM configs in c500-deep.",
    )
    parser.add_argument("--peak-fp64-tflops", type=float, default=4.0)
    parser.add_argument("--peak-bandwidth-tbps", type=float, default=1.8)
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each do_bench call and keep the best median.")
    parser.add_argument("--warmup", type=float, default=25)
    parser.add_argument("--rep", type=float, default=100)
    parser.add_argument(
        "--profile-backend",
        choices=["event", "cupti", "cudagraph"],
        default="event",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this FP64 TileLang benchmark.")
    if (args.m is None) != (args.k is None):
        raise ValueError("--M and --K must be provided together.")
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1.")

    if args.m is not None:
        shapes = [("custom", args.m, args.k)]
    elif args.shape == "all":
        shapes = [(name, m, k) for name, (m, k) in SHAPE_PRESETS.items()]
    else:
        m, k = SHAPE_PRESETS[args.shape]
        shapes = [(args.shape, m, k)]

    torch.manual_seed(args.seed)
    print_device_info()
    print("tilelang B layout: B[K,N], copied into shared as B_shared[block_N,block_K]")
    print("shared memory budget KiB:", args.shared_memory_kb)
    print("sweep level:", args.sweep_level)
    print("peak_fp64_tflops:", args.peak_fp64_tflops)
    print("peak_bandwidth_tbps:", args.peak_bandwidth_tbps)
    print("benchmark_repeat:", args.repeat)
    if args.config:
        print("config filter:", ",".join(args.config))
    print("include_persistent:", int(args.include_persistent))
    print()

    all_rows = []
    for shape_name, m, k in shapes:
        print(f"shape[{shape_name}]: A=({m}, {k}), B=({k}, N), dtype=fp64")
        for n in args.n_values:
            rows = bench_one(m, k, n, args)
            all_rows.extend(rows)
            torch_row = rows[0]
            print("candidate_count:", len(rows))
            print(
                f"M={m:>3} K={k:>3} N={n:>8} "
                f"torch={torch_row['torch_tflops']:8.4f} TFLOPS ({torch_row['torch_ms']:8.4f} ms) "
                f"bw={torch_row['torch_logical_bandwidth_tbps']:.4f} TB/s "
                f"fp64={torch_row['torch_fp64_peak_pct']:.1f}% "
                f"AI={torch_row['arithmetic_intensity']:.3f} flop/byte "
                f"roofline={torch_row['roofline_bound_tflops']:.3f} TFLOPS"
            )
            for row in rows:
                print_config_row(row)
            valid_rows = [row for row in rows if row["tilelang_tflops"] is not None]
            if valid_rows:
                best = max(valid_rows, key=lambda row: row["tilelang_tflops"])
                best_speedup = best["tilelang_tflops"] / best["torch_tflops"]
                print(
                    f"  best={best['config'].name} "
                    f"{best['tilelang_tflops']:.4f} TFLOPS ({best['tilelang_ms']:.4f} ms) "
                    f"padded={best['tilelang_padded_tflops']:.4f} TFLOPS "
                    f"fp64={best['tilelang_fp64_peak_pct']:.1f}% "
                    f"speedup={best_speedup:.3f}x"
                )
        print()
        torch.cuda.empty_cache()

    print_csv(all_rows)


if __name__ == "__main__":
    main()
