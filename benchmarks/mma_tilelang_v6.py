from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass

import torch
import tilelang
import tilelang.language as T
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
    )


def get_configs(m: int, k: int, shared_memory_limit: int, sweep_level: str, warp_size: int) -> list[KernelConfig]:
    if m == 105 and k == 35:
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

    return [
        config
        for config in dedupe_configs(candidates)
        if supported_config(config, shared_memory_limit, warp_size)
    ]


def tilelang_matmul_out(kernel, a, b, out):
    kernel(a, b, out)
    return out


def torch_matmul_out(a, b, out):
    torch.mm(a, b, out=out)
    return out


def tflops(ms: float, m: int, k: int, n: int) -> float:
    return 2.0 * m * k * n * 1.0e-12 / (ms * 1.0e-3)


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
    return fp64_matmul_tn(
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


def make_row(m: int, k: int, n: int, torch_ms: float, torch_q20: float, torch_q80: float, config: KernelConfig):
    return {
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
        "error": None,
    }


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
    row = make_row(m, k, n, torch_ms, torch_q20, torch_q80, config)
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

        quantiles = [0.5, 0.2, 0.8]
        tilelang_ms, tilelang_q20, tilelang_q80 = normalize_bench_result(
            do_bench(
                lambda: tilelang_matmul_out(kernel, a, b, out_tilelang),
                warmup=args.warmup,
                rep=args.rep,
                quantiles=quantiles,
                backend=args.profile_backend,
            )
        )
        row.update(
            {
                "tilelang_ms": tilelang_ms,
                "tilelang_tflops": tflops(tilelang_ms, m, k, n),
                "tilelang_min_tflops": tflops(tilelang_q80, m, k, n),
                "tilelang_max_tflops": tflops(tilelang_q20, m, k, n),
            }
        )
    except Exception as exc:
        row["error"] = format_exception(exc, trace=args.error_trace)
    return row


def bench_one(m: int, k: int, n: int, args):
    shared_memory_limit = args.shared_memory_kb * 1024
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    warp_size = getattr(props, "warp_size", 32) or 32
    configs = get_configs(m, k, shared_memory_limit, args.sweep_level, warp_size)
    if not configs:
        raise RuntimeError(f"no candidate configs fit {args.shared_memory_kb} KiB shared memory limit")

    a = torch.randn((m, k), device="cuda", dtype=torch.float64)
    b = torch.randn((k, n), device="cuda", dtype=torch.float64)
    out_torch = torch.empty((m, n), device="cuda", dtype=torch.float64)
    out_tilelang = torch.empty_like(out_torch)

    torch_matmul_out(a, b, out_torch)
    torch.cuda.synchronize()

    quantiles = [0.5, 0.2, 0.8]
    torch_ms, torch_q20, torch_q80 = normalize_bench_result(
        do_bench(
            lambda: torch_matmul_out(a, b, out_torch),
            warmup=args.warmup,
            rep=args.rep,
            quantiles=quantiles,
            backend=args.profile_backend,
        )
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
            f"policy={config.policy:<7} "
            f"tilelang=failed"
        )
        if row["error"] is not None:
            for line in row["error"].splitlines():
                print("    " + line)
        return

    speedup = row["tilelang_tflops"] / row["torch_tflops"]
    print(
        f"  {config.name:<{name_width}} shared={row['shared_memory_bytes'] // 1024:>2} KiB "
        f"policy={config.policy:<7} "
        f"tilelang={row['tilelang_tflops']:8.4f} TFLOPS ({row['tilelang_ms']:8.4f} ms) "
        f"speedup={speedup:6.3f}x"
    )


def print_csv(rows) -> None:
    print()
    print("csv:")
    print("M,K,N,config,policy,shared_kib,torch_ms,torch_tflops,tilelang_ms,tilelang_tflops,tilelang_speedup,error")
    for row in rows:
        config = row["config"]
        tilelang_ms = "" if row["tilelang_ms"] is None else f"{row['tilelang_ms']:.6f}"
        tilelang_tflops = "" if row["tilelang_tflops"] is None else f"{row['tilelang_tflops']:.6f}"
        tilelang_speedup = "" if row["tilelang_tflops"] is None else f"{row['tilelang_tflops'] / row['torch_tflops']:.6f}"
        error = "" if row["error"] is None else row["error"].splitlines()[0].replace(",", ";")
        print(
            f"{row['M']},{row['K']},{row['N']},{config.name},{config.policy},{row['shared_memory_bytes'] // 1024},"
            f"{row['torch_ms']:.6f},{row['torch_tflops']:.6f},"
            f"{tilelang_ms},{tilelang_tflops},{tilelang_speedup},{error}"
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
        choices=["core", "reg", "wide"],
        default="core",
        help="core runs stable configs; reg adds larger accumulator and t256 configs; wide adds legacy variants.",
    )
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
    print()

    all_rows = []
    for shape_name, m, k in shapes:
        print(f"shape[{shape_name}]: A=({m}, {k}), B=({k}, N), dtype=fp64")
        for n in args.n_values:
            rows = bench_one(m, k, n, args)
            all_rows.extend(rows)
            torch_row = rows[0]
            print(
                f"M={m:>3} K={k:>3} N={n:>8} "
                f"torch={torch_row['torch_tflops']:8.4f} TFLOPS ({torch_row['torch_ms']:8.4f} ms)"
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
                    f"speedup={best_speedup:.3f}x"
                )
        print()
        torch.cuda.empty_cache()

    print_csv(all_rows)


if __name__ == "__main__":
    main()
