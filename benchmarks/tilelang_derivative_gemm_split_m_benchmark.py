"""Benchmark split-M TileLang derivative GEMM candidates on scenario1 inputs.

This isolates the merged derivative hot path:

    D_merged[105, 35] @ q_by_node[35, N]

The current runtime backend uses a single TileLang kernel. This benchmark adds
split-M candidates that launch multiple specialized TileLang kernels over row
partitions to reduce padded M work.
"""

from __future__ import annotations

import argparse
import bisect
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.autograd.profiler import DeviceType


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TESTS_DIR))

from scenario1_utils import build_scenario1_simulation  # noqa: E402


DERIVATIVE_M = 105
DERIVATIVE_K = 35

T = None
_JITTED_FP64_DERIVATIVE_MATMUL_TN = None
_KERNEL_CACHE: dict[tuple[int, int, "GemmPartConfig"], object] = {}
_ACCEPTS_SKIP_VALIDATION: dict[int, bool] = {}


@dataclass(frozen=True)
class GemmPartConfig:
    rows: int
    block_M: int
    block_N: int
    block_K: int
    num_stages: int
    threads: int
    policy: str = "fullcol"
    enable_swizzle: bool = True


@dataclass(frozen=True)
class SplitMConfig:
    name: str
    parts: tuple[GemmPartConfig, ...]

    @property
    def total_rows(self) -> int:
        return sum(part.rows for part in self.parts)


@dataclass
class KernelInputs:
    d_merged: torch.Tensor
    q_by_node: torch.Tensor

    @property
    def dtype(self) -> torch.dtype:
        return self.q_by_node.dtype

    @property
    def device(self) -> torch.device:
        return self.q_by_node.device

    @property
    def n_columns(self) -> int:
        return self.q_by_node.shape[1]


@contextmanager
def temporary_env(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _fp64_derivative_matmul_tn(
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


def _jitted_fp64_derivative_matmul_tn():
    global T, _JITTED_FP64_DERIVATIVE_MATMUL_TN

    if _JITTED_FP64_DERIVATIVE_MATMUL_TN is not None:
        return _JITTED_FP64_DERIVATIVE_MATMUL_TN

    import tilelang
    import tilelang.language as tilelang_language

    T = tilelang_language
    _JITTED_FP64_DERIVATIVE_MATMUL_TN = tilelang.jit(_fp64_derivative_matmul_tn)
    return _JITTED_FP64_DERIVATIVE_MATMUL_TN


def policy_value(policy: str):
    if policy == "square":
        return T.GemmWarpPolicy.Square
    if policy == "fullrow":
        return T.GemmWarpPolicy.FullRow
    if policy == "fullcol":
        return T.GemmWarpPolicy.FullCol
    raise ValueError(f"unknown GEMM warp policy: {policy}")


def build_tilelang_kernel(n_columns: int, part: GemmPartConfig):
    cache_key = (part.rows, n_columns, part)
    kernel = _KERNEL_CACHE.get(cache_key)
    if kernel is not None:
        return kernel

    jitted = _jitted_fp64_derivative_matmul_tn()
    kernel = jitted(
        part.rows,
        n_columns,
        DERIVATIVE_K,
        part.block_M,
        part.block_N,
        part.block_K,
        part.num_stages,
        part.threads,
        part.enable_swizzle,
        policy_value(part.policy),
    )
    _KERNEL_CACHE[cache_key] = kernel
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


def synchronize() -> None:
    torch.cuda.synchronize()


def make_flush_cache() -> torch.Tensor:
    props = torch.cuda.get_device_properties(0)
    l2_bytes = int(getattr(props, "L2_cache_size", 0) or 0)
    if l2_bytes <= 0:
        l2_bytes = int(256e6)
    return torch.empty(max(l2_bytes // 4, 1), dtype=torch.int, device="cuda")


def _sum_kernel_time_us(kineto_results) -> tuple[float, int]:
    windows: list[tuple[int, int]] = []
    kernels: list[tuple[int, int]] = []
    for event in kineto_results.events():
        if event.device_type() != DeviceType.CUDA:
            continue
        if event.is_user_annotation():
            if event.name() == "edg_acoustics_kernel":
                windows.append((event.start_ns(), event.end_ns()))
            continue
        kernels.append((event.start_ns(), event.duration_ns()))

    windows.sort()
    starts = [window[0] for window in windows]
    ends = [window[1] for window in windows]
    total_us = 0.0
    for start_ns, duration_ns in kernels:
        idx = bisect.bisect_right(starts, start_ns) - 1
        if idx >= 0 and start_ns < ends[idx]:
            total_us += duration_ns / 1000.0
    return total_us, len(windows)


def time_event(fn, *, cache: torch.Tensor, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        cache.zero_()
        fn()
    synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        cache.zero_()
        start_events[index].record()
        fn()
        end_events[index].record()
    synchronize()
    return sum(
        float(start.elapsed_time(end))
        for start, end in zip(start_events, end_events, strict=True)
    ) / iterations


def time_cudagraph(fn, *, cache: torch.Tensor, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        cache.zero_()
        fn()
    synchronize()

    replay_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(replay_stream):
        synchronize()
        with torch.cuda.graph(graph):
            fn()
    synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        cache.zero_()
        start_events[index].record()
        graph.replay()
        end_events[index].record()
    synchronize()
    return sum(
        float(start.elapsed_time(end))
        for start, end in zip(start_events, end_events, strict=True)
    ) / iterations


def time_cupti(fn, *, cache: torch.Tensor, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        cache.zero_()
        fn()
    synchronize()

    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as profiler:
            for _ in range(iterations):
                cache.zero_()
                with torch.profiler.record_function("edg_acoustics_kernel"):
                    fn()
                synchronize()
        total_us, n_regions = _sum_kernel_time_us(profiler.profiler.kineto_results)
        if n_regions != iterations:
            raise RuntimeError(f"expected {iterations} profiled regions, got {n_regions}")
        return total_us / iterations * 1.0e-3
    except Exception as exc:
        print(f"cupti_fallback={type(exc).__name__}: {exc}")
        return time_event(fn, cache=cache, warmup=0, iterations=iterations)


def time_callable(fn, *, backend: str, warmup: int, iterations: int) -> float:
    cache = make_flush_cache()
    if backend == "cupti":
        return time_cupti(fn, cache=cache, warmup=warmup, iterations=iterations)
    if backend == "cudagraph":
        return time_cudagraph(fn, cache=cache, warmup=warmup, iterations=iterations)
    return time_event(fn, cache=cache, warmup=warmup, iterations=iterations)


def max_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, bool]:
    diff = (actual - expected).abs()
    max_abs = float(diff.max().item())
    rel = diff / expected.abs().clamp_min(1.0e-300)
    max_rel = float(rel.max().item())
    ok = bool(torch.allclose(actual, expected, rtol=1.0e-10, atol=1.0e-10))
    return max_abs, max_rel, ok


def print_compare(prefix: str, actual: torch.Tensor, expected: torch.Tensor) -> bool:
    max_abs, max_rel, ok = max_error(actual, expected)
    print(f"{prefix}_ok={int(ok)}")
    print(f"{prefix}_max_abs={max_abs:.6e}")
    print(f"{prefix}_max_rel={max_rel:.6e}")
    return ok


def ceildiv_int(a: int, b: int) -> int:
    return (a + b - 1) // b


def logical_flops(n_columns: int) -> int:
    return 2 * DERIVATIVE_M * DERIVATIVE_K * n_columns


def logical_global_bytes(n_columns: int) -> int:
    return (DERIVATIVE_M * DERIVATIVE_K + DERIVATIVE_K * n_columns + DERIVATIVE_M * n_columns) * 8


def part_padded_flops(part: GemmPartConfig, n_columns: int) -> int:
    return (
        2
        * ceildiv_int(part.rows, part.block_M)
        * part.block_M
        * ceildiv_int(n_columns, part.block_N)
        * part.block_N
        * ceildiv_int(DERIVATIVE_K, part.block_K)
        * part.block_K
    )


def padded_flops(config: SplitMConfig, n_columns: int) -> int:
    return sum(part_padded_flops(part, n_columns) for part in config.parts)


def estimated_tile_global_bytes(config: SplitMConfig, n_columns: int) -> int:
    total = 0
    for part in config.parts:
        m_tiles = ceildiv_int(part.rows, part.block_M)
        n_tiles = ceildiv_int(n_columns, part.block_N)
        k_tiles = ceildiv_int(DERIVATIVE_K, part.block_K)
        a_bytes = m_tiles * n_tiles * k_tiles * part.block_M * part.block_K * 8
        b_bytes = m_tiles * n_tiles * k_tiles * part.block_N * part.block_K * 8
        c_store_bytes = m_tiles * n_tiles * part.block_M * part.block_N * 8
        total += a_bytes + b_bytes + c_store_bytes
    return total


def tflops(ms: float, flops: int) -> float:
    return flops * 1.0e-12 / (ms * 1.0e-3)


def bandwidth_tbps(ms: float, num_bytes: int) -> float:
    return num_bytes * 1.0e-12 / (ms * 1.0e-3)


def shared_memory_bytes(part: GemmPartConfig) -> int:
    stage_factor = max(part.num_stages, 1)
    elements = part.block_M * part.block_K + part.block_N * part.block_K
    return elements * 8 * stage_factor


def config_work_inflation(config: SplitMConfig, n_columns: int) -> float:
    return padded_flops(config, n_columns) / logical_flops(n_columns)


def validate_config(config: SplitMConfig) -> None:
    if config.total_rows != DERIVATIVE_M:
        raise ValueError(
            f"{config.name} covers {config.total_rows} rows, expected {DERIVATIVE_M}"
        )
    for part in config.parts:
        if part.rows <= 0:
            raise ValueError(f"{config.name} has non-positive rows in {part}")
        if part.block_M % 16 != 0:
            raise ValueError(
                f"{config.name} requires block_M multiple of 16, got {part.block_M}"
            )
        if part.block_N % 16 != 0:
            raise ValueError(
                f"{config.name} requires block_N multiple of 16, got {part.block_N}"
            )
        if part.block_K % 4 != 0:
            raise ValueError(
                f"{config.name} requires block_K multiple of 4, got {part.block_K}"
            )
        if part.threads % 64 != 0:
            raise ValueError(
                f"{config.name} requires threads multiple of 64, got {part.threads}"
            )


def available_configs() -> tuple[SplitMConfig, ...]:
    configs = (
        SplitMConfig(
            "single_bm128_bn64_bk4_s0_t256_fullcol",
            (
                GemmPartConfig(
                    rows=105,
                    block_M=128,
                    block_N=64,
                    block_K=4,
                    num_stages=0,
                    threads=256,
                ),
            ),
        ),
        SplitMConfig(
            "splitm96_9_bm96_16_bn64_bk4_s0_t256_fullcol",
            (
                GemmPartConfig(96, 96, 64, 4, 0, 256),
                GemmPartConfig(9, 16, 64, 4, 0, 256),
            ),
        ),
        SplitMConfig(
            "splitm96_9_bm96_16_bn64_bk4_main0_tail1_t256_fullcol",
            (
                GemmPartConfig(96, 96, 64, 4, 0, 256),
                GemmPartConfig(9, 16, 64, 4, 1, 256),
            ),
        ),
        SplitMConfig(
            "splitm80_25_bm80_32_bn64_bk4_s0_t256_fullcol",
            (
                GemmPartConfig(80, 80, 64, 4, 0, 256),
                GemmPartConfig(25, 32, 64, 4, 0, 256),
            ),
        ),
        SplitMConfig(
            "splitm64_32_9_bm64_32_16_bn64_bk4_s0_t256_fullcol",
            (
                GemmPartConfig(64, 64, 64, 4, 0, 256),
                GemmPartConfig(32, 32, 64, 4, 0, 256),
                GemmPartConfig(9, 16, 64, 4, 0, 256),
            ),
        ),
    )
    for config in configs:
        validate_config(config)
    return configs


def config_by_name(name: str) -> SplitMConfig:
    for config in available_configs():
        if config.name == name:
            return config
    known = ", ".join(config.name for config in available_configs())
    raise ValueError(f"unknown split-M config {name!r}; known configs: {known}")


def column_resize_mode(source_columns: int, target_columns: int) -> str:
    if target_columns == source_columns:
        return "scenario"
    if target_columns < source_columns:
        return "truncated"
    return "repeated"


def resize_q_by_node(q_by_node: torch.Tensor, n_columns: int) -> torch.Tensor:
    if n_columns <= 0:
        raise ValueError(f"n_columns must be positive, got {n_columns}")
    if n_columns == q_by_node.shape[1]:
        return q_by_node.contiguous()
    if n_columns < q_by_node.shape[1]:
        return q_by_node[:, :n_columns].contiguous()
    repeat = ceildiv_int(n_columns, q_by_node.shape[1])
    return q_by_node.repeat(1, repeat)[:, :n_columns].contiguous()


def prepare_inputs(mesh_name: str) -> tuple[KernelInputs, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("split-M derivative GEMM benchmark requires CUDA")
    with temporary_env(
        {
            "EDG_ACOUSTICS_DEVICE": "cuda",
            "EDG_ACOUSTICS_TILELANG_DERIVATIVE_GEMM": "0",
        }
    ):
        sim = build_scenario1_simulation(mesh_name=mesh_name, device="cuda")
    if sim.Np != 35:
        raise RuntimeError(f"benchmark requires Np=35, got {sim.Np}")
    if sim._D_merged.shape != (105, 35):
        raise RuntimeError(
            "benchmark requires merged derivative shape (105,35), got "
            f"{tuple(sim._D_merged.shape)}"
        )
    inputs = KernelInputs(
        d_merged=sim._D_merged.contiguous(),
        q_by_node=sim.Q_flat.contiguous(),
    )
    return inputs, sim


def launch_torch(inputs: KernelInputs, out: torch.Tensor) -> None:
    torch.mm(inputs.d_merged, inputs.q_by_node, out=out)


def launch_config(inputs: KernelInputs, config: SplitMConfig, out: torch.Tensor) -> None:
    row_start = 0
    for part in config.parts:
        row_stop = row_start + part.rows
        kernel = build_tilelang_kernel(inputs.n_columns, part)
        call_tilelang_kernel(
            kernel,
            (
                inputs.d_merged[row_start:row_stop],
                inputs.q_by_node,
                out[row_start:row_stop],
            ),
        )
        row_start = row_stop


def report_perf(
    prefix: str,
    ms: float,
    config: SplitMConfig | None,
    n_columns: int,
    *,
    peak_fp64_tflops: float,
    peak_bandwidth_tbps: float,
) -> None:
    flops = logical_flops(n_columns)
    logical_bytes = logical_global_bytes(n_columns)
    print(f"{prefix}_ms={ms:.6f}")
    print(f"{prefix}_us={ms * 1000.0:.3f}")
    print(f"{prefix}_tflops={tflops(ms, flops):.6f}")
    print(f"{prefix}_logical_bandwidth_tbps={bandwidth_tbps(ms, logical_bytes):.6f}")
    print(f"{prefix}_fp64_peak_pct={100.0 * tflops(ms, flops) / peak_fp64_tflops:.6f}")
    print(
        f"{prefix}_bandwidth_peak_pct="
        f"{100.0 * bandwidth_tbps(ms, logical_bytes) / peak_bandwidth_tbps:.6f}"
    )
    if config is not None:
        padded = padded_flops(config, n_columns)
        estimated_bytes = estimated_tile_global_bytes(config, n_columns)
        print(f"{prefix}_padded_tflops={tflops(ms, padded):.6f}")
        print(
            f"{prefix}_estimated_tile_bandwidth_tbps="
            f"{bandwidth_tbps(ms, estimated_bytes):.6f}"
        )
        print(f"{prefix}_work_inflation={config_work_inflation(config, n_columns):.6f}")


def print_config_summary(config: SplitMConfig, n_columns: int) -> None:
    print(f"config={config.name}")
    print(f"parts={len(config.parts)}")
    row_start = 0
    for index, part in enumerate(config.parts):
        row_stop = row_start + part.rows
        print(
            f"part{index}=rows[{row_start}:{row_stop}) "
            f"bm={part.block_M} bn={part.block_N} bk={part.block_K} "
            f"s={part.num_stages} t={part.threads} policy={part.policy} "
            f"shared_kib={shared_memory_bytes(part) / 1024:.1f}"
        )
        row_start = row_stop
    print(f"logical_flops={logical_flops(n_columns)}")
    print(f"padded_flops={padded_flops(config, n_columns)}")
    print(f"work_inflation={config_work_inflation(config, n_columns):.6f}")
    print(f"estimated_tile_global_bytes={estimated_tile_global_bytes(config, n_columns)}")


def run_config(
    args,
    inputs: KernelInputs,
    config: SplitMConfig,
    reference: torch.Tensor,
    torch_ms: float,
    single_ms: float | None,
) -> tuple[float, bool]:
    print_config_summary(config, inputs.n_columns)
    actual = torch.empty_like(reference)
    try:
        launch_config(inputs, config, actual)
        synchronize()
    except Exception as exc:
        print(f"candidate_build_error={type(exc).__name__}: {exc}")
        return float("inf"), False

    if not print_compare("candidate", actual, reference):
        return float("inf"), False

    candidate_ms = time_callable(
        lambda: launch_config(inputs, config, actual),
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
    if single_ms is not None:
        print(f"speedup_vs_single={single_ms / candidate_ms:.6f}")
    return candidate_ms, True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark split-M TileLang derivative GEMM candidates."
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
        help="List available split-M configs and exit.",
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

    current_config = config_by_name("single_bm128_bn64_bk4_s0_t256_fullcol")
    if args.config:
        requested = args.config
    else:
        requested = [config.name for config in configs if config.name != current_config.name]
    selected = [config_by_name(name) for name in requested]
    base_inputs, sim = prepare_inputs(args.mesh_name)
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
            f"{column_resize_mode(base_inputs.n_columns, n_columns)}"
        )
        inputs = KernelInputs(
            d_merged=base_inputs.d_merged,
            q_by_node=resize_q_by_node(base_inputs.q_by_node, n_columns),
        )
        reference = torch.empty(
            (DERIVATIVE_M, n_columns),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        launch_torch(inputs, reference)
        synchronize()

        torch_ms = time_callable(
            lambda: launch_torch(inputs, reference),
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
        launch_config(inputs, current_config, single_out)
        synchronize()
        if not print_compare("single_current", single_out, reference):
            raise RuntimeError("current single-kernel baseline failed correctness")
        single_ms = time_callable(
            lambda: launch_config(inputs, current_config, single_out),
            backend=args.profile_backend,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        report_perf(
            "single_current",
            single_ms,
            current_config,
            n_columns,
            peak_fp64_tflops=args.peak_fp64_tflops,
            peak_bandwidth_tbps=args.peak_bandwidth_tbps,
        )
        print(f"single_current_speedup_vs_torch={torch_ms / single_ms:.6f}")

        best_name = current_config.name
        best_ms = single_ms
        for config in selected:
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
            if ok and config.name != current_config.name and candidate_ms < single_ms:
                any_candidate_faster = True
        print(f"best_config_for_N={best_name}")
        print(f"best_ms_for_N={best_ms:.6f}")
    print(f"any_candidate_faster_than_single={int(any_candidate_faster)}")


if __name__ == "__main__":
    main()
