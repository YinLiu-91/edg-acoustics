"""Experimental benchmark for retired manual fp64 derivative-volume microkernels.

This isolates the same scenario1 hot path as the fused derivative-volume
benchmark:

    baseline: torch.mm(D_merged, Q) + affine AoS volume-surface Triton kernel
    reference: current best TileLang T.gemm fused derivative-volume kernel
    candidate: manual SIMT TileLang microkernel with explicit fp64 FMAs

The manual microkernel path is kept only as a correctness and profiling
reference. Active optimization work has moved back to the T.gemm fused
derivative-volume kernel, so the broad manual-microkernel sweep is intentionally
retired.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

import torch

import tilelang_derivative_volume_aos_benchmark as aos_bench  # noqa: E402
from edg_acoustics.tilelang_derivative_volume_aos import (  # noqa: E402
    build_tilelang_derivative_volume_aos_kernel,
)


_M = 35
_K = 35
_K_PAD = 36
_Q_LAYOUT_QKN = 0
_Q_LAYOUT_QNK = 1
_D_SOURCE_SHARED = 0
_D_SOURCE_GLOBAL = 1
_LOOP_SERIAL = 0
_LOOP_UNROLL = 1
_REFERENCE_CONFIG_NAME = "bp16_be8_bn32_bk16_s0_t128_fullcol"
MANUAL_MICROKERNEL_STATUS = "experimental_retired"

T = None
_JITTED_FP64_DERIVATIVE_VOLUME_MICROKERNEL = None


@dataclass(frozen=True)
class MicrokernelConfig:
    name: str
    block_p: int
    block_e: int
    threads: int
    q_layout: str = "qkn"
    d_source: str = "shared"
    loop_kind: str = "unroll"

    @property
    def block_n(self) -> int:
        return 4 * self.block_e

    @property
    def explicit_shared_memory_bytes(self) -> int:
        q_bytes = _K_PAD * self.block_n * 8
        d_bytes = 0
        if self.d_source == "shared":
            d_bytes = 3 * self.block_p * _K_PAD * 8
        return q_bytes + d_bytes


_CONFIGS: dict[str, MicrokernelConfig] = {
    config.name: config
    for config in (
        MicrokernelConfig("bp16_be8_t128_qkn_dshared_unroll", 16, 8, 128),
        MicrokernelConfig(
            "bp16_be8_t128_qkn_dglobal_unroll",
            16,
            8,
            128,
            d_source="global",
        ),
        MicrokernelConfig(
            "bp16_be8_t128_qnk_dshared_unroll",
            16,
            8,
            128,
            q_layout="qnk",
        ),
        MicrokernelConfig(
            "bp16_be8_t128_qkn_dshared_serial",
            16,
            8,
            128,
            loop_kind="serial",
        ),
        MicrokernelConfig("bp8_be8_t64_qkn_dshared_unroll", 8, 8, 64),
        MicrokernelConfig("bp16_be4_t64_qkn_dshared_unroll", 16, 4, 64),
        MicrokernelConfig("bp16_be16_t256_qkn_dshared_unroll", 16, 16, 256),
    )
}


def available_config_names() -> tuple[str, ...]:
    return tuple(_CONFIGS)


def get_config(name: str) -> MicrokernelConfig:
    try:
        return _CONFIGS[name]
    except KeyError as exc:
        known = ", ".join(available_config_names())
        raise ValueError(
            f"unknown TileLang derivative-volume microkernel config {name!r}; known: {known}"
        ) from exc


def _q_layout_code(name: str) -> int:
    if name == "qkn":
        return _Q_LAYOUT_QKN
    if name == "qnk":
        return _Q_LAYOUT_QNK
    raise ValueError(f"unknown q layout {name!r}")


def _d_source_code(name: str) -> int:
    if name == "shared":
        return _D_SOURCE_SHARED
    if name == "global":
        return _D_SOURCE_GLOBAL
    raise ValueError(f"unknown derivative source {name!r}")


def _loop_kind_code(name: str) -> int:
    if name == "serial":
        return _LOOP_SERIAL
    if name == "unroll":
        return _LOOP_UNROLL
    raise ValueError(f"unknown loop kind {name!r}")


def _fp64_derivative_volume_microkernel(
    M,
    N,
    K,
    n_tets,
    block_p,
    block_e,
    threads,
    q_layout,
    d_source,
    loop_kind,
    update_state,
):
    block_n = 4 * block_e

    @T.prim_func
    def main(
        Q: T.Tensor((M, N), T.float64),
        Dr: T.Tensor((M, K), T.float64),
        Ds: T.Tensor((M, K), T.float64),
        Dt: T.Tensor((M, K), T.float64),
        metric_p: T.Tensor((3, 3, n_tets), T.float64),
        metric_v: T.Tensor((3, 3, n_tets), T.float64),
        surface: T.Tensor((M, N), T.float64),
        rhs: T.Tensor((M, N), T.float64),
        q_update: T.Tensor((M, N), T.float64),
        coefficient: T.Tensor((1,), T.float64),
    ):
        with T.Kernel(
            T.ceildiv(n_tets, block_e),
            T.ceildiv(M, block_p),
            threads=threads,
        ) as (bx, by):
            if q_layout == _Q_LAYOUT_QKN:
                q_shared = T.alloc_shared((_K_PAD, block_n), T.float64)
            else:
                q_shared = T.alloc_shared((block_n, _K_PAD), T.float64)

            if d_source == _D_SOURCE_SHARED:
                dr_shared = T.alloc_shared((block_p, _K_PAD), T.float64)
                ds_shared = T.alloc_shared((block_p, _K_PAD), T.float64)
                dt_shared = T.alloc_shared((block_p, _K_PAD), T.float64)

            p_r = T.alloc_local((1,), T.float64)
            p_s = T.alloc_local((1,), T.float64)
            p_t = T.alloc_local((1,), T.float64)
            vx_r = T.alloc_local((1,), T.float64)
            vx_s = T.alloc_local((1,), T.float64)
            vx_t = T.alloc_local((1,), T.float64)
            vy_r = T.alloc_local((1,), T.float64)
            vy_s = T.alloc_local((1,), T.float64)
            vy_t = T.alloc_local((1,), T.float64)
            vz_r = T.alloc_local((1,), T.float64)
            vz_s = T.alloc_local((1,), T.float64)
            vz_t = T.alloc_local((1,), T.float64)

            T.use_swizzle(panel_size=10, enable=True)

            if d_source == _D_SOURCE_SHARED:
                for i, kk in T.Parallel(block_p, _K_PAD):
                    row = by * block_p + i
                    valid = (row < M) & (kk < K)
                    dr_shared[i, kk] = T.if_then_else(
                        valid,
                        Dr[row, kk],
                        T.float64(0.0),
                    )
                    ds_shared[i, kk] = T.if_then_else(
                        valid,
                        Ds[row, kk],
                        T.float64(0.0),
                    )
                    dt_shared[i, kk] = T.if_then_else(
                        valid,
                        Dt[row, kk],
                        T.float64(0.0),
                    )

            if q_layout == _Q_LAYOUT_QKN:
                for kk, j in T.Parallel(_K_PAD, block_n):
                    elem = bx * block_e + j // 4
                    field = j % 4
                    n_idx = elem * 4 + field
                    q_shared[kk, j] = T.if_then_else(
                        (kk < K) & (n_idx < N),
                        Q[kk, n_idx],
                        T.float64(0.0),
                    )
            else:
                for j, kk in T.Parallel(block_n, _K_PAD):
                    elem = bx * block_e + j // 4
                    field = j % 4
                    n_idx = elem * 4 + field
                    q_shared[j, kk] = T.if_then_else(
                        (kk < K) & (n_idx < N),
                        Q[kk, n_idx],
                        T.float64(0.0),
                    )

            coeff = coefficient[0]
            for i, e in T.Parallel(block_p, block_e):
                node = by * block_p + i
                elem = bx * block_e + e
                if node < M and elem < n_tets:
                    c = elem * 4
                    lc = e * 4

                    p_r[0] = T.float64(0.0)
                    p_s[0] = T.float64(0.0)
                    p_t[0] = T.float64(0.0)
                    vx_r[0] = T.float64(0.0)
                    vx_s[0] = T.float64(0.0)
                    vx_t[0] = T.float64(0.0)
                    vy_r[0] = T.float64(0.0)
                    vy_s[0] = T.float64(0.0)
                    vy_t[0] = T.float64(0.0)
                    vz_r[0] = T.float64(0.0)
                    vz_s[0] = T.float64(0.0)
                    vz_t[0] = T.float64(0.0)

                    if loop_kind == _LOOP_UNROLL:
                        for kk in T.unroll(K):
                            if d_source == _D_SOURCE_SHARED:
                                dr_val = dr_shared[i, kk]
                                ds_val = ds_shared[i, kk]
                                dt_val = dt_shared[i, kk]
                            else:
                                dr_val = Dr[node, kk]
                                ds_val = Ds[node, kk]
                                dt_val = Dt[node, kk]

                            if q_layout == _Q_LAYOUT_QKN:
                                q_p = q_shared[kk, lc]
                                q_vx = q_shared[kk, lc + 1]
                                q_vy = q_shared[kk, lc + 2]
                                q_vz = q_shared[kk, lc + 3]
                            else:
                                q_p = q_shared[lc, kk]
                                q_vx = q_shared[lc + 1, kk]
                                q_vy = q_shared[lc + 2, kk]
                                q_vz = q_shared[lc + 3, kk]

                            p_r[0] += dr_val * q_p
                            p_s[0] += ds_val * q_p
                            p_t[0] += dt_val * q_p
                            vx_r[0] += dr_val * q_vx
                            vx_s[0] += ds_val * q_vx
                            vx_t[0] += dt_val * q_vx
                            vy_r[0] += dr_val * q_vy
                            vy_s[0] += ds_val * q_vy
                            vy_t[0] += dt_val * q_vy
                            vz_r[0] += dr_val * q_vz
                            vz_s[0] += ds_val * q_vz
                            vz_t[0] += dt_val * q_vz
                    else:
                        for kk in T.serial(K):
                            if d_source == _D_SOURCE_SHARED:
                                dr_val = dr_shared[i, kk]
                                ds_val = ds_shared[i, kk]
                                dt_val = dt_shared[i, kk]
                            else:
                                dr_val = Dr[node, kk]
                                ds_val = Ds[node, kk]
                                dt_val = Dt[node, kk]

                            if q_layout == _Q_LAYOUT_QKN:
                                q_p = q_shared[kk, lc]
                                q_vx = q_shared[kk, lc + 1]
                                q_vy = q_shared[kk, lc + 2]
                                q_vz = q_shared[kk, lc + 3]
                            else:
                                q_p = q_shared[lc, kk]
                                q_vx = q_shared[lc + 1, kk]
                                q_vy = q_shared[lc + 2, kk]
                                q_vz = q_shared[lc + 3, kk]

                            p_r[0] += dr_val * q_p
                            p_s[0] += ds_val * q_p
                            p_t[0] += dt_val * q_p
                            vx_r[0] += dr_val * q_vx
                            vx_s[0] += ds_val * q_vx
                            vx_t[0] += dt_val * q_vx
                            vy_r[0] += dr_val * q_vy
                            vy_s[0] += ds_val * q_vy
                            vy_t[0] += dt_val * q_vy
                            vz_r[0] += dr_val * q_vz
                            vz_s[0] += ds_val * q_vz
                            vz_t[0] += dt_val * q_vz

                    rhs_p = (
                        metric_p[0, 0, elem] * vx_r[0]
                        + metric_p[1, 0, elem] * vx_s[0]
                        + metric_p[2, 0, elem] * vx_t[0]
                        + metric_p[0, 1, elem] * vy_r[0]
                        + metric_p[1, 1, elem] * vy_s[0]
                        + metric_p[2, 1, elem] * vy_t[0]
                        + metric_p[0, 2, elem] * vz_r[0]
                        + metric_p[1, 2, elem] * vz_s[0]
                        + metric_p[2, 2, elem] * vz_t[0]
                        + surface[node, c]
                    )
                    rhs_vx = (
                        metric_v[0, 0, elem] * p_r[0]
                        + metric_v[1, 0, elem] * p_s[0]
                        + metric_v[2, 0, elem] * p_t[0]
                        + surface[node, c + 1]
                    )
                    rhs_vy = (
                        metric_v[0, 1, elem] * p_r[0]
                        + metric_v[1, 1, elem] * p_s[0]
                        + metric_v[2, 1, elem] * p_t[0]
                        + surface[node, c + 2]
                    )
                    rhs_vz = (
                        metric_v[0, 2, elem] * p_r[0]
                        + metric_v[1, 2, elem] * p_s[0]
                        + metric_v[2, 2, elem] * p_t[0]
                        + surface[node, c + 3]
                    )

                    rhs[node, c] = rhs_p
                    rhs[node, c + 1] = rhs_vx
                    rhs[node, c + 2] = rhs_vy
                    rhs[node, c + 3] = rhs_vz
                    if update_state:
                        q_update[node, c] = q_update[node, c] + coeff * rhs_p
                        q_update[node, c + 1] = q_update[node, c + 1] + coeff * rhs_vx
                        q_update[node, c + 2] = q_update[node, c + 2] + coeff * rhs_vy
                        q_update[node, c + 3] = q_update[node, c + 3] + coeff * rhs_vz

    return main


def _jitted_fp64_derivative_volume_microkernel():
    global T, _JITTED_FP64_DERIVATIVE_VOLUME_MICROKERNEL

    if _JITTED_FP64_DERIVATIVE_VOLUME_MICROKERNEL is not None:
        return _JITTED_FP64_DERIVATIVE_VOLUME_MICROKERNEL

    import tilelang
    import tilelang.language as tilelang_language

    T = tilelang_language
    _JITTED_FP64_DERIVATIVE_VOLUME_MICROKERNEL = tilelang.jit(
        _fp64_derivative_volume_microkernel
    )
    return _JITTED_FP64_DERIVATIVE_VOLUME_MICROKERNEL


def build_microkernel(
    n_tets: int,
    *,
    config_name: str,
    update_state: bool,
):
    if n_tets <= 0:
        raise ValueError("TileLang derivative-volume microkernel requires n_tets > 0.")

    config = get_config(config_name)
    jitted = _jitted_fp64_derivative_volume_microkernel()
    return jitted(
        _M,
        4 * n_tets,
        _K,
        n_tets,
        config.block_p,
        config.block_e,
        config.threads,
        _q_layout_code(config.q_layout),
        _d_source_code(config.d_source),
        _loop_kind_code(config.loop_kind),
        update_state,
    )


def run_config(args, inputs: aos_bench.KernelInputs, config_name: str, reference_kernel) -> bool:
    config = get_config(config_name)
    props = torch.cuda.get_device_properties(0)
    shared_limit = props.shared_memory_per_block
    update_state = args.mode == "update"
    coefficient = inputs.coefficient
    q_by_node = inputs.q_by_node
    flops = aos_bench.logical_flops(inputs, update_state=update_state)
    baseline_memory_bytes = aos_bench.logical_memory_bytes(
        inputs,
        update_state=update_state,
        candidate=False,
    )
    fused_memory_bytes = aos_bench.logical_memory_bytes(
        inputs,
        update_state=update_state,
        candidate=True,
    )
    candidate_memory_bytes = fused_memory_bytes

    print(f"config={config.name}")
    print(f"block_p={config.block_p}")
    print(f"block_e={config.block_e}")
    print(f"block_n={config.block_n}")
    print(f"threads={config.threads}")
    print(f"q_layout={config.q_layout}")
    print(f"d_source={config.d_source}")
    print(f"loop_kind={config.loop_kind}")
    print(f"explicit_shared_memory_kib={config.explicit_shared_memory_bytes / 1024:.1f}")
    print(f"logical_flops={flops:.0f}")
    print(f"baseline_logical_bytes={baseline_memory_bytes}")
    print(f"reference_logical_bytes={fused_memory_bytes}")
    print(f"candidate_logical_bytes={candidate_memory_bytes}")
    print(f"mode={args.mode}")
    if config.explicit_shared_memory_bytes > shared_limit:
        print(
            "candidate_skip=shared_memory "
            f"required={config.explicit_shared_memory_bytes} limit={shared_limit}"
        )
        return False

    try:
        candidate_kernel = build_microkernel(
            inputs.n_tets,
            config_name=config.name,
            update_state=update_state,
        )
        if args.export_sources is not None:
            args.export_sources.mkdir(parents=True, exist_ok=True)
            aos_bench.export_candidate_sources(
                candidate_kernel,
                args.export_sources,
                f"microkernel_{config.name}",
            )
    except Exception as exc:
        print(f"candidate_build_error={type(exc).__name__}: {exc}")
        if args.debug_traceback:
            traceback.print_exc()
        return False

    baseline_rhs = torch.empty_like(q_by_node)
    reference_rhs = torch.empty_like(q_by_node)
    candidate_rhs = torch.empty_like(q_by_node)
    baseline_update = q_by_node.clone() if update_state else None
    reference_update = q_by_node.clone() if update_state else None
    candidate_update = q_by_node.clone() if update_state else None

    aos_bench.launch_baseline(
        inputs,
        baseline_rhs,
        baseline_update,
        coefficient,
    )
    aos_bench.call_tilelang_kernel(
        reference_kernel,
        (
            q_by_node,
            inputs.dr,
            inputs.ds,
            inputs.dt,
            inputs.metric_p_affine,
            inputs.metric_v_affine,
            inputs.surface_by_node,
            reference_rhs,
            reference_update if reference_update is not None else reference_rhs,
            inputs.coefficient_tensor.fill_(coefficient),
        ),
        skip_validation=not args.no_skip_validation,
    )
    aos_bench.call_tilelang_kernel(
        candidate_kernel,
        (
            q_by_node,
            inputs.dr,
            inputs.ds,
            inputs.dt,
            inputs.metric_p_affine,
            inputs.metric_v_affine,
            inputs.surface_by_node,
            candidate_rhs,
            candidate_update if candidate_update is not None else candidate_rhs,
            inputs.coefficient_tensor.fill_(coefficient),
        ),
        skip_validation=not args.no_skip_validation,
    )
    aos_bench.synchronize()

    baseline_ok = aos_bench.print_compare("reference_rhs", reference_rhs, baseline_rhs)
    candidate_ok = aos_bench.print_compare("candidate_rhs", candidate_rhs, baseline_rhs)
    reference_state_ok = True
    candidate_state_ok = True
    if update_state:
        reference_state_ok = aos_bench.print_compare(
            "reference_state",
            reference_update,
            baseline_update,
        )
        candidate_state_ok = aos_bench.print_compare(
            "candidate_state",
            candidate_update,
            baseline_update,
        )
    if not (baseline_ok and candidate_ok and reference_state_ok and candidate_state_ok):
        return False

    timing_baseline_rhs = torch.empty_like(q_by_node)
    timing_reference_rhs = torch.empty_like(q_by_node)
    timing_candidate_rhs = torch.empty_like(q_by_node)
    timing_baseline_update = q_by_node.clone() if update_state else None
    timing_reference_update = q_by_node.clone() if update_state else None
    timing_candidate_update = q_by_node.clone() if update_state else None

    def baseline_fn():
        aos_bench.launch_baseline(
            inputs,
            timing_baseline_rhs,
            timing_baseline_update,
            coefficient,
        )

    def reference_fn():
        aos_bench.call_tilelang_kernel(
            reference_kernel,
            (
                q_by_node,
                inputs.dr,
                inputs.ds,
                inputs.dt,
                inputs.metric_p_affine,
                inputs.metric_v_affine,
                inputs.surface_by_node,
                timing_reference_rhs,
                timing_reference_update
                if timing_reference_update is not None
                else timing_reference_rhs,
                inputs.coefficient_tensor,
            ),
            skip_validation=not args.no_skip_validation,
        )

    def candidate_fn():
        aos_bench.call_tilelang_kernel(
            candidate_kernel,
            (
                q_by_node,
                inputs.dr,
                inputs.ds,
                inputs.dt,
                inputs.metric_p_affine,
                inputs.metric_v_affine,
                inputs.surface_by_node,
                timing_candidate_rhs,
                timing_candidate_update
                if timing_candidate_update is not None
                else timing_candidate_rhs,
                inputs.coefficient_tensor,
            ),
            skip_validation=not args.no_skip_validation,
        )

    inputs.coefficient_tensor.fill_(coefficient)
    try:
        baseline_ms = aos_bench.time_callable(
            baseline_fn,
            backend=args.profile_backend,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        reference_ms = aos_bench.time_callable(
            reference_fn,
            backend=args.profile_backend,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        candidate_ms = aos_bench.time_callable(
            candidate_fn,
            backend=args.profile_backend,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    except Exception as exc:
        print(f"bench_error={type(exc).__name__}: {exc}")
        return False

    aos_bench.report_perf("baseline", baseline_ms, flops, baseline_memory_bytes)
    aos_bench.report_perf("reference", reference_ms, flops, fused_memory_bytes)
    aos_bench.report_perf("candidate", candidate_ms, flops, candidate_memory_bytes)
    print(f"reference_vs_baseline_speedup={baseline_ms / reference_ms:.6f}")
    print(f"candidate_vs_baseline_speedup={baseline_ms / candidate_ms:.6f}")
    print(f"candidate_vs_reference_speedup={reference_ms / candidate_ms:.6f}")

    if args.profile_target in {"baseline", "all"}:
        aos_bench.run_profile_loop(
            f"baseline_derivative_plus_volume_{config.name}",
            baseline_fn,
            repeat=args.profile_repeat,
        )
    if args.profile_target in {"reference", "both", "all"}:
        aos_bench.run_profile_loop(
            f"reference_tgemm_derivative_volume_{config.name}",
            reference_fn,
            repeat=args.profile_repeat,
        )
    if args.profile_target in {"candidate", "both", "all"}:
        aos_bench.run_profile_loop(
            f"candidate_manual_derivative_volume_{config.name}",
            candidate_fn,
            repeat=args.profile_repeat,
        )
    return candidate_ms < reference_ms


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-name",
        default="scenario1_profile_lc0p20.msh",
        help="Mesh name or path to use for scenario1 setup.",
    )
    parser.add_argument(
        "--mode",
        choices=("rhs", "update"),
        default="update",
        help="Benchmark RHS-only or fused RHS+state-update epilogue.",
    )
    parser.add_argument(
        "--profile-backend",
        choices=("event", "eager", "cudagraph", "cupti"),
        default="cudagraph",
    )
    parser.add_argument(
        "--config",
        default="bp16_be8_t128_qkn_dshared_unroll",
    )
    parser.add_argument(
        "--reference-config",
        default=_REFERENCE_CONFIG_NAME,
        help="Current best T.gemm fused config used as the reference candidate.",
    )
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--save-inputs",
        type=Path,
        default=None,
        help="Save real scenario-derived tensors for later standalone profiling.",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Exit after --save-inputs without compiling or benchmarking kernels.",
    )
    parser.add_argument(
        "--load-inputs",
        type=Path,
        default=None,
        help="Load tensors saved by --save-inputs and skip scenario construction.",
    )
    parser.add_argument(
        "--profile-target",
        choices=("none", "baseline", "reference", "candidate", "both", "all"),
        default="none",
        help="Emit an NVTX-marked loop after timing for external profilers.",
    )
    parser.add_argument(
        "--profile-repeat",
        type=int,
        default=200,
        help="Number of calls inside each NVTX profiling loop.",
    )
    parser.add_argument(
        "--export-sources",
        type=Path,
        default=None,
        help="Export generated TileLang kernel/host sources per config.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--no-skip-validation",
        action="store_true",
        help="Do not pass skip_tensor_validation=True to TileLang calls.",
    )
    parser.add_argument(
        "--debug-traceback",
        action="store_true",
        help="Print full Python tracebacks for TileLang build and benchmark failures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.sweep:
        raise SystemExit(
            "manual microkernel sweep is retired; use a single --config for "
            "correctness/profiling or switch to "
            "benchmarks/tilelang_derivative_volume_aos_autotune.py for active "
            "T.gemm fused-kernel tuning."
        )

    print(f"pid={os.getpid()}")
    print(f"experiment_status={MANUAL_MICROKERNEL_STATUS}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"torch_version={torch.__version__}")
    print(f"torch_version_maca={getattr(torch.version, 'maca', None)}")
    props = torch.cuda.get_device_properties(0)
    print(f"shared_memory_per_block={props.shared_memory_per_block}")
    print(f"warp_size={getattr(props, 'warp_size', 32)}")
    print(f"L2_cache_size={getattr(props, 'L2_cache_size', 0)}")
    print(f"profile_backend={args.profile_backend}")
    print(f"mesh_name={args.mesh_name}")
    print(f"reference_config={args.reference_config}")

    if args.load_inputs is not None:
        inputs = aos_bench.load_inputs(args.load_inputs, torch.device("cuda"))
    else:
        inputs = aos_bench.prepare_simulation(args.mesh_name)
        if args.save_inputs is not None:
            aos_bench.save_inputs(inputs, args.save_inputs)
            if args.save_only:
                return

    print(f"N_tets={inputs.n_tets}")
    print(f"Np={inputs.n_p}")
    print("Nfp=15")
    print(f"N_columns={inputs.n_columns}")
    print(f"coefficient={inputs.coefficient:.17e}")
    print(f"input_source={'file' if args.load_inputs is not None else 'scenario'}")

    update_state = args.mode == "update"
    try:
        reference_kernel = build_tilelang_derivative_volume_aos_kernel(
            inputs.n_tets,
            config_name=args.reference_config,
            update_state=update_state,
        )
        if args.export_sources is not None:
            args.export_sources.mkdir(parents=True, exist_ok=True)
            aos_bench.export_candidate_sources(
                reference_kernel,
                args.export_sources,
                f"reference_{args.reference_config}",
            )
    except Exception as exc:
        if args.debug_traceback:
            traceback.print_exc()
        raise RuntimeError(
            f"failed to build reference TileLang derivative-volume kernel {args.reference_config!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    configs = (args.config,)
    any_faster = False
    for index, config_name in enumerate(configs):
        if index:
            print("---")
        if run_config(args, inputs, config_name, reference_kernel):
            any_faster = True

    print(f"any_candidate_faster_than_reference={int(any_faster)}")


if __name__ == "__main__":
    main()
