"""Offline autotune driver for the TileLang fused derivative-volume AoS kernel.

This benchmark keeps runtime selection conservative: tuning happens here against
real scenario-derived inputs, and the chosen config can then be benchmarked or
manually promoted into the forced runtime path.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

import tilelang_derivative_volume_aos_benchmark as aos_bench  # noqa: E402
import edg_acoustics.tilelang_derivative_volume_aos as tl_aos  # noqa: E402


_TARGET_VARIANTS = ("copy_shared", "direct_epilogue", "merged3")
_TARGET_POLICIES = ("fullcol", "square")
_K_M_PER_WARP = 16
_K_N_PER_WARP = 16


@dataclass(frozen=True)
class AutotuneCandidate:
    name: str
    block_p: int
    block_e: int
    block_k: int
    num_stages: int
    threads: int
    policy: str
    variant: str

    @property
    def block_n(self) -> int:
        return 4 * self.block_e

    @property
    def gemm_m(self) -> int:
        if self.variant == "merged3":
            return 3 * self.block_p
        return self.block_p

    @property
    def explicit_shared_memory_bytes(self) -> int:
        config = tl_aos.DerivativeVolumeAosConfig(
            name=self.name,
            block_p=self.block_p,
            block_e=self.block_e,
            block_k=self.block_k,
            num_stages=self.num_stages,
            threads=self.threads,
            policy=self.policy,
            variant=self.variant,
        )
        return config.explicit_shared_memory_bytes

    def to_autotune_dict(self, T) -> dict[str, object]:
        return {
            "block_p": self.block_p,
            "block_n": self.block_n,
            "block_e": self.block_e,
            "block_k": self.block_k,
            "num_stages": self.num_stages,
            "threads": self.threads,
            "policy": policy_value(T, self.policy),
            "variant": tl_aos._VARIANT_CODES[self.variant],
        }


def policy_value(T, policy: str):
    if policy == "fullcol":
        return T.GemmWarpPolicy.FullCol
    if policy == "fullrow":
        return T.GemmWarpPolicy.FullRow
    if policy == "square":
        return T.GemmWarpPolicy.Square
    raise ValueError(f"unknown GEMM warp policy: {policy}")


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _warp_partition_supported(config: AutotuneCandidate, warp_size: int) -> bool:
    if config.threads % warp_size != 0:
        return False
    num_warps = config.threads // warp_size
    if num_warps < 1:
        return False

    block_m = config.gemm_m
    block_n = config.block_n

    if config.policy == "fullrow":
        m_warp = num_warps
        n_warp = 1
        if block_m % (m_warp * _K_M_PER_WARP) != 0:
            m_warp = block_m // _K_M_PER_WARP
            if m_warp < 1:
                return False
            n_warp = num_warps // m_warp
            if n_warp == 0:
                n_warp = 1
        return m_warp * n_warp == num_warps

    if config.policy == "fullcol":
        m_warp = 1
        n_warp = num_warps
        if block_n % (n_warp * _K_N_PER_WARP) != 0:
            n_warp = block_n // _K_N_PER_WARP
            if n_warp < 1:
                return False
            m_warp = num_warps // n_warp
            if m_warp == 0:
                m_warp = 1
        return m_warp * n_warp == num_warps

    if config.policy != "square":
        return False

    max_m_warps = block_m // _K_M_PER_WARP
    for m_warp in range(1, min(max_m_warps, num_warps) + 1):
        n_warp = num_warps // m_warp
        if m_warp * n_warp != num_warps:
            continue
        if block_m < m_warp * _K_M_PER_WARP:
            continue
        if block_n < n_warp * _K_N_PER_WARP:
            continue
        return True
    return False


def _supported_candidate(
    config: AutotuneCandidate,
    *,
    shared_memory_limit: int,
    warp_size: int,
) -> bool:
    if config.block_k % 4 != 0:
        return False
    if config.gemm_m % _K_M_PER_WARP != 0:
        return False
    if config.block_n % _K_N_PER_WARP != 0:
        return False
    if config.explicit_shared_memory_bytes > shared_memory_limit:
        return False
    return _warp_partition_supported(config, warp_size)


def available_autotune_candidates(
    *,
    variant_names: tuple[str, ...],
    policy_names: tuple[str, ...],
    shared_memory_limit: int,
    warp_size: int,
) -> tuple[AutotuneCandidate, ...]:
    selected_variants = set(variant_names)
    selected_policies = set(policy_names)

    base_shapes: dict[tuple[int, int, int, int, int, str], tl_aos.DerivativeVolumeAosConfig] = {}
    for config_name in tl_aos.available_config_names():
        config = tl_aos.get_config(config_name)
        if config.variant not in selected_variants:
            continue
        key = (
            config.block_p,
            config.block_e,
            config.block_k,
            config.num_stages,
            config.threads,
            config.variant,
        )
        base_shapes.setdefault(key, config)

    candidates: list[AutotuneCandidate] = []
    for key, base in sorted(base_shapes.items()):
        for policy in policy_names:
            if policy not in selected_policies:
                continue
            candidate = AutotuneCandidate(
                name=(
                    f"bp{base.block_p}_be{base.block_e}_bn{base.block_n}_"
                    f"bk{base.block_k}_s{base.num_stages}_t{base.threads}_"
                    f"{policy}_{base.variant}"
                ),
                block_p=base.block_p,
                block_e=base.block_e,
                block_k=base.block_k,
                num_stages=base.num_stages,
                threads=base.threads,
                policy=policy,
                variant=base.variant,
            )
            if _supported_candidate(
                candidate,
                shared_memory_limit=shared_memory_limit,
                warp_size=warp_size,
            ):
                candidates.append(candidate)

    return tuple(candidates)


def _build_autotune_impl(
    *,
    configs: tuple[AutotuneCandidate, ...],
    tune_warmup: int,
    tune_rep: int,
):
    if not configs:
        raise RuntimeError("no supported configs available for TileLang autotune")

    import tilelang
    import tilelang.language as T
    from tilelang.autotuner import autotune

    tl_aos.T = T
    default_config = configs[0]
    default_policy = policy_value(T, default_config.policy)
    default_variant = tl_aos._VARIANT_CODES[default_config.variant]
    autotune_configs = [candidate.to_autotune_dict(T) for candidate in configs]

    @autotune(
        configs=autotune_configs,
        warmup=tune_warmup,
        rep=tune_rep,
        skip_check=True,
        cache_input_tensors=True,
    )
    @tilelang.jit(out_idx=[7, 8])
    def derivative_volume_aos(
        M,
        N,
        K,
        n_tets,
        update_state,
        block_p=default_config.block_p,
        block_n=default_config.block_n,
        block_e=default_config.block_e,
        block_k=default_config.block_k,
        num_stages=default_config.num_stages,
        threads=default_config.threads,
        policy=default_policy,
        variant=default_variant,
    ):
        return tl_aos._fp64_derivative_volume_aos(
            M,
            N,
            K,
            n_tets,
            block_p,
            block_n,
            block_e,
            block_k,
            num_stages,
            threads,
            update_state,
            policy,
            variant,
        )

    return derivative_volume_aos


def _benchmark_kernel(args, inputs: aos_bench.KernelInputs, kernel) -> None:
    update_state = args.mode == "update"
    coefficient = inputs.coefficient
    q_by_node = inputs.q_by_node
    flops = aos_bench.logical_flops(inputs, update_state=update_state)
    baseline_memory_bytes = aos_bench.logical_memory_bytes(
        inputs,
        update_state=update_state,
        candidate=False,
    )
    candidate_memory_bytes = aos_bench.logical_memory_bytes(
        inputs,
        update_state=update_state,
        candidate=True,
    )

    baseline_rhs = torch.empty_like(q_by_node)
    candidate_rhs = torch.empty_like(q_by_node)
    baseline_update = q_by_node.clone() if update_state else None
    candidate_update = q_by_node.clone() if update_state else None

    aos_bench.launch_baseline(
        inputs,
        baseline_rhs,
        baseline_update,
        coefficient,
    )
    aos_bench.call_tilelang_kernel(
        kernel,
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

    rhs_ok = aos_bench.print_compare("rhs", candidate_rhs, baseline_rhs)
    state_ok = True
    if update_state:
        state_ok = aos_bench.print_compare("state", candidate_update, baseline_update)
    if not (rhs_ok and state_ok):
        raise RuntimeError("best autotuned kernel failed correctness validation")

    timing_baseline_rhs = torch.empty_like(q_by_node)
    timing_candidate_rhs = torch.empty_like(q_by_node)
    timing_baseline_update = q_by_node.clone() if update_state else None
    timing_candidate_update = q_by_node.clone() if update_state else None

    def baseline_fn():
        aos_bench.launch_baseline(
            inputs,
            timing_baseline_rhs,
            timing_baseline_update,
            coefficient,
        )

    def candidate_fn():
        aos_bench.call_tilelang_kernel(
            kernel,
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
    baseline_ms = aos_bench.time_callable(
        baseline_fn,
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

    aos_bench.report_perf("baseline", baseline_ms, flops, baseline_memory_bytes)
    aos_bench.report_perf("candidate", candidate_ms, flops, candidate_memory_bytes)
    print(f"speedup={baseline_ms / candidate_ms:.6f}")

    if args.profile_target in {"baseline", "both"}:
        aos_bench.run_profile_loop(
            "baseline_derivative_plus_volume_autotuned",
            baseline_fn,
            repeat=args.profile_repeat,
        )
    if args.profile_target in {"candidate", "both"}:
        aos_bench.run_profile_loop(
            "candidate_tilelang_derivative_volume_aos_autotuned",
            candidate_fn,
            repeat=args.profile_repeat,
        )


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
        "--variants",
        default="copy_shared,direct_epilogue,merged3",
        help="Comma-separated derivative-volume variants to tune.",
    )
    parser.add_argument(
        "--policies",
        default="fullcol,square",
        help="Comma-separated T.gemm warp policies to tune.",
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="Print the candidate config list after device filtering and exit.",
    )
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
        choices=("none", "baseline", "candidate", "both"),
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
        help="Export generated TileLang kernel/host sources for the best config.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations for the final baseline-vs-best benchmark.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Measured iterations for the final baseline-vs-best benchmark.",
    )
    parser.add_argument(
        "--tune-warmup",
        type=int,
        default=5,
        help="Warmup iterations used inside TileLang autotune.",
    )
    parser.add_argument(
        "--tune-rep",
        type=int,
        default=20,
        help="Measured repetitions used inside TileLang autotune.",
    )
    parser.add_argument(
        "--no-skip-validation",
        action="store_true",
        help="Do not pass skip_tensor_validation=True to TileLang calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print(f"pid={os.getpid()}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"torch_version={torch.__version__}")
    print(f"torch_version_maca={getattr(torch.version, 'maca', None)}")
    props = torch.cuda.get_device_properties(0)
    print(f"shared_memory_per_block={props.shared_memory_per_block}")
    print(f"warp_size={getattr(props, 'warp_size', 32)}")
    print(f"L2_cache_size={getattr(props, 'L2_cache_size', 0)}")
    print(f"profile_backend={args.profile_backend}")
    print(f"mesh_name={args.mesh_name}")
    print(f"variants={args.variants}")
    print(f"policies={args.policies}")

    candidates = available_autotune_candidates(
        variant_names=_split_csv(args.variants),
        policy_names=_split_csv(args.policies),
        shared_memory_limit=props.shared_memory_per_block,
        warp_size=getattr(props, "warp_size", 32),
    )
    print(f"candidate_count={len(candidates)}")
    for candidate in candidates:
        print(
            "candidate="
            f"{candidate.name},shared_kib={candidate.explicit_shared_memory_bytes / 1024:.1f}"
        )
    if args.list_candidates:
        return
    if not candidates:
        raise RuntimeError("no supported TileLang derivative-volume autotune candidates")

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

    impl = _build_autotune_impl(
        configs=candidates,
        tune_warmup=args.tune_warmup,
        tune_rep=args.tune_rep,
    )

    from tilelang.autotuner import set_autotune_inputs

    update_state = args.mode == "update"
    with set_autotune_inputs(
        inputs.q_by_node,
        inputs.dr,
        inputs.ds,
        inputs.dt,
        inputs.metric_p_affine,
        inputs.metric_v_affine,
        inputs.surface_by_node,
        inputs.coefficient_tensor,
    ):
        best_result = impl(
            inputs.n_p,
            inputs.n_columns,
            inputs.n_p,
            inputs.n_tets,
            update_state,
        )

    best_kernel = best_result.kernel
    if best_kernel is None:
        raise RuntimeError("TileLang autotune did not return a compiled kernel")

    print(f"autotune_best_latency_ms={best_result.latency:.6f}")
    print(f"autotune_best_config={best_result.config}")
    print(f"autotune_ref_latency_ms={best_result.ref_latency}")
    if args.export_sources is not None:
        args.export_sources.mkdir(parents=True, exist_ok=True)
        aos_bench.export_candidate_sources(
            best_kernel,
            args.export_sources,
            "autotuned_best_derivative_volume_aos",
        )

    _benchmark_kernel(args, inputs, best_kernel)


if __name__ == "__main__":
    main()
