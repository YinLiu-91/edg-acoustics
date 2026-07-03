"""Benchmark TileLang fused derivative-volume AoS kernels.

This isolates the scenario1 profile hot path after lift surface has been
computed:

    baseline: torch.mm(D_merged, Q) + affine AoS volume-surface Triton kernel
    candidate: TileLang fused derivative GEMM + affine volume-surface epilogue
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import triton


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TESTS_DIR))

from edg_acoustics.acoustics_simulation import (  # noqa: E402
    volume_surface_rhs_affine_metric_aos_vector_kernel,
)
from edg_acoustics.tilelang_derivative_volume_aos import (  # noqa: E402
    TILELANG_DERIVATIVE_VOLUME_AOS_CONFIG_NAME,
    available_config_names,
    build_tilelang_derivative_volume_aos_kernel,
    get_config,
)
from scenario1_utils import build_scenario1_simulation, clone_bcvar  # noqa: E402


def synchronize() -> None:
    torch.cuda.synchronize()


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


def call_tilelang_kernel(kernel, args, *, skip_validation: bool) -> None:
    if skip_validation:
        try:
            kernel(*args, skip_tensor_validation=True)
            return
        except TypeError as exc:
            if "skip_tensor_validation" not in str(exc):
                raise
    kernel(*args)


def time_eager(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    synchronize()
    return float(start.elapsed_time(end)) / iterations


def time_cudagraph(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    synchronize()
    return float(start.elapsed_time(end)) / iterations


def time_callable(fn, *, backend: str, warmup: int, iterations: int) -> float:
    if backend == "cudagraph":
        return time_cudagraph(fn, warmup=warmup, iterations=iterations)
    return time_eager(fn, warmup=warmup, iterations=iterations)


def prepare_simulation(mesh_name: str):
    os.environ["EDG_ACOUSTICS_DEVICE"] = "cuda"
    os.environ["EDG_ACOUSTICS_AOS_STATE_LAYOUT"] = "1"
    os.environ["EDG_ACOUSTICS_AFFINE_METRIC_RHS"] = "1"
    os.environ["EDG_ACOUSTICS_TILELANG_DERIVATIVE_VOLUME_AOS"] = "0"
    sim = build_scenario1_simulation(mesh_name=mesh_name, device="cuda")
    if not sim._use_aos_state_layout:
        raise RuntimeError("benchmark requires AoS state layout")
    if not sim._use_affine_metric_rhs:
        raise RuntimeError("benchmark requires affine metric RHS")
    if sim.Np != 35 or sim.Nfp != 15:
        raise RuntimeError(f"benchmark requires Np=35,Nfp=15; got {sim.Np},{sim.Nfp}")

    sim._rhs_operator_packed_pre_lift(sim.Q_flat, clone_bcvar(sim.BC.BCvar))
    sim._compute_lift_surface()
    synchronize()
    return sim


def launch_baseline(
    sim,
    q_by_node: torch.Tensor,
    surface_by_node: torch.Tensor,
    rhs_by_node: torch.Tensor,
    q_update: torch.Tensor | None,
    coefficient: float,
) -> None:
    sim._compute_packed_derivatives(q_by_node)
    total_nodes = sim.Np * sim.N_tets
    block_size = 128
    volume_surface_rhs_affine_metric_aos_vector_kernel[
        (triton.cdiv(total_nodes, block_size),)
    ](
        sim._dQdr_by_node,
        sim._dQds_by_node,
        sim._dQdt_by_node,
        sim._metric_p_affine,
        sim._metric_v_affine,
        surface_by_node,
        rhs_by_node,
        q_update if q_update is not None else rhs_by_node,
        total_nodes,
        sim.N_tets,
        4 * sim.N_tets,
        coefficient,
        q_update is not None,
        BLOCK_SIZE=block_size,
    )


def run_config(args, sim, config_name: str) -> bool:
    config = get_config(config_name)
    shared_limit = torch.cuda.get_device_properties(0).shared_memory_per_block
    update_state = args.mode == "update"
    coefficient = float(sim.time_integrator.taylor_coefficients[0])
    q_by_node = sim.Q_flat
    surface_by_node = sim._surface_by_node

    print(f"config={config.name}")
    print(f"block_p={config.block_p}")
    print(f"block_e={config.block_e}")
    print(f"block_n={config.block_n}")
    print(f"block_k={config.block_k}")
    print(f"num_stages={config.num_stages}")
    print(f"threads={config.threads}")
    print(f"explicit_shared_memory_kib={config.explicit_shared_memory_bytes / 1024:.1f}")
    print(f"mode={args.mode}")
    if config.explicit_shared_memory_bytes > shared_limit:
        print(
            "candidate_skip=shared_memory "
            f"required={config.explicit_shared_memory_bytes} limit={shared_limit}"
        )
        return False

    try:
        kernel = build_tilelang_derivative_volume_aos_kernel(
            sim.N_tets,
            config_name=config.name,
            update_state=update_state,
        )
    except Exception as exc:
        print(f"candidate_build_error={type(exc).__name__}: {exc}")
        return False

    baseline_rhs = torch.empty_like(q_by_node)
    candidate_rhs = torch.empty_like(q_by_node)
    baseline_update = q_by_node.clone() if update_state else None
    candidate_update = q_by_node.clone() if update_state else None

    launch_baseline(
        sim,
        q_by_node,
        surface_by_node,
        baseline_rhs,
        baseline_update,
        coefficient,
    )
    call_tilelang_kernel(
        kernel,
        (
            q_by_node,
            sim.Dr,
            sim.Ds,
            sim.Dt,
            sim._metric_p_affine,
            sim._metric_v_affine,
            surface_by_node,
            candidate_rhs,
            candidate_update if candidate_update is not None else candidate_rhs,
            sim._coefficient_tensor.fill_(coefficient),
        ),
        skip_validation=not args.no_skip_validation,
    )
    synchronize()

    rhs_ok = print_compare("rhs", candidate_rhs, baseline_rhs)
    update_ok = True
    if update_state:
        update_ok = print_compare("state", candidate_update, baseline_update)
    if not (rhs_ok and update_ok):
        return False

    timing_baseline_rhs = torch.empty_like(q_by_node)
    timing_candidate_rhs = torch.empty_like(q_by_node)
    timing_baseline_update = q_by_node.clone() if update_state else None
    timing_candidate_update = q_by_node.clone() if update_state else None

    def baseline_fn():
        launch_baseline(
            sim,
            q_by_node,
            surface_by_node,
            timing_baseline_rhs,
            timing_baseline_update,
            coefficient,
        )

    def candidate_fn():
        call_tilelang_kernel(
            kernel,
            (
                q_by_node,
                sim.Dr,
                sim.Ds,
                sim.Dt,
                sim._metric_p_affine,
                sim._metric_v_affine,
                surface_by_node,
                timing_candidate_rhs,
                timing_candidate_update
                if timing_candidate_update is not None
                else timing_candidate_rhs,
                sim._coefficient_tensor,
            ),
            skip_validation=not args.no_skip_validation,
        )

    sim._coefficient_tensor.fill_(coefficient)
    try:
        baseline_ms = time_callable(
            baseline_fn,
            backend=args.profile_backend,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    except Exception as exc:
        print(f"baseline_bench_error={type(exc).__name__}: {exc}")
        return False
    try:
        candidate_ms = time_callable(
            candidate_fn,
            backend=args.profile_backend,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    except Exception as exc:
        print(f"candidate_bench_error={type(exc).__name__}: {exc}")
        return False

    print(f"baseline_ms={baseline_ms:.6f}")
    print(f"candidate_ms={candidate_ms:.6f}")
    print(f"baseline_us={baseline_ms * 1000.0:.3f}")
    print(f"candidate_us={candidate_ms * 1000.0:.3f}")
    print(f"speedup={baseline_ms / candidate_ms:.6f}")
    return candidate_ms < baseline_ms


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
        choices=("eager", "cudagraph"),
        default="cudagraph",
    )
    parser.add_argument("--config", default=TILELANG_DERIVATIVE_VOLUME_AOS_CONFIG_NAME)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
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
    print(f"profile_backend={args.profile_backend}")
    print(f"mesh_name={args.mesh_name}")

    sim = prepare_simulation(args.mesh_name)
    print(f"N_tets={sim.N_tets}")
    print(f"Np={sim.Np}")
    print(f"Nfp={sim.Nfp}")
    print(f"N_columns={4 * sim.N_tets}")

    configs = available_config_names() if args.sweep else (args.config,)
    any_faster = False
    for index, config_name in enumerate(configs):
        if index:
            print("---")
        if run_config(args, sim, config_name):
            any_faster = True

    print(f"any_candidate_faster={int(any_faster)}")


if __name__ == "__main__":
    main()
