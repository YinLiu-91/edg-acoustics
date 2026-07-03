"""Benchmark scenario 1 time stepping with CUDA-aware timing.

Examples:

    python benchmarks/scenario1_benchmark.py --steps 200
    python benchmarks/scenario1_benchmark.py --steps 50 --profile
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TESTS_DIR))


def format_order_stats(stats) -> str:
    if not stats:
        return "edges=0,max=0,mean=0.000,p50=0.0,p90=0.0,p99=0.0,p999=0.0"
    return (
        f"edges={int(stats.get('edges', 0))},"
        f"max={int(stats.get('max', 0))},"
        f"mean={float(stats.get('mean', 0.0)):.3f},"
        f"p50={float(stats.get('p50', 0.0)):.1f},"
        f"p90={float(stats.get('p90', 0.0)):.1f},"
        f"p99={float(stats.get('p99', 0.0)):.1f},"
        f"p999={float(stats.get('p999', 0.0)):.1f}"
    )


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def print_common_metadata(sim, *, mesh_name: str, record_receivers: bool, cuda_graph: bool):
    tilelang_graph_supported = getattr(
        sim, "_tilelang_lift_graph_capture_supported", None
    )
    if tilelang_graph_supported is None:
        tilelang_graph_supported_text = "unknown"
    else:
        tilelang_graph_supported_text = str(int(tilelang_graph_supported))
    tilelang_segmented_graph_supported = getattr(
        sim, "_tilelang_lift_segmented_graph_supported", None
    )
    if tilelang_segmented_graph_supported is None:
        tilelang_segmented_graph_supported_text = "unknown"
    else:
        tilelang_segmented_graph_supported_text = str(
            int(tilelang_segmented_graph_supported)
        )
    tilelang_fallback_reason = getattr(sim, "_tilelang_lift_fallback_reason", "")
    if not tilelang_fallback_reason:
        tilelang_fallback_reason = "none"
    tilelang_segmented_fallback_reason = getattr(
        sim, "_tilelang_lift_segmented_graph_fallback_reason", ""
    )
    if not tilelang_segmented_fallback_reason:
        tilelang_segmented_fallback_reason = "none"

    print(f"mesh_name={mesh_name}")
    print(f"N_tets={sim.N_tets}")
    print(f"Np={sim.Np}")
    print(f"Nfp={sim.Nfp}")
    print(f"dt={sim.time_integrator.dt}")
    total_face_nodes = 4 * sim.Nfp * sim.N_tets
    boundary_face_nodes = sum(node["map"].numel() for node in sim.BCnode)
    interior_face_nodes = total_face_nodes - boundary_face_nodes
    paired_interior_face_nodes = sim._interior_pair_offsets.numel()
    print(f"total_face_nodes={total_face_nodes}")
    print(f"boundary_face_nodes={boundary_face_nodes}")
    print(f"interior_face_nodes={interior_face_nodes}")
    print(f"unique_interior_face_nodes={interior_face_nodes // 2}")
    print(f"paired_interior_face_nodes={paired_interior_face_nodes}")
    print(f"affine_face_geometry={int(sim._face_geometry_is_affine)}")
    print(f"affine_face_delta={getattr(sim, '_face_geometry_delta', float('nan')):.6e}")
    print(
        f"affine_metric_geometry={int(getattr(sim, '_metric_geometry_is_affine', False))}"
    )
    print(
        f"affine_metric_delta={getattr(sim, '_metric_affine_delta', float('nan')):.6e}"
    )
    print(f"interior_face_order={getattr(sim, '_interior_face_order_method', 'natural')}")
    print(
        f"face_order_tile_size={getattr(sim, '_interior_face_order_tile_size', -1)}"
    )
    print(
        f"face_order_block_size={getattr(sim, '_interior_face_order_block_size', -1)}"
    )
    print(f"face_order_enabled={int(getattr(sim, '_use_ordered_aos_flux', False))}")
    print(
        f"face_order_storage={getattr(sim, '_interior_face_order_storage', 'disabled')}"
    )
    print(
        f"ordered_aos_variant={getattr(sim, '_ordered_aos_variant_label', lambda: 'base')()}"
    )
    print(
        "ordered_aos_state_load_mode="
        f"{getattr(sim, '_ordered_aos_state_load_mode', 'scalar')}"
    )
    print(
        "face_order_delta_before="
        f"{format_order_stats(getattr(sim, '_interior_face_order_stats_before', None))}"
    )
    print(
        "face_order_delta_after="
        f"{format_order_stats(getattr(sim, '_interior_face_order_stats_after', None))}"
    )
    print(
        "face_order_work_offset_delta_after="
        f"{format_order_stats(getattr(sim, '_interior_face_work_offset_stats_after', None))}"
    )
    print(f"dtype={sim.P.dtype}")
    print(f"device={sim.P.device}")
    print(f"cuda_graph={cuda_graph and sim.P.device.type == 'cuda'}")
    print(
        "cuda_graph_mode="
        f"{getattr(sim, 'last_time_integration_cuda_graph_mode', 'not_run')}"
    )
    print(f"record_receivers={record_receivers}")
    print(
        "tilelang_lift_enabled="
        f"{int(getattr(sim, '_use_tilelang_lift_surface', False))}"
    )
    print(
        "tilelang_lift_config="
        f"{getattr(sim, '_tilelang_lift_config', 'disabled')}"
    )
    print(
        "tilelang_lift_graph_capture_supported="
        f"{tilelang_graph_supported_text}"
    )
    print(
        "tilelang_lift_segmented_graph_mode="
        f"{getattr(sim, '_tilelang_segmented_graph_mode', 'auto')}"
    )
    print(
        "tilelang_lift_segmented_graph_supported="
        f"{tilelang_segmented_graph_supported_text}"
    )
    print(
        "tilelang_lift_segmented_graph_fallback_reason="
        f"{tilelang_segmented_fallback_reason}"
    )
    print(f"tilelang_lift_fallback_reason={tilelang_fallback_reason}")
    print(
        "optimizations="
        f"volume_rhs:{int(sim._use_triton_volume_rhs)},"
        f"interior_flux:{int(sim._use_triton_interior_flux)},"
        f"boundary_ri:{int(sim._use_triton_boundary_ri)},"
        f"boundary_ade:{int(sim._use_triton_boundary_ade)},"
        f"batched_derivatives:{int(sim._use_batched_derivatives)},"
        f"volume_surface_rhs:{int(sim._use_triton_volume_surface_rhs)},"
        f"scaled_flux:{int(sim._use_scaled_flux_kernels)},"
        f"fused_state_accumulation:{int(sim._use_fused_state_accumulation)},"
        f"derivative_volume:{int(sim._use_triton_derivative_volume)},"
        f"lift_surface:{int(sim._use_triton_lift_surface)},"
        f"tilelang_lift:{int(getattr(sim, '_use_tilelang_lift_surface', False))},"
        f"compact_flux:{int(sim._use_compact_flux_coefficients)},"
        f"aos_state_layout:{int(getattr(sim, '_use_aos_state_layout', False))},"
        f"aos_volume_vector_loads:{int(getattr(sim, '_use_aos_volume_vector_loads', False))},"
        f"ordered_aos_state_vec4:{int(getattr(sim, '_use_ordered_aos_state_vec4', False))},"
        f"affine_metric_rhs:{int(getattr(sim, '_use_affine_metric_rhs', False))},"
        f"paired_interior_flux:{int(sim._use_paired_interior_flux)},"
        f"merged_derivatives:{int(sim._use_merged_derivatives)}"
    )
    if sim.P.device.type == "cuda":
        print(f"cuda_name={torch.cuda.get_device_name(sim.P.device)}")
    else:
        print(f"cpu_threads={torch.get_num_threads()}")


def run_fixed_steps(
    steps: int,
    mesh_name: str,
    profile: bool,
    profile_row_limit: int,
    cuda_graph: bool,
    cuda_graph_chunk_steps: int,
    record_receivers: bool,
):
    from scenario1_utils import build_scenario1_simulation

    sim = build_scenario1_simulation(mesh_name=mesh_name)

    # Warm up lazy kernels and allocator paths.
    sim.time_integration(
        n_time_steps=min(5, steps),
        progress=False,
        use_cuda_graph=cuda_graph,
        cuda_graph_chunk_steps=cuda_graph_chunk_steps,
        record_receivers=record_receivers,
    )
    synchronize()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        wall_start = time.perf_counter()

    if profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            sim.time_integration(
                n_time_steps=steps,
                progress=False,
                use_cuda_graph=cuda_graph,
                cuda_graph_chunk_steps=cuda_graph_chunk_steps,
                record_receivers=record_receivers,
            )
    else:
        prof = None
        sim.time_integration(
            n_time_steps=steps,
            progress=False,
            use_cuda_graph=cuda_graph,
            cuda_graph_chunk_steps=cuda_graph_chunk_steps,
            record_receivers=record_receivers,
        )

    if torch.cuda.is_available():
        end.record()
        synchronize()
        elapsed_ms = start.elapsed_time(end)
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        elapsed_ms = (time.perf_counter() - wall_start) * 1000
        peak_memory_mb = 0.0

    print(f"steps={steps}")
    print("mode=fixed_steps")
    print(f"elapsed_ms={elapsed_ms:.6f}")
    print(f"ms_per_step={elapsed_ms / steps:.6f}")
    print(f"peak_memory_mb={peak_memory_mb:.3f}")
    print(f"steps={steps}")
    print(f"cuda_graph_chunk_steps={cuda_graph_chunk_steps if cuda_graph else 1}")
    print_common_metadata(
        sim,
        mesh_name=mesh_name,
        record_receivers=record_receivers,
        cuda_graph=cuda_graph,
    )

    if prof is not None:
        sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_by, row_limit=profile_row_limit))


def run_real_case(
    total_time: float,
    mesh_name: str,
    cuda_graph: bool,
    record_receivers: bool,
):
    from scenario1_utils import build_scenario1_simulation

    sim = build_scenario1_simulation(mesh_name=mesh_name)
    if sim.P.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sim.time_integration(
        total_time=total_time,
        progress=False,
        use_cuda_graph=cuda_graph,
        record_receivers=record_receivers,
        synchronize_timing=True,
    )
    elapsed_s = sim.last_time_integration_elapsed_s
    steps = sim.last_time_integration_steps
    print("mode=real_case")
    print(f"total_time={sim.last_time_integration_total_time}")
    print(f"steps={steps}")
    print(f"elapsed_s={elapsed_s:.6f}")
    print(f"elapsed_ms={elapsed_s * 1000.0:.6f}")
    ms_per_step = float("nan") if steps == 0 else (elapsed_s * 1000.0) / steps
    print(f"ms_per_step={ms_per_step:.6f}")
    print(
        "peak_memory_mb=0.000"
        if sim.P.device.type != "cuda"
        else f"peak_memory_mb={torch.cuda.max_memory_allocated() / 1024**2:.3f}"
    )
    print(f"cuda_graph_chunk_steps={1}")
    print_common_metadata(
        sim,
        mesh_name=mesh_name,
        record_receivers=record_receivers,
        cuda_graph=cuda_graph,
    )


def run_exact_script():
    from scenario1_utils import EXAMPLE_DIR

    command = [sys.executable, str(EXAMPLE_DIR / "main.py")]
    subprocess.run(command, cwd=EXAMPLE_DIR, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--mesh-name", default="scenario1_coarser.msh")
    parser.add_argument(
        "--real-case-total-time",
        type=float,
        default=None,
        help="Run full time_integration(total_time=...) instead of fixed steps.",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-row-limit", type=int, default=40)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--cuda-graph-chunk-steps", type=int, default=1)
    parser.add_argument(
        "--log-cuda-graph-selection",
        action="store_true",
        help="Print CUDA graph selection decisions during graph capture.",
    )
    parser.add_argument(
        "--no-record-receivers",
        action="store_true",
        help="Skip per-step receiver sampling to measure solver-only time.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to use for this fresh benchmark process.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Set torch CPU thread count for CPU benchmarks.",
    )
    parser.add_argument(
        "--exact-script",
        action="store_true",
        help="Run examples/scenario1/main.py exactly instead of fixed-step benchmark.",
    )
    parser.add_argument("--disable-triton-volume-rhs", action="store_true")
    parser.add_argument("--disable-triton-interior-flux", action="store_true")
    parser.add_argument("--disable-triton-boundary-ri", action="store_true")
    parser.add_argument("--disable-triton-boundary-ade", action="store_true")
    parser.add_argument("--disable-batched-derivatives", action="store_true")
    parser.add_argument("--disable-triton-volume-surface-rhs", action="store_true")
    parser.add_argument("--disable-scaled-flux-kernels", action="store_true")
    fused_state_group = parser.add_mutually_exclusive_group()
    fused_state_group.add_argument(
        "--enable-fused-state-accumulation", action="store_true"
    )
    fused_state_group.add_argument(
        "--disable-fused-state-accumulation", action="store_true"
    )
    parser.add_argument("--enable-triton-derivative-volume", action="store_true")
    parser.add_argument("--enable-triton-lift-surface", action="store_true")
    tilelang_lift_group = parser.add_mutually_exclusive_group()
    tilelang_lift_group.add_argument("--enable-tilelang-lift", action="store_true")
    tilelang_lift_group.add_argument("--disable-tilelang-lift", action="store_true")
    segmented_graph_group = parser.add_mutually_exclusive_group()
    segmented_graph_group.add_argument(
        "--enable-tilelang-segmented-graph",
        action="store_true",
        help="Force segmented CUDA graph around TileLang lift.",
    )
    segmented_graph_group.add_argument(
        "--disable-tilelang-segmented-graph",
        action="store_true",
        help="Disable segmented CUDA graph fallback around TileLang lift.",
    )
    parser.add_argument("--disable-compact-flux-coefficients", action="store_true")
    parser.add_argument("--disable-paired-interior-flux", action="store_true")
    merged_derivatives_group = parser.add_mutually_exclusive_group()
    merged_derivatives_group.add_argument(
        "--enable-merged-derivatives", action="store_true"
    )
    merged_derivatives_group.add_argument(
        "--disable-merged-derivatives", action="store_true"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["EDG_ACOUSTICS_DEVICE"] = args.device
    if args.disable_triton_volume_rhs:
        os.environ["EDG_ACOUSTICS_TRITON_VOLUME_RHS"] = "0"
    if args.disable_triton_interior_flux:
        os.environ["EDG_ACOUSTICS_TRITON_INTERIOR_FLUX"] = "0"
    if args.disable_triton_boundary_ri:
        os.environ["EDG_ACOUSTICS_TRITON_BOUNDARY_RI"] = "0"
    if args.disable_triton_boundary_ade:
        os.environ["EDG_ACOUSTICS_TRITON_BOUNDARY_ADE"] = "0"
    if args.disable_batched_derivatives:
        os.environ["EDG_ACOUSTICS_BATCHED_DERIVATIVES"] = "0"
    if args.disable_triton_volume_surface_rhs:
        os.environ["EDG_ACOUSTICS_TRITON_VOLUME_SURFACE_RHS"] = "0"
    if args.disable_scaled_flux_kernels:
        os.environ["EDG_ACOUSTICS_SCALED_FLUX_KERNELS"] = "0"
    if args.enable_fused_state_accumulation:
        os.environ["EDG_ACOUSTICS_FUSED_STATE_ACCUMULATION"] = "1"
    if args.disable_fused_state_accumulation:
        os.environ["EDG_ACOUSTICS_FUSED_STATE_ACCUMULATION"] = "0"
    if args.enable_triton_derivative_volume:
        os.environ["EDG_ACOUSTICS_TRITON_DERIVATIVE_VOLUME"] = "1"
    if args.enable_triton_lift_surface:
        os.environ["EDG_ACOUSTICS_TRITON_LIFT_SURFACE"] = "1"
    if args.enable_tilelang_lift:
        os.environ["EDG_ACOUSTICS_TILELANG_LIFT"] = "1"
    if args.disable_tilelang_lift:
        os.environ["EDG_ACOUSTICS_TILELANG_LIFT"] = "0"
    if args.enable_tilelang_segmented_graph:
        os.environ["EDG_ACOUSTICS_TILELANG_SEGMENTED_CUDA_GRAPH"] = "1"
    if args.disable_tilelang_segmented_graph:
        os.environ["EDG_ACOUSTICS_TILELANG_SEGMENTED_CUDA_GRAPH"] = "0"
    if args.disable_compact_flux_coefficients:
        os.environ["EDG_ACOUSTICS_COMPACT_FLUX_COEFFICIENTS"] = "0"
    if args.disable_paired_interior_flux:
        os.environ["EDG_ACOUSTICS_PAIRED_INTERIOR_FLUX"] = "0"
    if args.enable_merged_derivatives:
        os.environ["EDG_ACOUSTICS_MERGED_DERIVATIVES"] = "1"
    if args.disable_merged_derivatives:
        os.environ["EDG_ACOUSTICS_MERGED_DERIVATIVES"] = "0"
    if args.log_cuda_graph_selection:
        os.environ["EDG_ACOUSTICS_CUDA_GRAPH_SELECTION_LOG"] = "1"
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)

    if args.exact_script:
        run_exact_script()
    elif args.real_case_total_time is not None:
        run_real_case(
            args.real_case_total_time,
            args.mesh_name,
            args.cuda_graph,
            not args.no_record_receivers,
        )
    else:
        run_fixed_steps(
            args.steps,
            args.mesh_name,
            args.profile,
            args.profile_row_limit,
            args.cuda_graph,
            args.cuda_graph_chunk_steps,
            not args.no_record_receivers,
        )


if __name__ == "__main__":
    main()
