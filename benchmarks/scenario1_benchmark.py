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
sys.path.insert(0, str(TESTS_DIR))


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_fixed_steps(
    steps: int,
    profile: bool,
    cuda_graph: bool,
    cuda_graph_chunk_steps: int,
    record_receivers: bool,
):
    from scenario1_utils import build_scenario1_simulation

    sim = build_scenario1_simulation()

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
    print(f"elapsed_ms={elapsed_ms:.6f}")
    print(f"ms_per_step={elapsed_ms / steps:.6f}")
    print(f"peak_memory_mb={peak_memory_mb:.3f}")
    print(f"dtype={sim.P.dtype}")
    print(f"device={sim.P.device}")
    print(f"cuda_graph={cuda_graph and sim.P.device.type == 'cuda'}")
    print(f"cuda_graph_chunk_steps={cuda_graph_chunk_steps if cuda_graph else 1}")
    print(f"record_receivers={record_receivers}")
    if sim.P.device.type == "cuda":
        print(f"cuda_name={torch.cuda.get_device_name(sim.P.device)}")
    else:
        print(f"cpu_threads={torch.get_num_threads()}")

    if prof is not None:
        sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_by, row_limit=40))


def run_exact_script():
    from scenario1_utils import EXAMPLE_DIR

    command = [sys.executable, str(EXAMPLE_DIR / "main.py")]
    subprocess.run(command, cwd=EXAMPLE_DIR, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--cuda-graph-chunk-steps", type=int, default=1)
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
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["EDG_ACOUSTICS_DEVICE"] = args.device
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)

    if args.exact_script:
        run_exact_script()
    else:
        run_fixed_steps(
            args.steps,
            args.profile,
            args.cuda_graph,
            args.cuda_graph_chunk_steps,
            not args.no_record_receivers,
        )


if __name__ == "__main__":
    main()
