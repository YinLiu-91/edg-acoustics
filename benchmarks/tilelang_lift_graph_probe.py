"""Standalone CUDA graph probe for the TileLang lift GEMM.

This isolates the EDG lift multiply shape:

    C[35, N] = A[35, 60] @ B[60, N]

It does not build a mesh or run the EDG timestep. The goal is to check whether
the TileLang kernel launch itself is captured and replayed by PyTorch CUDA graph.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from edg_acoustics.tilelang_lift import (  # noqa: E402
    TILELANG_LIFT_CONFIG_NAME,
    build_tilelang_lift_kernel,
)


def synchronize() -> None:
    torch.cuda.synchronize()


def current_raw_stream() -> int:
    stream = torch.cuda.current_stream()
    if hasattr(stream, "cuda_stream"):
        return int(stream.cuda_stream)
    return int(torch._C._cuda_getCurrentRawStream(torch.cuda.current_device()))


def max_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, bool]:
    diff = (actual - expected).abs()
    max_abs = float(diff.max().item())
    rel = diff / expected.abs().clamp_min(1.0e-300)
    max_rel = float(rel.max().item())
    ok = bool(torch.allclose(actual, expected, rtol=1.0e-10, atol=1.0e-10))
    return max_abs, max_rel, ok


def print_result(prefix: str, actual: torch.Tensor, expected: torch.Tensor) -> bool:
    max_abs, max_rel, ok = max_error(actual, expected)
    print(f"{prefix}_ok={int(ok)}")
    print(f"{prefix}_max_abs={max_abs:.6e}")
    print(f"{prefix}_max_rel={max_rel:.6e}")
    return ok


def graph_probe(name: str, fn, out: torch.Tensor, expected: torch.Tensor) -> None:
    out.zero_()
    synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_error = ""
    try:
        with torch.cuda.graph(graph):
            fn()
    except Exception as exc:
        capture_error = f"{type(exc).__name__}: {exc}"

    print(f"{name}_capture_error={capture_error or 'none'}")
    if capture_error:
        return

    synchronize()
    capture_ok = print_result(f"{name}_after_capture", out, expected)

    out.zero_()
    synchronize()
    replay_error = ""
    try:
        graph.replay()
    except Exception as exc:
        replay_error = f"{type(exc).__name__}: {exc}"

    print(f"{name}_replay_error={replay_error or 'none'}")
    if replay_error:
        return

    synchronize()
    replay_ok = print_result(f"{name}_after_replay", out, expected)
    print(f"{name}_captured_and_replayed={int(capture_ok and replay_ok)}")


def call_tilelang_kernel(kernel, a, b, out, *, skip_validation: bool, stream: int | None = None):
    kwargs = {}
    if stream is not None:
        kwargs["stream"] = stream
    if skip_validation:
        kwargs["skip_tensor_validation"] = True

    try:
        return kernel(a, b, out, **kwargs)
    except TypeError as exc:
        if "skip_tensor_validation" in str(exc) and "skip_tensor_validation" in kwargs:
            kwargs.pop("skip_tensor_validation")
            return kernel(a, b, out, **kwargs)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=181140,
        help="N dimension. scenario1_profile_lc0p20 uses 4*45285=181140.",
    )
    parser.add_argument("--seed", type=int, default=0)
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
    if args.n <= 0:
        raise ValueError("--n must be positive.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    print(f"pid={os.getpid()}")
    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"torch_version={torch.__version__}")
    print(f"torch_version_maca={getattr(torch.version, 'maca', None)}")
    print(f"shape=35x60_60x{args.n}")
    print(f"tilelang_lift_config={TILELANG_LIFT_CONFIG_NAME}")

    a = torch.randn((35, 60), device=device, dtype=torch.float64)
    b = torch.randn((60, args.n), device=device, dtype=torch.float64)
    expected = torch.empty((35, args.n), device=device, dtype=torch.float64)
    out = torch.empty_like(expected)

    torch.mm(a, b, out=expected)
    synchronize()

    print("building_tilelang_kernel=1")
    try:
        kernel = build_tilelang_lift_kernel(args.n)
    except Exception as exc:
        print(f"tilelang_build_error={type(exc).__name__}: {exc}")
        raise SystemExit(2) from exc
    print("building_tilelang_kernel=0")

    def call_tilelang_default_stream():
        call_tilelang_kernel(
            kernel,
            a,
            b,
            out,
            skip_validation=not args.no_skip_validation,
        )

    def call_tilelang_explicit_current_stream():
        call_tilelang_kernel(
            kernel,
            a,
            b,
            out,
            stream=current_raw_stream(),
            skip_validation=not args.no_skip_validation,
        )

    def call_torch_mm():
        torch.mm(a, b, out=out)

    out.zero_()
    call_tilelang_default_stream()
    synchronize()
    print_result("tilelang_eager", out, expected)

    print("probe=torch_mm")
    graph_probe("torch_mm_graph", call_torch_mm, out, expected)

    print("probe=tilelang_default_stream")
    graph_probe("tilelang_default_stream_graph", call_tilelang_default_stream, out, expected)

    print("probe=tilelang_explicit_current_stream")
    graph_probe(
        "tilelang_explicit_current_stream_graph",
        call_tilelang_explicit_current_stream,
        out,
        expected,
    )


if __name__ == "__main__":
    main()