"""TileLang FP64 GEMM kernel for the lift surface operation.

Provides a drop-in replacement for ``torch.mm(lift, flux_by_face)``
where ``lift`` is ``[Np, 4*Nfp]`` and ``flux_by_face`` is ``[4*Nfp, 4*N_tets]``.

The kernel is lazily compiled on first use with the runtime-determined
``N = 4 * N_tets`` dimension.
"""

from __future__ import annotations

import os
import torch

try:
    import tilelang
    import tilelang.language as T
    _HAS_TILELANG = True
except ImportError:
    _HAS_TILELANG = False

# Best config for lift shape (M=35, K=60) from sweeps in /data/mma_tilelang_v6.py:
# bm48_bn64_bk16_s0_t256_fullcol  →  2.0943 TFLOPS, 2.256x speedup vs torch.mm
_LIFT_BLOCK_M = 48
_LIFT_BLOCK_N = 64
_LIFT_BLOCK_K = 16
_LIFT_NUM_STAGES = 0
_LIFT_THREADS = 256
_LIFT_ENABLE_SWIZZLE = True
_LIFT_USE_SHARED_STORE = False

# Cache for the compiled kernel keyed by (M, N, K).
_cache: dict[tuple[int, int, int], object] = {}


def _compile_kernel(M: int, N: int, K: int) -> object:
    """Compile and return the TileLang FP64 GEMM kernel for ``C[M,N] = A[M,K] @ B[K,N]``.

    Uses the pre-benchmarked best config for the lift shape (M=35, K=60).
    The kernel is compiled with ``T.GemmWarpPolicy.FullCol``.
    """
    @tilelang.jit
    def fp64_matmul_tn_inner(
        _M,
        _N,
        _K,
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
            A: T.Tensor((_M, _K), T.float64),
            B: T.Tensor((_K, _N), T.float64),
            C: T.Tensor((_M, _N), T.float64),
        ):
            with T.Kernel(T.ceildiv(_N, block_N), T.ceildiv(_M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), T.float64)
                B_shared = T.alloc_shared((block_N, block_K), T.float64)
                C_local = T.alloc_fragment((block_M, block_N), T.float64)
                if use_shared_store:
                    C_shared = T.alloc_shared((block_M, block_N), T.float64)

                T.use_swizzle(panel_size=10, enable=enable_swizzle)
                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(_K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    for j, kk in T.Parallel(block_N, block_K):
                        k_idx = ko * block_K + kk
                        n_idx = bx * block_N + j
                        B_shared[j, kk] = T.if_then_else(
                            (k_idx < _K) & (n_idx < _N),
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

    return fp64_matmul_tn_inner(
        M, N, K,
        _LIFT_BLOCK_M,
        _LIFT_BLOCK_N,
        _LIFT_BLOCK_K,
        _LIFT_NUM_STAGES,
        _LIFT_THREADS,
        _LIFT_ENABLE_SWIZZLE,
        T.GemmWarpPolicy.FullCol,
        _LIFT_USE_SHARED_STORE,
    )


def get_lift_kernel(M: int, N: int, K: int) -> object | None:
    """Return a compiled TileLang kernel for ``C[M,N] = A[M,K] @ B[K,N]``.

    The kernel is compiled once and cached.  Returns ``None`` if TileLang
    is not installed or CUDA is not available.

    Args:
        M: Number of rows of A and C (``Np``, typically 35 for Nx=4).
        N: Number of columns of B and C (``4 * N_tets``).
        K: Inner dimension (``4 * Nfp``, typically 60 for Nx=4).

    Returns:
        A callable ``kernel(A, B, C)`` that computes ``C = A @ B`` in-place,
        or ``None``.
    """
    if not _HAS_TILELANG:
        return None
    if not torch.cuda.is_available():
        return None

    key = (M, N, K)
    if key not in _cache:
        _cache[key] = _compile_kernel(M, N, K)
    return _cache[key]


__all__ = ["get_lift_kernel"]
