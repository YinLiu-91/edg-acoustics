"""Optional TileLang kernels for the EDG lift surface multiply."""

from __future__ import annotations

TILELANG_LIFT_CONFIG_NAME = "bm48_bn64_bk16_s0_t256_fullcol"

_LIFT_M = 35
_LIFT_K = 60
_BLOCK_M = 48
_BLOCK_N = 64
_BLOCK_K = 16
_NUM_STAGES = 0
_THREADS = 256

T = None
_JITTED_FP64_LIFT_MATMUL_TN = None


def _fp64_lift_matmul_tn(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_stages,
    threads,
    policy,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float64),
        B: T.Tensor((K, N), T.float64),
        C: T.Tensor((M, N), T.float64),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float64)
            B_shared = T.alloc_shared((block_N, block_K), T.float64)
            C_local = T.alloc_fragment((block_M, block_N), T.float64)

            T.use_swizzle(panel_size=10, enable=True)
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


def _jitted_fp64_lift_matmul_tn():
    global T, _JITTED_FP64_LIFT_MATMUL_TN

    if _JITTED_FP64_LIFT_MATMUL_TN is not None:
        return _JITTED_FP64_LIFT_MATMUL_TN

    import tilelang
    import tilelang.language as tilelang_language

    T = tilelang_language
    _JITTED_FP64_LIFT_MATMUL_TN = tilelang.jit(_fp64_lift_matmul_tn)
    return _JITTED_FP64_LIFT_MATMUL_TN


def build_tilelang_lift_kernel(n_columns: int):
    """Build the fixed MetaX-tuned fp64 lift GEMM kernel.

    The kernel computes C[35, N] = A[35, 60] @ B[60, N].
    TileLang is imported lazily so normal EDG imports do not require it.
    """
    if n_columns <= 0:
        raise ValueError("TileLang lift kernel requires a positive N dimension.")

    jitted = _jitted_fp64_lift_matmul_tn()
    return jitted(
        _LIFT_M,
        n_columns,
        _LIFT_K,
        _BLOCK_M,
        _BLOCK_N,
        _BLOCK_K,
        _NUM_STAGES,
        _THREADS,
        T.GemmWarpPolicy.FullCol,
    )
