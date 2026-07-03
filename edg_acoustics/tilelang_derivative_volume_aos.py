"""Optional TileLang fused derivative-volume AoS kernels.

The kernel targets the scenario1 profile hot path:

    Dr/Ds/Dt[35, 35] @ Q[35, 4 * N_tets]

It keeps the three derivative products in TileLang fragments and applies the
affine metric volume RHS epilogue before writing the AoS RHS buffer.
"""

from __future__ import annotations

from dataclasses import dataclass


TILELANG_DERIVATIVE_VOLUME_AOS_CONFIG_NAME = (
    "bp16_be16_bn64_bk16_s0_t256_fullcol"
)

_M = 35
_K = 35

T = None
_JITTED_FP64_DERIVATIVE_VOLUME_AOS = None


@dataclass(frozen=True)
class DerivativeVolumeAosConfig:
    name: str
    block_p: int
    block_e: int
    block_k: int
    num_stages: int
    threads: int
    policy: str = "fullcol"

    @property
    def block_n(self) -> int:
        return 4 * self.block_e

    @property
    def explicit_shared_memory_bytes(self) -> int:
        elements = (
            3 * self.block_p * self.block_k
            + self.block_n * self.block_k
            + 3 * self.block_p * self.block_n
        )
        return elements * 8


_CONFIGS: dict[str, DerivativeVolumeAosConfig] = {
    config.name: config
    for config in (
        DerivativeVolumeAosConfig(
            "bp16_be8_bn32_bk16_s0_t128_fullcol", 16, 8, 16, 0, 128
        ),
        DerivativeVolumeAosConfig(
            "bp16_be16_bn64_bk16_s0_t256_fullcol", 16, 16, 16, 0, 256
        ),
        DerivativeVolumeAosConfig(
            "bp16_be16_bn64_bk16_s1_t256_fullcol", 16, 16, 16, 1, 256
        ),
        DerivativeVolumeAosConfig(
            "bp16_be32_bn128_bk16_s0_t256_fullcol", 16, 32, 16, 0, 256
        ),
        DerivativeVolumeAosConfig(
            "bp32_be8_bn32_bk16_s0_t256_fullcol", 32, 8, 16, 0, 256
        ),
        DerivativeVolumeAosConfig(
            "bp32_be16_bn64_bk16_s0_t256_fullcol", 32, 16, 16, 0, 256
        ),
    )
}


def available_config_names() -> tuple[str, ...]:
    return tuple(_CONFIGS)


def get_config(name: str) -> DerivativeVolumeAosConfig:
    try:
        return _CONFIGS[name]
    except KeyError as exc:
        known = ", ".join(available_config_names())
        raise ValueError(f"unknown TileLang derivative-volume AoS config {name!r}; known: {known}") from exc


def _policy_value(policy: str):
    if policy == "fullcol":
        return T.GemmWarpPolicy.FullCol
    if policy == "fullrow":
        return T.GemmWarpPolicy.FullRow
    if policy == "square":
        return T.GemmWarpPolicy.Square
    raise ValueError(f"unknown TileLang GEMM warp policy: {policy}")


def _fp64_derivative_volume_aos(
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
):
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
            T.ceildiv(N, block_n), T.ceildiv(M, block_p), threads=threads
        ) as (bx, by):
            dr_shared = T.alloc_shared((block_p, block_k), T.float64)
            ds_shared = T.alloc_shared((block_p, block_k), T.float64)
            dt_shared = T.alloc_shared((block_p, block_k), T.float64)
            q_shared = T.alloc_shared((block_n, block_k), T.float64)
            acc_r = T.alloc_fragment((block_p, block_n), T.float64)
            acc_s = T.alloc_fragment((block_p, block_n), T.float64)
            acc_t = T.alloc_fragment((block_p, block_n), T.float64)
            acc_r_shared = T.alloc_shared((block_p, block_n), T.float64)
            acc_s_shared = T.alloc_shared((block_p, block_n), T.float64)
            acc_t_shared = T.alloc_shared((block_p, block_n), T.float64)

            T.use_swizzle(panel_size=10, enable=True)
            T.clear(acc_r)
            T.clear(acc_s)
            T.clear(acc_t)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                for i, kk in T.Parallel(block_p, block_k):
                    row = by * block_p + i
                    k_idx = ko * block_k + kk
                    valid = (row < M) & (k_idx < K)
                    dr_shared[i, kk] = T.if_then_else(
                        valid, Dr[row, k_idx], T.float64(0.0)
                    )
                    ds_shared[i, kk] = T.if_then_else(
                        valid, Ds[row, k_idx], T.float64(0.0)
                    )
                    dt_shared[i, kk] = T.if_then_else(
                        valid, Dt[row, k_idx], T.float64(0.0)
                    )

                for j, kk in T.Parallel(block_n, block_k):
                    k_idx = ko * block_k + kk
                    n_idx = bx * block_n + j
                    q_shared[j, kk] = T.if_then_else(
                        (k_idx < K) & (n_idx < N),
                        Q[k_idx, n_idx],
                        T.float64(0.0),
                    )

                T.gemm(
                    dr_shared,
                    q_shared,
                    acc_r,
                    transpose_B=True,
                    policy=policy,
                )
                T.gemm(
                    ds_shared,
                    q_shared,
                    acc_s,
                    transpose_B=True,
                    policy=policy,
                )
                T.gemm(
                    dt_shared,
                    q_shared,
                    acc_t,
                    transpose_B=True,
                    policy=policy,
                )

            T.copy(acc_r, acc_r_shared)
            T.copy(acc_s, acc_s_shared)
            T.copy(acc_t, acc_t_shared)

            coeff = coefficient[0]
            for i, e in T.Parallel(block_p, block_e):
                node = by * block_p + i
                elem = bx * block_e + e
                if node < M and elem < n_tets:
                    c = elem * 4
                    lc = e * 4

                    p_r = acc_r_shared[i, lc]
                    p_s = acc_s_shared[i, lc]
                    p_t = acc_t_shared[i, lc]
                    vx_r = acc_r_shared[i, lc + 1]
                    vx_s = acc_s_shared[i, lc + 1]
                    vx_t = acc_t_shared[i, lc + 1]
                    vy_r = acc_r_shared[i, lc + 2]
                    vy_s = acc_s_shared[i, lc + 2]
                    vy_t = acc_t_shared[i, lc + 2]
                    vz_r = acc_r_shared[i, lc + 3]
                    vz_s = acc_s_shared[i, lc + 3]
                    vz_t = acc_t_shared[i, lc + 3]

                    rhs_p = (
                        metric_p[0, 0, elem] * vx_r
                        + metric_p[1, 0, elem] * vx_s
                        + metric_p[2, 0, elem] * vx_t
                        + metric_p[0, 1, elem] * vy_r
                        + metric_p[1, 1, elem] * vy_s
                        + metric_p[2, 1, elem] * vy_t
                        + metric_p[0, 2, elem] * vz_r
                        + metric_p[1, 2, elem] * vz_s
                        + metric_p[2, 2, elem] * vz_t
                        + surface[node, c]
                    )
                    rhs_vx = (
                        metric_v[0, 0, elem] * p_r
                        + metric_v[1, 0, elem] * p_s
                        + metric_v[2, 0, elem] * p_t
                        + surface[node, c + 1]
                    )
                    rhs_vy = (
                        metric_v[0, 1, elem] * p_r
                        + metric_v[1, 1, elem] * p_s
                        + metric_v[2, 1, elem] * p_t
                        + surface[node, c + 2]
                    )
                    rhs_vz = (
                        metric_v[0, 2, elem] * p_r
                        + metric_v[1, 2, elem] * p_s
                        + metric_v[2, 2, elem] * p_t
                        + surface[node, c + 3]
                    )

                    rhs[node, c] = rhs_p
                    rhs[node, c + 1] = rhs_vx
                    rhs[node, c + 2] = rhs_vy
                    rhs[node, c + 3] = rhs_vz

                    if update_state:
                        q_update[node, c] = q_update[node, c] + coeff * rhs_p
                        q_update[node, c + 1] = (
                            q_update[node, c + 1] + coeff * rhs_vx
                        )
                        q_update[node, c + 2] = (
                            q_update[node, c + 2] + coeff * rhs_vy
                        )
                        q_update[node, c + 3] = (
                            q_update[node, c + 3] + coeff * rhs_vz
                        )

    return main


def _jitted_fp64_derivative_volume_aos():
    global T, _JITTED_FP64_DERIVATIVE_VOLUME_AOS

    if _JITTED_FP64_DERIVATIVE_VOLUME_AOS is not None:
        return _JITTED_FP64_DERIVATIVE_VOLUME_AOS

    import tilelang
    import tilelang.language as tilelang_language

    T = tilelang_language
    _JITTED_FP64_DERIVATIVE_VOLUME_AOS = tilelang.jit(
        _fp64_derivative_volume_aos
    )
    return _JITTED_FP64_DERIVATIVE_VOLUME_AOS


def build_tilelang_derivative_volume_aos_kernel(
    n_tets: int,
    *,
    config_name: str = TILELANG_DERIVATIVE_VOLUME_AOS_CONFIG_NAME,
    update_state: bool = False,
):
    """Build the fixed-shape fused derivative-volume AoS TileLang kernel."""
    if n_tets <= 0:
        raise ValueError("TileLang derivative-volume AoS requires n_tets > 0.")

    config = get_config(config_name)
    jitted = _jitted_fp64_derivative_volume_aos()
    return jitted(
        _M,
        4 * n_tets,
        _K,
        n_tets,
        config.block_p,
        config.block_n,
        config.block_e,
        config.block_k,
        config.num_stages,
        config.threads,
        update_state,
        _policy_value(config.policy),
    )


__all__ = [
    "DerivativeVolumeAosConfig",
    "TILELANG_DERIVATIVE_VOLUME_AOS_CONFIG_NAME",
    "available_config_names",
    "build_tilelang_derivative_volume_aos_kernel",
    "get_config",
]
