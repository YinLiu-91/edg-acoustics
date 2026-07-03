"""Optional TileLang fused derivative-volume AoS kernels.

The kernel targets the scenario1 profile hot path:

    Dr/Ds/Dt[35, 35] @ Q[35, 4 * N_tets]

It keeps the three derivative products in TileLang fragments and applies the
affine metric volume RHS epilogue before writing the AoS RHS buffer.
"""

from __future__ import annotations

from dataclasses import dataclass


TILELANG_DERIVATIVE_VOLUME_AOS_CONFIG_NAME = (
    "bp16_be8_bn32_bk16_s0_t128_fullcol"
)

_M = 35
_K = 35
_VARIANT_COPY_SHARED = 0
_VARIANT_DIRECT_EPILOGUE = 1
_VARIANT_FIELD_FRAGMENTS = 2
_VARIANT_FIELD_PAIRS = 3
_VARIANT_CODES = {
    "copy_shared": _VARIANT_COPY_SHARED,
    "direct_epilogue": _VARIANT_DIRECT_EPILOGUE,
    "field_fragments": _VARIANT_FIELD_FRAGMENTS,
    "field_pairs": _VARIANT_FIELD_PAIRS,
}

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
    variant: str = "copy_shared"

    @property
    def block_n(self) -> int:
        return 4 * self.block_e

    @property
    def field_n(self) -> int:
        return max(self.block_e, 16)

    @property
    def pair_n(self) -> int:
        return max(2 * self.block_e, 16)

    @property
    def explicit_shared_memory_bytes(self) -> int:
        if self.variant == "field_fragments":
            elements = 3 * self.block_p * self.block_k + self.field_n * self.block_k
        elif self.variant == "field_pairs":
            elements = (
                3 * self.block_p * self.block_k
                + self.pair_n * self.block_k
                + 6 * self.block_p * self.pair_n
            )
        elif self.variant == "direct_epilogue":
            elements = 3 * self.block_p * self.block_k + self.block_n * self.block_k
        else:
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
            "bp16_be4_bn16_bk16_s0_t64_fullcol", 16, 4, 16, 0, 64
        ),
        DerivativeVolumeAosConfig(
            "bp16_be4_bn16_bk16_s0_t128_fullcol", 16, 4, 16, 0, 128
        ),
        DerivativeVolumeAosConfig(
            "bp16_be8_bn32_bk16_s0_t64_fullcol", 16, 8, 16, 0, 64
        ),
        DerivativeVolumeAosConfig(
            "bp16_be8_bn32_bk16_s0_t128_fullcol", 16, 8, 16, 0, 128
        ),
        DerivativeVolumeAosConfig(
            "bp16_be8_bn32_bk16_s1_t128_fullcol", 16, 8, 16, 1, 128
        ),
        DerivativeVolumeAosConfig(
            "bp16_be12_bn48_bk16_s0_t128_fullcol", 16, 12, 16, 0, 128
        ),
        DerivativeVolumeAosConfig(
            "bp16_be12_bn48_bk16_s0_t256_fullcol", 16, 12, 16, 0, 256
        ),
        DerivativeVolumeAosConfig(
            "bp16_be16_bn64_bk16_s0_t128_fullcol", 16, 16, 16, 0, 128
        ),
        DerivativeVolumeAosConfig(
            "bp16_be16_bn64_bk16_s0_t256_fullcol", 16, 16, 16, 0, 256
        ),
        DerivativeVolumeAosConfig(
            "bp16_be16_bn64_bk16_s1_t256_fullcol", 16, 16, 16, 1, 256
        ),
        DerivativeVolumeAosConfig(
            "bp16_be8_bn32_bk32_s0_t128_fullcol", 16, 8, 32, 0, 128
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
        DerivativeVolumeAosConfig(
            "bp16_be8_bn32_bk16_s0_t128_fullcol_direct",
            16,
            8,
            16,
            0,
            128,
            variant="direct_epilogue",
        ),
        DerivativeVolumeAosConfig(
            "bp16_be8_bn32_bk16_s0_t64_fullcol_fieldfrag",
            16,
            8,
            16,
            0,
            64,
            variant="field_fragments",
        ),
        DerivativeVolumeAosConfig(
            "bp16_be8_bn32_bk16_s0_t128_fullcol_fieldfrag",
            16,
            8,
            16,
            0,
            128,
            variant="field_fragments",
        ),
        DerivativeVolumeAosConfig(
            "bp16_be16_bn64_bk16_s0_t64_fullcol_fieldfrag",
            16,
            16,
            16,
            0,
            64,
            variant="field_fragments",
        ),
        DerivativeVolumeAosConfig(
            "bp16_be8_bn32_bk16_s0_t64_fullcol_fieldpair",
            16,
            8,
            16,
            0,
            64,
            variant="field_pairs",
        ),
        DerivativeVolumeAosConfig(
            "bp16_be8_bn32_bk16_s0_t128_fullcol_fieldpair",
            16,
            8,
            16,
            0,
            128,
            variant="field_pairs",
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
    variant,
):
    field_n = max(block_e, 16)
    pair_n = max(2 * block_e, 16)

    def _write_rhs_state(
        node,
        elem,
        coeff,
        metric_p,
        metric_v,
        surface,
        rhs,
        q_update,
        p_r,
        p_s,
        p_t,
        vx_r,
        vx_s,
        vx_t,
        vy_r,
        vy_s,
        vy_t,
        vz_r,
        vz_s,
        vz_t,
    ):
        c = elem * 4
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
            q_update[node, c + 1] = q_update[node, c + 1] + coeff * rhs_vx
            q_update[node, c + 2] = q_update[node, c + 2] + coeff * rhs_vy
            q_update[node, c + 3] = q_update[node, c + 3] + coeff * rhs_vz

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
            T.use_swizzle(panel_size=10, enable=True)
            coeff = coefficient[0]
            if variant in (_VARIANT_COPY_SHARED, _VARIANT_DIRECT_EPILOGUE):
                q_shared = T.alloc_shared((block_n, block_k), T.float64)
                acc_r = T.alloc_fragment((block_p, block_n), T.float64)
                acc_s = T.alloc_fragment((block_p, block_n), T.float64)
                acc_t = T.alloc_fragment((block_p, block_n), T.float64)
                if variant == _VARIANT_COPY_SHARED:
                    acc_r_shared = T.alloc_shared((block_p, block_n), T.float64)
                    acc_s_shared = T.alloc_shared((block_p, block_n), T.float64)
                    acc_t_shared = T.alloc_shared((block_p, block_n), T.float64)

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

                if variant == _VARIANT_COPY_SHARED:
                    T.copy(acc_r, acc_r_shared)
                    T.copy(acc_s, acc_s_shared)
                    T.copy(acc_t, acc_t_shared)

                for i, e in T.Parallel(block_p, block_e):
                    node = by * block_p + i
                    elem = bx * block_e + e
                    if node < M and elem < n_tets:
                        lc = e * 4
                        if variant == _VARIANT_COPY_SHARED:
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
                        else:
                            p_r = acc_r[i, lc]
                            p_s = acc_s[i, lc]
                            p_t = acc_t[i, lc]
                            vx_r = acc_r[i, lc + 1]
                            vx_s = acc_s[i, lc + 1]
                            vx_t = acc_t[i, lc + 1]
                            vy_r = acc_r[i, lc + 2]
                            vy_s = acc_s[i, lc + 2]
                            vy_t = acc_t[i, lc + 2]
                            vz_r = acc_r[i, lc + 3]
                            vz_s = acc_s[i, lc + 3]
                            vz_t = acc_t[i, lc + 3]
                        _write_rhs_state(
                            node,
                            elem,
                            coeff,
                            metric_p,
                            metric_v,
                            surface,
                            rhs,
                            q_update,
                            p_r,
                            p_s,
                            p_t,
                            vx_r,
                            vx_s,
                            vx_t,
                            vy_r,
                            vy_s,
                            vy_t,
                            vz_r,
                            vz_s,
                            vz_t,
                        )
            elif variant == _VARIANT_FIELD_FRAGMENTS:
                q_field_shared = T.alloc_shared((field_n, block_k), T.float64)
                p_r_acc = T.alloc_fragment((block_p, field_n), T.float64)
                p_s_acc = T.alloc_fragment((block_p, field_n), T.float64)
                p_t_acc = T.alloc_fragment((block_p, field_n), T.float64)
                vx_r_acc = T.alloc_fragment((block_p, field_n), T.float64)
                vx_s_acc = T.alloc_fragment((block_p, field_n), T.float64)
                vx_t_acc = T.alloc_fragment((block_p, field_n), T.float64)
                vy_r_acc = T.alloc_fragment((block_p, field_n), T.float64)
                vy_s_acc = T.alloc_fragment((block_p, field_n), T.float64)
                vy_t_acc = T.alloc_fragment((block_p, field_n), T.float64)
                vz_r_acc = T.alloc_fragment((block_p, field_n), T.float64)
                vz_s_acc = T.alloc_fragment((block_p, field_n), T.float64)
                vz_t_acc = T.alloc_fragment((block_p, field_n), T.float64)

                T.clear(p_r_acc)
                T.clear(p_s_acc)
                T.clear(p_t_acc)
                T.clear(vx_r_acc)
                T.clear(vx_s_acc)
                T.clear(vx_t_acc)
                T.clear(vy_r_acc)
                T.clear(vy_s_acc)
                T.clear(vy_t_acc)
                T.clear(vz_r_acc)
                T.clear(vz_s_acc)
                T.clear(vz_t_acc)
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

                    for field_offset in range(4):
                        for j, kk in T.Parallel(field_n, block_k):
                            k_idx = ko * block_k + kk
                            elem = bx * block_e + j
                            valid = (
                                (j < block_e)
                                & (elem < n_tets)
                                & (k_idx < K)
                            )
                            n_idx = bx * block_n + j * 4 + field_offset
                            q_field_shared[j, kk] = T.if_then_else(
                                valid & (n_idx < N),
                                Q[k_idx, n_idx],
                                T.float64(0.0),
                            )

                        if field_offset == 0:
                            T.gemm(
                                dr_shared,
                                q_field_shared,
                                p_r_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                            T.gemm(
                                ds_shared,
                                q_field_shared,
                                p_s_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                            T.gemm(
                                dt_shared,
                                q_field_shared,
                                p_t_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                        elif field_offset == 1:
                            T.gemm(
                                dr_shared,
                                q_field_shared,
                                vx_r_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                            T.gemm(
                                ds_shared,
                                q_field_shared,
                                vx_s_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                            T.gemm(
                                dt_shared,
                                q_field_shared,
                                vx_t_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                        elif field_offset == 2:
                            T.gemm(
                                dr_shared,
                                q_field_shared,
                                vy_r_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                            T.gemm(
                                ds_shared,
                                q_field_shared,
                                vy_s_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                            T.gemm(
                                dt_shared,
                                q_field_shared,
                                vy_t_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                        else:
                            T.gemm(
                                dr_shared,
                                q_field_shared,
                                vz_r_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                            T.gemm(
                                ds_shared,
                                q_field_shared,
                                vz_s_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                            T.gemm(
                                dt_shared,
                                q_field_shared,
                                vz_t_acc,
                                transpose_B=True,
                                policy=policy,
                            )

                for i, e in T.Parallel(block_p, block_e):
                    node = by * block_p + i
                    elem = bx * block_e + e
                    if node < M and elem < n_tets:
                        _write_rhs_state(
                            node,
                            elem,
                            coeff,
                            metric_p,
                            metric_v,
                            surface,
                            rhs,
                            q_update,
                            p_r_acc[i, e],
                            p_s_acc[i, e],
                            p_t_acc[i, e],
                            vx_r_acc[i, e],
                            vx_s_acc[i, e],
                            vx_t_acc[i, e],
                            vy_r_acc[i, e],
                            vy_s_acc[i, e],
                            vy_t_acc[i, e],
                            vz_r_acc[i, e],
                            vz_s_acc[i, e],
                            vz_t_acc[i, e],
                        )
            else:
                q_pair_shared = T.alloc_shared((pair_n, block_k), T.float64)
                pair0_r_acc = T.alloc_fragment((block_p, pair_n), T.float64)
                pair0_s_acc = T.alloc_fragment((block_p, pair_n), T.float64)
                pair0_t_acc = T.alloc_fragment((block_p, pair_n), T.float64)
                pair1_r_acc = T.alloc_fragment((block_p, pair_n), T.float64)
                pair1_s_acc = T.alloc_fragment((block_p, pair_n), T.float64)
                pair1_t_acc = T.alloc_fragment((block_p, pair_n), T.float64)
                pair0_r_shared = T.alloc_shared((block_p, pair_n), T.float64)
                pair0_s_shared = T.alloc_shared((block_p, pair_n), T.float64)
                pair0_t_shared = T.alloc_shared((block_p, pair_n), T.float64)
                pair1_r_shared = T.alloc_shared((block_p, pair_n), T.float64)
                pair1_s_shared = T.alloc_shared((block_p, pair_n), T.float64)
                pair1_t_shared = T.alloc_shared((block_p, pair_n), T.float64)

                T.clear(pair0_r_acc)
                T.clear(pair0_s_acc)
                T.clear(pair0_t_acc)
                T.clear(pair1_r_acc)
                T.clear(pair1_s_acc)
                T.clear(pair1_t_acc)
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

                    for pair_index in range(2):
                        for j, kk in T.Parallel(pair_n, block_k):
                            k_idx = ko * block_k + kk
                            local_elem = j // 2
                            elem = bx * block_e + local_elem
                            field_offset = pair_index * 2 + (j % 2)
                            n_idx = bx * block_n + local_elem * 4 + field_offset
                            valid = (
                                (local_elem < block_e)
                                & (elem < n_tets)
                                & (k_idx < K)
                                & (n_idx < N)
                            )
                            q_pair_shared[j, kk] = T.if_then_else(
                                valid,
                                Q[k_idx, n_idx],
                                T.float64(0.0),
                            )

                        if pair_index == 0:
                            T.gemm(
                                dr_shared,
                                q_pair_shared,
                                pair0_r_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                            T.gemm(
                                ds_shared,
                                q_pair_shared,
                                pair0_s_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                            T.gemm(
                                dt_shared,
                                q_pair_shared,
                                pair0_t_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                        else:
                            T.gemm(
                                dr_shared,
                                q_pair_shared,
                                pair1_r_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                            T.gemm(
                                ds_shared,
                                q_pair_shared,
                                pair1_s_acc,
                                transpose_B=True,
                                policy=policy,
                            )
                            T.gemm(
                                dt_shared,
                                q_pair_shared,
                                pair1_t_acc,
                                transpose_B=True,
                                policy=policy,
                            )

                T.copy(pair0_r_acc, pair0_r_shared)
                T.copy(pair0_s_acc, pair0_s_shared)
                T.copy(pair0_t_acc, pair0_t_shared)
                T.copy(pair1_r_acc, pair1_r_shared)
                T.copy(pair1_s_acc, pair1_s_shared)
                T.copy(pair1_t_acc, pair1_t_shared)

                for i, e in T.Parallel(block_p, block_e):
                    node = by * block_p + i
                    elem = bx * block_e + e
                    if node < M and elem < n_tets:
                        pair_col = e * 2
                        _write_rhs_state(
                            node,
                            elem,
                            coeff,
                            metric_p,
                            metric_v,
                            surface,
                            rhs,
                            q_update,
                            pair0_r_shared[i, pair_col],
                            pair0_s_shared[i, pair_col],
                            pair0_t_shared[i, pair_col],
                            pair0_r_shared[i, pair_col + 1],
                            pair0_s_shared[i, pair_col + 1],
                            pair0_t_shared[i, pair_col + 1],
                            pair1_r_shared[i, pair_col],
                            pair1_s_shared[i, pair_col],
                            pair1_t_shared[i, pair_col],
                            pair1_r_shared[i, pair_col + 1],
                            pair1_s_shared[i, pair_col + 1],
                            pair1_t_shared[i, pair_col + 1],
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
        _VARIANT_CODES[config.variant],
    )


__all__ = [
    "DerivativeVolumeAosConfig",
    "TILELANG_DERIVATIVE_VOLUME_AOS_CONFIG_NAME",
    "available_config_names",
    "build_tilelang_derivative_volume_aos_kernel",
    "get_config",
]
