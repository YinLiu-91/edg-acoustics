"""This module provides the AcousticsSimulation class, which includes the data structure for running a DG acoustics simulation.
The AcousticsSimulation class sets up the DG finite element discretization for the solution to the acoustic wave propagation problem.

Note that objects enclosed by square brackets (e.g., ``[dimension1, dimension2]``) denotes a matrix of dimension `[dimension1, dimension2]`.
"""

from __future__ import annotations
import math
import numpy
import scipy
import modepy
from scipy.spatial.qhull import Delaunay
import edg_acoustics
import time
import torch
import sys
import os
import triton
import triton.language as tl

import edg_acoustics.device_ini as device_ini

try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func

__all__ = ["AcousticsSimulation", "NODETOL"]

# Constants
NODETOL = 1.0e-7
"""float: Tolerance used to determine if a node lies on a facet."""

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = device_ini.device


@triton.jit
def build_maps_kernel(
    nodeids_ptr,  # 节点ID数组 [Np, N_tets]
    fmask_ptr,  # 面掩码 [4, Nfp]
    vmapM_ptr,  # 输出缓冲区 [4, Nfp, N_tets]
    N_tets,  # 总四面体数量
    Nfp,  # 每个面的节点数
    BLOCK_SIZE: tl.constexpr,  # 每个块处理的四面体数量
):
    # 计算程序ID和偏移量
    pid = tl.program_id(0)
    tet_start = pid * BLOCK_SIZE
    # 每个block处理BLOCK_SIZE个四面体
    for tet_idx in range(tet_start, tl.minimum(tet_start + BLOCK_SIZE, N_tets)):
        # 处理四个面
        for face in range(4):
            # 处理每个面上的节点
            for node in range(Nfp):
                # 获取Fmask中的索引
                fmask_idx = tl.load(fmask_ptr + face * Nfp + node)
                # 获取nodeids中的值
                nodeid = tl.load(nodeids_ptr + fmask_idx * N_tets + tet_idx)
                # 存储到vmapM
                tl.store(vmapM_ptr + (face * Nfp + node) * N_tets + tet_idx, nodeid)


@triton.jit
def volume_rhs_kernel(
    dQdr_ptr,
    dQds_ptr,
    dQdt_ptr,
    metric_p_ptr,
    metric_v_ptr,
    rhs_ptr,
    total_nodes: tl.constexpr,
    n_tets: tl.constexpr,
    n_var_tets: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_nodes
    node = offsets // n_tets
    tet = offsets - node * n_tets

    p = node * n_var_tets + tet
    vx = p + n_tets
    vy = vx + n_tets
    vz = vy + n_tets

    dPdr = tl.load(dQdr_ptr + p, mask=mask)
    dPds = tl.load(dQds_ptr + p, mask=mask)
    dPdt = tl.load(dQdt_ptr + p, mask=mask)

    dVxdr = tl.load(dQdr_ptr + vx, mask=mask)
    dVxds = tl.load(dQds_ptr + vx, mask=mask)
    dVxdt = tl.load(dQdt_ptr + vx, mask=mask)
    dVydr = tl.load(dQdr_ptr + vy, mask=mask)
    dVyds = tl.load(dQds_ptr + vy, mask=mask)
    dVydt = tl.load(dQdt_ptr + vy, mask=mask)
    dVzdr = tl.load(dQdr_ptr + vz, mask=mask)
    dVzds = tl.load(dQds_ptr + vz, mask=mask)
    dVzdt = tl.load(dQdt_ptr + vz, mask=mask)

    m00 = node * n_tets + tet
    m10 = 3 * total_nodes + m00
    m20 = 6 * total_nodes + m00
    m01 = total_nodes + m00
    m11 = 4 * total_nodes + m00
    m21 = 7 * total_nodes + m00
    m02 = 2 * total_nodes + m00
    m12 = 5 * total_nodes + m00
    m22 = 8 * total_nodes + m00

    rhs_vx = (
        tl.load(metric_v_ptr + m00, mask=mask) * dPdr
        + tl.load(metric_v_ptr + m10, mask=mask) * dPds
        + tl.load(metric_v_ptr + m20, mask=mask) * dPdt
    )
    rhs_vy = (
        tl.load(metric_v_ptr + m01, mask=mask) * dPdr
        + tl.load(metric_v_ptr + m11, mask=mask) * dPds
        + tl.load(metric_v_ptr + m21, mask=mask) * dPdt
    )
    rhs_vz = (
        tl.load(metric_v_ptr + m02, mask=mask) * dPdr
        + tl.load(metric_v_ptr + m12, mask=mask) * dPds
        + tl.load(metric_v_ptr + m22, mask=mask) * dPdt
    )

    rhs_p = (
        tl.load(metric_p_ptr + m00, mask=mask) * dVxdr
        + tl.load(metric_p_ptr + m10, mask=mask) * dVxds
        + tl.load(metric_p_ptr + m20, mask=mask) * dVxdt
        + tl.load(metric_p_ptr + m01, mask=mask) * dVydr
        + tl.load(metric_p_ptr + m11, mask=mask) * dVyds
        + tl.load(metric_p_ptr + m21, mask=mask) * dVydt
        + tl.load(metric_p_ptr + m02, mask=mask) * dVzdr
        + tl.load(metric_p_ptr + m12, mask=mask) * dVzds
        + tl.load(metric_p_ptr + m22, mask=mask) * dVzdt
    )

    tl.store(rhs_ptr + p, rhs_p, mask=mask)
    tl.store(rhs_ptr + vx, rhs_vx, mask=mask)
    tl.store(rhs_ptr + vy, rhs_vy, mask=mask)
    tl.store(rhs_ptr + vz, rhs_vz, mask=mask)


@triton.jit
def volume_surface_rhs_kernel(
    dQdr_ptr,
    dQds_ptr,
    dQdt_ptr,
    metric_p_ptr,
    metric_v_ptr,
    surface_ptr,
    rhs_ptr,
    q_update_ptr,
    total_nodes: tl.constexpr,
    n_tets: tl.constexpr,
    n_var_tets: tl.constexpr,
    coefficient: tl.constexpr,
    UPDATE_STATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_nodes
    node = offsets // n_tets
    tet = offsets - node * n_tets

    p = node * n_var_tets + tet
    vx = p + n_tets
    vy = vx + n_tets
    vz = vy + n_tets

    dPdr = tl.load(dQdr_ptr + p, mask=mask)
    dPds = tl.load(dQds_ptr + p, mask=mask)
    dPdt = tl.load(dQdt_ptr + p, mask=mask)

    dVxdr = tl.load(dQdr_ptr + vx, mask=mask)
    dVxds = tl.load(dQds_ptr + vx, mask=mask)
    dVxdt = tl.load(dQdt_ptr + vx, mask=mask)
    dVydr = tl.load(dQdr_ptr + vy, mask=mask)
    dVyds = tl.load(dQds_ptr + vy, mask=mask)
    dVydt = tl.load(dQdt_ptr + vy, mask=mask)
    dVzdr = tl.load(dQdr_ptr + vz, mask=mask)
    dVzds = tl.load(dQds_ptr + vz, mask=mask)
    dVzdt = tl.load(dQdt_ptr + vz, mask=mask)

    m00 = node * n_tets + tet
    m10 = 3 * total_nodes + m00
    m20 = 6 * total_nodes + m00
    m01 = total_nodes + m00
    m11 = 4 * total_nodes + m00
    m21 = 7 * total_nodes + m00
    m02 = 2 * total_nodes + m00
    m12 = 5 * total_nodes + m00
    m22 = 8 * total_nodes + m00

    rhs_vx = (
        tl.load(metric_v_ptr + m00, mask=mask) * dPdr
        + tl.load(metric_v_ptr + m10, mask=mask) * dPds
        + tl.load(metric_v_ptr + m20, mask=mask) * dPdt
        + tl.load(surface_ptr + vx, mask=mask)
    )
    rhs_vy = (
        tl.load(metric_v_ptr + m01, mask=mask) * dPdr
        + tl.load(metric_v_ptr + m11, mask=mask) * dPds
        + tl.load(metric_v_ptr + m21, mask=mask) * dPdt
        + tl.load(surface_ptr + vy, mask=mask)
    )
    rhs_vz = (
        tl.load(metric_v_ptr + m02, mask=mask) * dPdr
        + tl.load(metric_v_ptr + m12, mask=mask) * dPds
        + tl.load(metric_v_ptr + m22, mask=mask) * dPdt
        + tl.load(surface_ptr + vz, mask=mask)
    )

    rhs_p = (
        tl.load(metric_p_ptr + m00, mask=mask) * dVxdr
        + tl.load(metric_p_ptr + m10, mask=mask) * dVxds
        + tl.load(metric_p_ptr + m20, mask=mask) * dVxdt
        + tl.load(metric_p_ptr + m01, mask=mask) * dVydr
        + tl.load(metric_p_ptr + m11, mask=mask) * dVyds
        + tl.load(metric_p_ptr + m21, mask=mask) * dVydt
        + tl.load(metric_p_ptr + m02, mask=mask) * dVzdr
        + tl.load(metric_p_ptr + m12, mask=mask) * dVzds
        + tl.load(metric_p_ptr + m22, mask=mask) * dVzdt
        + tl.load(surface_ptr + p, mask=mask)
    )

    tl.store(rhs_ptr + p, rhs_p, mask=mask)
    tl.store(rhs_ptr + vx, rhs_vx, mask=mask)
    tl.store(rhs_ptr + vy, rhs_vy, mask=mask)
    tl.store(rhs_ptr + vz, rhs_vz, mask=mask)
    if UPDATE_STATE:
        tl.store(
            q_update_ptr + p,
            tl.load(q_update_ptr + p, mask=mask) + coefficient * rhs_p,
            mask=mask,
        )
        tl.store(
            q_update_ptr + vx,
            tl.load(q_update_ptr + vx, mask=mask) + coefficient * rhs_vx,
            mask=mask,
        )
        tl.store(
            q_update_ptr + vy,
            tl.load(q_update_ptr + vy, mask=mask) + coefficient * rhs_vy,
            mask=mask,
        )
        tl.store(
            q_update_ptr + vz,
            tl.load(q_update_ptr + vz, mask=mask) + coefficient * rhs_vz,
            mask=mask,
        )


@triton.jit
def derivative_volume_surface_rhs_kernel(
    q_ptr,
    dr_ptr,
    ds_ptr,
    dt_ptr,
    metric_p_ptr,
    metric_v_ptr,
    surface_ptr,
    rhs_ptr,
    q_update_ptr,
    total_nodes: tl.constexpr,
    n_tets: tl.constexpr,
    n_var_tets: tl.constexpr,
    n_p: tl.constexpr,
    coefficient: tl.constexpr,
    UPDATE_STATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_nodes
    node = offsets // n_tets
    tet = offsets - node * n_tets

    dPdr = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    dPds = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    dPdt = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    dVxdr = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    dVxds = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    dVxdt = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    dVydr = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    dVyds = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    dVydt = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    dVzdr = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    dVzds = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    dVzdt = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)

    for k in range(n_p):
        d_index = node * n_p + k
        q_base = k * n_var_tets + tet
        p = q_base
        vx = q_base + n_tets
        vy = q_base + 2 * n_tets
        vz = q_base + 3 * n_tets
        dr = tl.load(dr_ptr + d_index, mask=mask)
        ds = tl.load(ds_ptr + d_index, mask=mask)
        dt = tl.load(dt_ptr + d_index, mask=mask)
        q_p = tl.load(q_ptr + p, mask=mask)
        q_vx = tl.load(q_ptr + vx, mask=mask)
        q_vy = tl.load(q_ptr + vy, mask=mask)
        q_vz = tl.load(q_ptr + vz, mask=mask)
        dPdr += dr * q_p
        dPds += ds * q_p
        dPdt += dt * q_p
        dVxdr += dr * q_vx
        dVxds += ds * q_vx
        dVxdt += dt * q_vx
        dVydr += dr * q_vy
        dVyds += ds * q_vy
        dVydt += dt * q_vy
        dVzdr += dr * q_vz
        dVzds += ds * q_vz
        dVzdt += dt * q_vz

    out_p = node * n_var_tets + tet
    out_vx = out_p + n_tets
    out_vy = out_p + 2 * n_tets
    out_vz = out_p + 3 * n_tets

    m00 = node * n_tets + tet
    m10 = 3 * total_nodes + m00
    m20 = 6 * total_nodes + m00
    m01 = total_nodes + m00
    m11 = 4 * total_nodes + m00
    m21 = 7 * total_nodes + m00
    m02 = 2 * total_nodes + m00
    m12 = 5 * total_nodes + m00
    m22 = 8 * total_nodes + m00

    rhs_vx = (
        tl.load(metric_v_ptr + m00, mask=mask) * dPdr
        + tl.load(metric_v_ptr + m10, mask=mask) * dPds
        + tl.load(metric_v_ptr + m20, mask=mask) * dPdt
        + tl.load(surface_ptr + out_vx, mask=mask)
    )
    rhs_vy = (
        tl.load(metric_v_ptr + m01, mask=mask) * dPdr
        + tl.load(metric_v_ptr + m11, mask=mask) * dPds
        + tl.load(metric_v_ptr + m21, mask=mask) * dPdt
        + tl.load(surface_ptr + out_vy, mask=mask)
    )
    rhs_vz = (
        tl.load(metric_v_ptr + m02, mask=mask) * dPdr
        + tl.load(metric_v_ptr + m12, mask=mask) * dPds
        + tl.load(metric_v_ptr + m22, mask=mask) * dPdt
        + tl.load(surface_ptr + out_vz, mask=mask)
    )

    rhs_p = (
        tl.load(metric_p_ptr + m00, mask=mask) * dVxdr
        + tl.load(metric_p_ptr + m10, mask=mask) * dVxds
        + tl.load(metric_p_ptr + m20, mask=mask) * dVxdt
        + tl.load(metric_p_ptr + m01, mask=mask) * dVydr
        + tl.load(metric_p_ptr + m11, mask=mask) * dVyds
        + tl.load(metric_p_ptr + m21, mask=mask) * dVydt
        + tl.load(metric_p_ptr + m02, mask=mask) * dVzdr
        + tl.load(metric_p_ptr + m12, mask=mask) * dVzds
        + tl.load(metric_p_ptr + m22, mask=mask) * dVzdt
        + tl.load(surface_ptr + out_p, mask=mask)
    )

    tl.store(rhs_ptr + out_p, rhs_p, mask=mask)
    tl.store(rhs_ptr + out_vx, rhs_vx, mask=mask)
    tl.store(rhs_ptr + out_vy, rhs_vy, mask=mask)
    tl.store(rhs_ptr + out_vz, rhs_vz, mask=mask)
    if UPDATE_STATE:
        tl.store(
            q_update_ptr + out_p,
            tl.load(q_update_ptr + out_p, mask=mask) + coefficient * rhs_p,
            mask=mask,
        )
        tl.store(
            q_update_ptr + out_vx,
            tl.load(q_update_ptr + out_vx, mask=mask) + coefficient * rhs_vx,
            mask=mask,
        )
        tl.store(
            q_update_ptr + out_vy,
            tl.load(q_update_ptr + out_vy, mask=mask) + coefficient * rhs_vy,
            mask=mask,
        )
        tl.store(
            q_update_ptr + out_vz,
            tl.load(q_update_ptr + out_vz, mask=mask) + coefficient * rhs_vz,
            mask=mask,
        )


@triton.jit
def lift_surface_kernel(
    lift_ptr,
    flux_ptr,
    surface_ptr,
    total_outputs: tl.constexpr,
    n_var_tets: tl.constexpr,
    n_faces: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_outputs
    node = offsets // n_var_tets
    col = offsets - node * n_var_tets
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    for face_node in range(n_faces):
        acc += (
            tl.load(lift_ptr + node * n_faces + face_node, mask=mask)
            * tl.load(flux_ptr + face_node * n_var_tets + col, mask=mask)
        )
    tl.store(surface_ptr + offsets, acc, mask=mask)


@triton.jit
def interior_flux_kernel(
    q_ptr,
    vmapM_q_ptr,
    vmapP_q_ptr,
    cn1s_ptr,
    cn2s_ptr,
    cn3s_ptr,
    cn1n2_ptr,
    cn1n3_ptr,
    cn2n3_ptr,
    n1rho_ptr,
    n2rho_ptr,
    n3rho_ptr,
    csn1rho_ptr,
    csn2rho_ptr,
    csn3rho_ptr,
    fscale_ptr,
    flux_ptr,
    total_faces: tl.constexpr,
    n_tets: tl.constexpr,
    n_var_tets: tl.constexpr,
    c0: tl.constexpr,
    SCALE_FLUX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_faces
    tet = offsets % n_tets
    face = offsets // n_tets

    pM = tl.load(q_ptr + tl.load(vmapM_q_ptr + offsets, mask=mask), mask=mask)
    pP = tl.load(q_ptr + tl.load(vmapP_q_ptr + offsets, mask=mask), mask=mask)
    vxM = tl.load(q_ptr + tl.load(vmapM_q_ptr + total_faces + offsets, mask=mask), mask=mask)
    vxP = tl.load(q_ptr + tl.load(vmapP_q_ptr + total_faces + offsets, mask=mask), mask=mask)
    vyM = tl.load(q_ptr + tl.load(vmapM_q_ptr + 2 * total_faces + offsets, mask=mask), mask=mask)
    vyP = tl.load(q_ptr + tl.load(vmapP_q_ptr + 2 * total_faces + offsets, mask=mask), mask=mask)
    vzM = tl.load(q_ptr + tl.load(vmapM_q_ptr + 3 * total_faces + offsets, mask=mask), mask=mask)
    vzP = tl.load(q_ptr + tl.load(vmapP_q_ptr + 3 * total_faces + offsets, mask=mask), mask=mask)

    dp = pM - pP
    dvx = vxM - vxP
    dvy = vyM - vyP
    dvz = vzM - vzP

    flux_vx = (
        tl.load(n1rho_ptr + offsets, mask=mask) * dp
        + tl.load(cn1s_ptr + offsets, mask=mask) * dvx
        + tl.load(cn1n2_ptr + offsets, mask=mask) * dvy
        + tl.load(cn1n3_ptr + offsets, mask=mask) * dvz
    )
    flux_vy = (
        tl.load(n2rho_ptr + offsets, mask=mask) * dp
        + tl.load(cn1n2_ptr + offsets, mask=mask) * dvx
        + tl.load(cn2s_ptr + offsets, mask=mask) * dvy
        + tl.load(cn2n3_ptr + offsets, mask=mask) * dvz
    )
    flux_vz = (
        tl.load(n3rho_ptr + offsets, mask=mask) * dp
        + tl.load(cn1n3_ptr + offsets, mask=mask) * dvx
        + tl.load(cn2n3_ptr + offsets, mask=mask) * dvy
        + tl.load(cn3s_ptr + offsets, mask=mask) * dvz
    )
    flux_p = (
        -0.5 * c0 * dp
        + tl.load(csn1rho_ptr + offsets, mask=mask) * dvx
        + tl.load(csn2rho_ptr + offsets, mask=mask) * dvy
        + tl.load(csn3rho_ptr + offsets, mask=mask) * dvz
    )
    if SCALE_FLUX:
        fscale = tl.load(fscale_ptr + offsets, mask=mask)
        flux_p *= fscale
        flux_vx *= fscale
        flux_vy *= fscale
        flux_vz *= fscale

    out_base = face * n_var_tets + tet
    tl.store(flux_ptr + out_base, flux_p, mask=mask)
    tl.store(flux_ptr + out_base + n_tets, flux_vx, mask=mask)
    tl.store(flux_ptr + out_base + 2 * n_tets, flux_vy, mask=mask)
    tl.store(flux_ptr + out_base + 3 * n_tets, flux_vz, mask=mask)


@triton.jit
def boundary_ri_flux_kernel(
    q_ptr,
    vmap_q_ptr,
    flux_map_q_ptr,
    nx_ptr,
    ny_ptr,
    nz_ptr,
    fscale_ptr,
    flux_ptr,
    vn_ptr,
    ou_ptr,
    in_ptr,
    n_boundary: tl.constexpr,
    rho0: tl.constexpr,
    c0: tl.constexpr,
    ri: tl.constexpr,
    SCALE_FLUX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_boundary

    idx_p = tl.load(vmap_q_ptr + offsets, mask=mask, other=0)
    idx_vx = tl.load(vmap_q_ptr + n_boundary + offsets, mask=mask, other=0)
    idx_vy = tl.load(vmap_q_ptr + 2 * n_boundary + offsets, mask=mask, other=0)
    idx_vz = tl.load(vmap_q_ptr + 3 * n_boundary + offsets, mask=mask, other=0)

    p = tl.load(q_ptr + idx_p, mask=mask)
    vx = tl.load(q_ptr + idx_vx, mask=mask)
    vy = tl.load(q_ptr + idx_vy, mask=mask)
    vz = tl.load(q_ptr + idx_vz, mask=mask)
    nx = tl.load(nx_ptr + offsets, mask=mask)
    ny = tl.load(ny_ptr + offsets, mask=mask)
    nz = tl.load(nz_ptr + offsets, mask=mask)

    vn = nx * vx + ny * vy + nz * vz
    ou = vn + p / (rho0 * c0)
    incoming = ri * ou
    velocity_flux = p / rho0 - 0.5 * c0 * (ou + incoming)
    pressure_flux = (vn - 0.5 * ou + 0.5 * incoming) * (c0 * c0 * rho0)
    if SCALE_FLUX:
        fscale = tl.load(fscale_ptr + offsets, mask=mask)
        pressure_flux *= fscale
        velocity_flux *= fscale

    tl.store(vn_ptr + offsets, vn, mask=mask)
    tl.store(ou_ptr + offsets, ou, mask=mask)
    tl.store(in_ptr + offsets, incoming, mask=mask)

    out_p = tl.load(flux_map_q_ptr + offsets, mask=mask, other=0)
    out_vx = tl.load(flux_map_q_ptr + n_boundary + offsets, mask=mask, other=0)
    out_vy = tl.load(flux_map_q_ptr + 2 * n_boundary + offsets, mask=mask, other=0)
    out_vz = tl.load(flux_map_q_ptr + 3 * n_boundary + offsets, mask=mask, other=0)
    tl.store(flux_ptr + out_p, pressure_flux, mask=mask)
    tl.store(flux_ptr + out_vx, nx * velocity_flux, mask=mask)
    tl.store(flux_ptr + out_vy, ny * velocity_flux, mask=mask)
    tl.store(flux_ptr + out_vz, nz * velocity_flux, mask=mask)


@triton.jit
def boundary_rp_flux_kernel(
    q_ptr,
    vmap_q_ptr,
    flux_map_q_ptr,
    nx_ptr,
    ny_ptr,
    nz_ptr,
    fscale_ptr,
    rp_a_ptr,
    rp_zeta_ptr,
    flux_ptr,
    vn_ptr,
    ou_ptr,
    in_ptr,
    phi_ptr,
    n_boundary: tl.constexpr,
    rho0: tl.constexpr,
    c0: tl.constexpr,
    ri: tl.constexpr,
    RP_COUNT: tl.constexpr,
    SCALE_FLUX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_boundary

    idx_p = tl.load(vmap_q_ptr + offsets, mask=mask, other=0)
    idx_vx = tl.load(vmap_q_ptr + n_boundary + offsets, mask=mask, other=0)
    idx_vy = tl.load(vmap_q_ptr + 2 * n_boundary + offsets, mask=mask, other=0)
    idx_vz = tl.load(vmap_q_ptr + 3 * n_boundary + offsets, mask=mask, other=0)

    p = tl.load(q_ptr + idx_p, mask=mask)
    vx = tl.load(q_ptr + idx_vx, mask=mask)
    vy = tl.load(q_ptr + idx_vy, mask=mask)
    vz = tl.load(q_ptr + idx_vz, mask=mask)
    nx = tl.load(nx_ptr + offsets, mask=mask)
    ny = tl.load(ny_ptr + offsets, mask=mask)
    nz = tl.load(nz_ptr + offsets, mask=mask)

    vn = nx * vx + ny * vy + nz * vz
    ou = vn + p / (rho0 * c0)
    incoming = ri * ou

    for pole in range(RP_COUNT):
        phi_index = pole * n_boundary + offsets
        phi_value = tl.load(phi_ptr + phi_index, mask=mask)
        incoming += tl.load(rp_a_ptr + pole) * phi_value
        tl.store(
            phi_ptr + phi_index,
            ou - tl.load(rp_zeta_ptr + pole) * phi_value,
            mask=mask,
        )

    velocity_flux = p / rho0 - 0.5 * c0 * (ou + incoming)
    pressure_flux = (vn - 0.5 * ou + 0.5 * incoming) * (c0 * c0 * rho0)
    if SCALE_FLUX:
        fscale = tl.load(fscale_ptr + offsets, mask=mask)
        pressure_flux *= fscale
        velocity_flux *= fscale

    tl.store(vn_ptr + offsets, vn, mask=mask)
    tl.store(ou_ptr + offsets, ou, mask=mask)
    tl.store(in_ptr + offsets, incoming, mask=mask)

    out_p = tl.load(flux_map_q_ptr + offsets, mask=mask, other=0)
    out_vx = tl.load(flux_map_q_ptr + n_boundary + offsets, mask=mask, other=0)
    out_vy = tl.load(flux_map_q_ptr + 2 * n_boundary + offsets, mask=mask, other=0)
    out_vz = tl.load(flux_map_q_ptr + 3 * n_boundary + offsets, mask=mask, other=0)
    tl.store(flux_ptr + out_p, pressure_flux, mask=mask)
    tl.store(flux_ptr + out_vx, nx * velocity_flux, mask=mask)
    tl.store(flux_ptr + out_vy, ny * velocity_flux, mask=mask)
    tl.store(flux_ptr + out_vz, nz * velocity_flux, mask=mask)


@triton.jit
def boundary_rp_cp_flux_kernel(
    q_ptr,
    vmap_q_ptr,
    flux_map_q_ptr,
    nx_ptr,
    ny_ptr,
    nz_ptr,
    fscale_ptr,
    rp_a_ptr,
    rp_zeta_ptr,
    cp_b_ptr,
    cp_c_ptr,
    cp_alpha_ptr,
    cp_beta_ptr,
    flux_ptr,
    vn_ptr,
    ou_ptr,
    in_ptr,
    phi_ptr,
    kexi1_ptr,
    kexi2_ptr,
    n_boundary: tl.constexpr,
    rho0: tl.constexpr,
    c0: tl.constexpr,
    ri: tl.constexpr,
    RP_COUNT: tl.constexpr,
    CP_COUNT: tl.constexpr,
    SCALE_FLUX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_boundary

    idx_p = tl.load(vmap_q_ptr + offsets, mask=mask, other=0)
    idx_vx = tl.load(vmap_q_ptr + n_boundary + offsets, mask=mask, other=0)
    idx_vy = tl.load(vmap_q_ptr + 2 * n_boundary + offsets, mask=mask, other=0)
    idx_vz = tl.load(vmap_q_ptr + 3 * n_boundary + offsets, mask=mask, other=0)

    p = tl.load(q_ptr + idx_p, mask=mask)
    vx = tl.load(q_ptr + idx_vx, mask=mask)
    vy = tl.load(q_ptr + idx_vy, mask=mask)
    vz = tl.load(q_ptr + idx_vz, mask=mask)
    nx = tl.load(nx_ptr + offsets, mask=mask)
    ny = tl.load(ny_ptr + offsets, mask=mask)
    nz = tl.load(nz_ptr + offsets, mask=mask)

    vn = nx * vx + ny * vy + nz * vz
    ou = vn + p / (rho0 * c0)
    incoming = ri * ou

    for pole in range(RP_COUNT):
        phi_index = pole * n_boundary + offsets
        phi_value = tl.load(phi_ptr + phi_index, mask=mask)
        incoming += tl.load(rp_a_ptr + pole) * phi_value
        tl.store(
            phi_ptr + phi_index,
            ou - tl.load(rp_zeta_ptr + pole) * phi_value,
            mask=mask,
        )

    for pole in range(CP_COUNT):
        cp_index = pole * n_boundary + offsets
        kexi1 = tl.load(kexi1_ptr + cp_index, mask=mask)
        kexi2 = tl.load(kexi2_ptr + cp_index, mask=mask)
        alpha = tl.load(cp_alpha_ptr + pole)
        beta = tl.load(cp_beta_ptr + pole)
        incoming += tl.load(cp_b_ptr + pole) * kexi1 + tl.load(cp_c_ptr + pole) * kexi2
        tl.store(kexi1_ptr + cp_index, ou - alpha * kexi1 - beta * kexi2, mask=mask)
        tl.store(kexi2_ptr + cp_index, -alpha * kexi2 + beta * kexi1, mask=mask)

    velocity_flux = p / rho0 - 0.5 * c0 * (ou + incoming)
    pressure_flux = (vn - 0.5 * ou + 0.5 * incoming) * (c0 * c0 * rho0)
    if SCALE_FLUX:
        fscale = tl.load(fscale_ptr + offsets, mask=mask)
        pressure_flux *= fscale
        velocity_flux *= fscale

    tl.store(vn_ptr + offsets, vn, mask=mask)
    tl.store(ou_ptr + offsets, ou, mask=mask)
    tl.store(in_ptr + offsets, incoming, mask=mask)

    out_p = tl.load(flux_map_q_ptr + offsets, mask=mask, other=0)
    out_vx = tl.load(flux_map_q_ptr + n_boundary + offsets, mask=mask, other=0)
    out_vy = tl.load(flux_map_q_ptr + 2 * n_boundary + offsets, mask=mask, other=0)
    out_vz = tl.load(flux_map_q_ptr + 3 * n_boundary + offsets, mask=mask, other=0)
    tl.store(flux_ptr + out_p, pressure_flux, mask=mask)
    tl.store(flux_ptr + out_vx, nx * velocity_flux, mask=mask)
    tl.store(flux_ptr + out_vy, ny * velocity_flux, mask=mask)
    tl.store(flux_ptr + out_vz, nz * velocity_flux, mask=mask)


class AcousticsSimulation:
    """Acoustics simulation data structure for running a DG acoustics simulation.

    :class:`.AcousticsSimulation` contains the domain discretization and sets up the DG finite element discretisation
    for the solution to the acoustic wave propagation.

    Args:
        rho0 (float): the density of the medium in which the acoustic wave propagates.
        c0 (float): the speed of sound in the medium in which the acoustic wave propagates.
        Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
            propagation problem.
        mesh (edg_acoustics.Mesh): the mesh object containing the mesh information for the domain discretisation.
        BC_list (dict[str, int]): a dictionary containing the definition of the boundary
            conditions that are present in the mesh. BC_list must contain the same keys as
            :attr:`edg_acoustics.Mesh.BC_triangles`, i.e., all boundary conditions in the mesh must have an associated boundary condition
            definition.
        node_tolerance (float): the tolerance used to check if a node belongs to a facet or not.
            <default>: :py:const:`edg_acoustics.acoustics_simulation.NODETOL`

    Raises:
        ValueError: If BC_list['my_label'] is not present in the mesh, an error is raised. If a label
            is present in the mesh but not in BC_list, an error is raised.

    Attributes:
        BC_list (dict[str, int]): a dictionary containing the definition of the boundary
            conditions that are present in the mesh. BC_list must contain the same keys as
            :attr:`edg_acoustics.Mesh.BC_triangles`, i.e., all boundary conditions in the mesh must have an associated boundary condition
            definition.

        BCnode (list[dict]): List of boundary map nodes, each element being a dictionary
                with keys (values) ['label' (int), 'map' (torch.tensor), 'vmap' (torch.tensor)].

        c0 (float): the speed of sound in the medium in which the acoustic wave propagates.

        Dr (torch.tensor): the reference element differentiation matrices containing the discrete representation
            of :math:`\\frac{\\partial}{\\partial r}`, in the rst reference element coordinate system.
        Ds (torch.tensor): the reference element differentiation matrices containing the discrete representation
            of :math:`\\frac{\\partial}{\\partial s}`, in the rst reference element coordinate system.
        Dt (torch.tensor): the reference element differentiation matrices containing the discrete representation
            of :math:`\\frac{\\partial}{\\partial t}`, in the rst reference element coordinate system.

        dim (int): the geometric dimension of the space where the acoustic problem is solved. Always set to 3.

        dtscale (float): the time step scale based on the mesh size measure, is set to the minimum diameter of the inscribed spheres in all elements.

        Fmask (torch.tensor): ``[4, Nfp]`` array containing indices of nodes per surface of the tetrahedron.

        Fscale (torch.tensor): ``[4*Nfp, N_tets]`` ratio of surface to volume Jacobian of facial node.

        J (torch.tensor): ``[Np, N_tets]`` The determinant of the Jacobian matrix for the coordinate transformation, at the collocation nodes. 

        lift (torch.tensor): ``[Np, 4*Nfp]`` an array containing the product of inverse of the mass matrix (3D) with the face-mass matrices (2D).
        
        mesh (edg_acoustics.Mesh): the mesh object containing the mesh information for the domain discretisation.

        Nfp (int): number of collocation nodes in a face.

        Np (int): number of collocation nodes in an element.

        N_tets (int): number of tetrahedra in the mesh.

        Nx (int): the polynomial degree of the approximating DG finite element space used to solve the acoustic wave
            propagation problem.

        node_tolerance (float): the tolerance used to check if a node belongs to a facet or not.
            <default>: :py:const:`edg_acoustics.acoustics_simulation.NODETOL`.

        rho0 (float): the density of the medium in which the acoustic wave propagates.

        rst (torch.tensor): the reference element coordinates :math:`(r, s, t)` of the collocation points.
            Physical coordinates :attr:`xyz` are obtained by mapping for each element the :attr:`rst` coordinates of the reference element into
            the physical domain. `rst[0]` contains the r-coordinates, `rst[1]` contains the s-coordinates, `rst[2]`
            contains the t-coordinates.

        rst_xyz (torch.tensor): ``[3, 3, Np, N_tets]`` The derivative of the local coordinates :math:`R = (r, s, t)` with 
            respect to the physical coordinates :math:`X = (x, y, z)`, i.e., :math:`\\frac{\\partial R}{\\partial X}`, at the collocation nodes. Specifically:

            - rst_xyz[0, 0]: rx (torch.tensor): ``[Np, N_tets]`` The derivative of the local coordinates
              :math:`r` with respect to the physical coordinates :math:`x`, i.e., :math:`\\frac{\\partial r}{\\partial x}`, at the collocation nodes.
            - rst_xyz[1, 0]: sx (torch.tensor): ``[Np, N_tets]`` The derivative of the local coordinates
              :math:`s` with respect to the physical coordinates :math:`x`, i.e., :math:`\\frac{\\partial s}{\\partial x}`, at the collocation nodes.
            - rst_xyz[2, 0]: tx (torch.tensor): ``[Np, N_tets]`` The derivative of the local coordinates
              :math:`t` with respect to the physical coordinates :math:`x`, i.e., :math:`\\frac{\\partial t}{\\partial x}`, at the collocation nodes.
            - rst_xyz[0, 1]: ry (torch.tensor): ``[Np, N_tets]`` The derivative of the local coordinates
              :math:`r` with respect to the physical coordinates :math:`y`, i.e., :math:`\\frac{\\partial r}{\\partial y}`, at the collocation nodes.
            - rst_xyz[1, 1]: sy (torch.tensor): ``[Np, N_tets]`` The derivative of the local coordinates
              :math:`s` with respect to the physical coordinates :math:`y`, i.e., :math:`\\frac{\\partial s}{\\partial y}`, at the collocation nodes.
            - rst_xyz[2, 1]: ty (torch.tensor): ``[Np, N_tets]`` The derivative of the local coordinates
              :math:`t` with respect to the physical coordinates :math:`y`, i.e., :math:`\\frac{\\partial t}{\\partial y}`, at the collocation nodes.
            - rst_xyz[0, 2]: rz (torch.tensor): ``[Np, N_tets]`` The derivative of the local coordinates
              :math:`r` with respect to the physical coordinates :math:`z`, i.e., :math:`\\frac{\\partial r}{\\partial z}`, at the collocation nodes.
            - rst_xyz[1, 2]: sz (torch.tensor): ``[Np, N_tets]`` The derivative of the local coordinates
              :math:`s` with respect to the physical coordinates :math:`z`, i.e., :math:`\\frac{\\partial s}{\\partial z}`, at the collocation nodes.
            - rst_xyz[2, 2]: tz (torch.tensor): ``[Np, N_tets]`` The derivative of the local coordinates
              :math:`t` with respect to the physical coordinates :math:`z`, i.e., :math:`\\frac{\\partial t}{\\partial z}`, at the collocation nodes.

        sJ (torch.tensor): ``[4*Nfp, N_tets]`` The determinant of the surface Jacobian matrix at each of the
            collocation nodes, for each of the 4 faces of the :attr:`N_tets` elements. 

        V (torch.tensor): ``[Np, Np]`` vandermonde matrix of the orthonormal basis functions on the reference simplex element. Polynomial basis 
            can exactly represent polynomials up to degree :attr:`Nx`. Consider the set of :attr:`~.AcousticsSimulation.Np` 3D nodes, with the coordinates of each node :math:`i` equal to
            :math:`(r_{i}, s_{i}, t_{i})`, in :attr:`.rst`, and the set of :math:`m` orthonormal basis functions, 
            the vandermonde matrix will be :math:`V_{i,j} = f_{j}(r_{i}, s_{i}, t_{i})`. 

        xyz (torch.tensor): ``[3, Np, N_tets]`` the physical space coordinates :math:`(x, y, z)` of the collocation points of each element of the\
            mesh. `xyz[0]` contains the x-coordinates, `xyz[1]` contains the y-coordinates, `xyz[2]`
            contains the z-coordinates.

        n_xyz (torch.tensor): ``[3, 4*Np, N_tets]`` the outwards normals :math:`\\vec{n}= (n_x, n_y, n_z)` at each collocation
            point on the element faces. Specifically:
                
            - n_xyz[0, :] (torch.tensor): nx ``[4*Nfp, N_tets]`` The :math:`x`-component of the outward normal :math:`\\vec{n}`
              at each of the :attr:`Nfp` nodes on each of the 4 facets of each of the :attr:`N_tets` elements.
            - n_xyz[1, :] (torch.tensor): ny ``[4*Nfp, N_tets]`` The :math:`y`-component of the outward normal :math:`\\vec{n}`
              at each of the :attr:`Nfp` nodes on each of the 4 facets of each of the :attr:`N_tets` elements.
            - n_xyz[2, :] (torch.tensor): nz ``[4*Nfp, N_tets]`` The :math:`z`-component of the outward normal :math:`\\vec{n}`
              at each of the :attr:`Nfp` nodes on each of the 4 facets of each of the :attr:`N_tets` elements.

        vmapM (torch.tensor): ``[4*Nfp*N_tets, 1]`` an array containing the global indices for interior values.

        vmapP (torch.tensor): ``[4*Nfp*N_tets, 1]`` an array containing the global indices for exterior values.

    """

    def __init__(
        self,
        rho0: float,
        c0: float,
        Nx: int,
        mesh: edg_acoustics.Mesh,
        BC_list: dict[str, int],
        node_tolerance: float = NODETOL,
    ):
        self.device = device
        # Check if BC_list and mesh are compatible
        if not AcousticsSimulation.check_BC_list(BC_list, mesh):
            raise ValueError(
                "[edg_acoustics.AcousticSimulation] All BC labels must be present in the mesh and all labels in the mesh must be "
                "present in BC_list."
            )

        # Store input parameters
        self.rho0 = rho0
        self.c0 = c0
        self.mesh = mesh
        self.Nx = Nx
        self.N_tets = mesh.EToV.shape[1]
        self.BC_list = BC_list
        self.dim = 3  # we are always in 3D, just added for external reference
        self.node_tolerance = node_tolerance  # define a tolerance value for determining if a node belongs to a facet or not

        # Compute attributes
        self.Np = AcousticsSimulation.compute_Np(
            Nx
        )  # number of colocation nodes in an element
        self.Nfp = AcousticsSimulation.compute_Nfp(Nx)  # number of nodes in a face

        # dtype_input: str ='float64'
        # self.dtype_dict={'float64': numpy.float64, 'float32': numpy.float32}

        self.init_local_system()

    def init_local_system(self):
        """Call the methods to initialise the local system and compute all the attributes of the AcousticsSimulation object."""

        self.rst, self.xyz = AcousticsSimulation.compute_collocation_nodes(
            self.mesh.EToV,
            torch.from_numpy(self.mesh.vertices).to(device_ini.device),
            self.Nx,
            dim=self.dim,
        )

        # Compute the van der Monde matrix for the basis functions
        self.V = AcousticsSimulation.compute_van_der_monde_matrix(self.Nx, self.rst)

        # Compute the derivative matrices
        self.Dr, self.Ds, self.Dt = AcousticsSimulation.compute_derivative_matrix(
            self.Nx, self.rst
        )

        # Find all the ``Nfp`` face nodes that lie on each surface.
        self.Fmask = AcousticsSimulation.compute_Fmask(
            torch.from_numpy(self.rst), self.node_tolerance
        )

        # Compute the product of inverse of the mass matrix (3D) with the face-mass matrices (2D)
        self.lift = AcousticsSimulation.compute_lift(
            self.V, torch.from_numpy(self.rst).to(device_ini.device), self.Fmask
        )
        self.lift = self.lift.to(self.device)

        # Compute the metric terms for the mesh
        self.rst_xyz, self.J = AcousticsSimulation.geometric_factors_3d(
            self.xyz,
            self.Dr,
            self.Ds,
            self.Dt,
        )

        # Compute the face normals at the collocation points and the surface Jacobians
        self.n_xyz, self.sJ = AcousticsSimulation.normals_3d(
            self.xyz,
            self.rst_xyz,
            self.J,
            self.Fmask,
        )
        self.n_xyz = self.n_xyz.to(self.device).to(device_ini.dtype)
        self.sJ = self.sJ.to(self.device).to(device_ini.dtype)

        # Compute ratio of surface to volume Jacobian of facial node
        self.Fscale = self.sJ / self.J[self.Fmask.reshape(-1), :]

        # Find connectivity for nodes given per surface in all elements
        self.vmapM, self.vmapP = AcousticsSimulation.build_maps_3d(
            self.xyz,
            self.mesh.EToE,
            self.mesh.EToF,
            self.Fmask,
            self.node_tolerance,
        )

        # Build specialized nodal maps for various types of boundary conditions,specified in BC_list
        self.BCnode = AcousticsSimulation.build_BCmaps_3d(
            self.BC_list,
            self.mesh.EToV,
            self.vmapM,
            self.mesh.BC_triangles,
            self.Nx,
        )

        self.dtscale = (
            AcousticsSimulation.diameter_3d(self.Fscale) / self.c0 / (2 * self.Nx + 1)
        )
        self.init_runtime_buffers()
        self.cache_static_indices()

    def init_runtime_buffers(self):
        """Preallocate buffers reused by RHS computations."""
        face_shape = self.Fscale.shape
        node_shape = (self.Np, self.N_tets)
        kwargs = {"device": self.device, "dtype": device_ini.dtype}
        self._dVx = torch.empty(face_shape, **kwargs)
        self._dVy = torch.empty(face_shape, **kwargs)
        self._dVz = torch.empty(face_shape, **kwargs)
        self._dP = torch.empty(face_shape, **kwargs)
        self._fluxVx = torch.empty(face_shape, **kwargs)
        self._fluxVy = torch.empty(face_shape, **kwargs)
        self._fluxVz = torch.empty(face_shape, **kwargs)
        self._fluxP = torch.empty(face_shape, **kwargs)
        self._jump_left = torch.empty(self.vmapM.shape, **kwargs)
        self._jump_right = torch.empty(self.vmapP.shape, **kwargs)
        self._q_by_node = torch.empty((self.Np, 4 * self.N_tets), **kwargs)
        self._q_by_node_view = self._q_by_node.view(self.Np, 4, self.N_tets)
        self._D_stack = torch.stack((self.Dr, self.Ds, self.Dt), dim=0).contiguous()
        self._dQ_by_derivative = torch.empty(
            (3, self.Np, 4 * self.N_tets), **kwargs
        )
        self._dQdr_by_node = self._dQ_by_derivative[0]
        self._dQds_by_node = self._dQ_by_derivative[1]
        self._dQdt_by_node = self._dQ_by_derivative[2]
        self._dQdr_view = self._dQdr_by_node.view(self.Np, 4, self.N_tets)
        self._dQds_view = self._dQds_by_node.view(self.Np, 4, self.N_tets)
        self._dQdt_view = self._dQdt_by_node.view(self.Np, 4, self.N_tets)
        self._dPdx = torch.empty(node_shape, **kwargs)
        self._dPdy = torch.empty(node_shape, **kwargs)
        self._dPdz = torch.empty(node_shape, **kwargs)
        self._divV = torch.empty(node_shape, **kwargs)
        self._metric_p = self.rst_xyz * (-(self.c0**2) * self.rho0)
        self._metric_v = self.rst_xyz * (-1.0 / self.rho0)
        self._use_triton_volume_rhs = (
            self.device.type == "cuda"
            and os.environ.get("EDG_ACOUSTICS_TRITON_VOLUME_RHS", "1") != "0"
        )
        self._use_triton_interior_flux = (
            self.device.type == "cuda"
            and os.environ.get("EDG_ACOUSTICS_TRITON_INTERIOR_FLUX", "1") != "0"
        )
        self._use_triton_boundary_ri = (
            self.device.type == "cuda"
            and os.environ.get("EDG_ACOUSTICS_TRITON_BOUNDARY_RI", "1") != "0"
        )
        self._use_triton_boundary_ade = (
            self.device.type == "cuda"
            and os.environ.get("EDG_ACOUSTICS_TRITON_BOUNDARY_ADE", "1") != "0"
        )
        self._use_batched_derivatives = (
            self.device.type == "cuda"
            and os.environ.get("EDG_ACOUSTICS_BATCHED_DERIVATIVES", "1") != "0"
        )
        self._use_triton_volume_surface_rhs = (
            self.device.type == "cuda"
            and os.environ.get("EDG_ACOUSTICS_TRITON_VOLUME_SURFACE_RHS", "1")
            != "0"
        )
        self._use_scaled_flux_kernels = (
            self.device.type == "cuda"
            and os.environ.get("EDG_ACOUSTICS_SCALED_FLUX_KERNELS", "1") != "0"
            and self._use_triton_interior_flux
            and self._use_triton_boundary_ri
            and self._use_triton_boundary_ade
        )
        self._use_fused_state_accumulation = (
            self.device.type == "cuda"
            and os.environ.get("EDG_ACOUSTICS_FUSED_STATE_ACCUMULATION", "0")
            != "0"
        )
        self._use_triton_derivative_volume = (
            self.device.type == "cuda"
            and os.environ.get("EDG_ACOUSTICS_TRITON_DERIVATIVE_VOLUME", "0")
            != "0"
        )
        self._use_triton_lift_surface = (
            self.device.type == "cuda"
            and os.environ.get("EDG_ACOUSTICS_TRITON_LIFT_SURFACE", "0") != "0"
        )
        self._flux_by_face = torch.empty((4 * self.Nfp, 4 * self.N_tets), **kwargs)
        self._flux_by_face_view = self._flux_by_face.view(4 * self.Nfp, 4, self.N_tets)
        self._surface_by_node = torch.empty((self.Np, 4 * self.N_tets), **kwargs)
        self._surface_view = self._surface_by_node.view(self.Np, 4, self.N_tets)
        self._face_left_packed = torch.empty((4, face_shape[0] * face_shape[1]), **kwargs)
        self._face_right_packed = torch.empty_like(self._face_left_packed)
        self._rhs_by_node_buffers = tuple(
            torch.empty((self.Np, 4 * self.N_tets), **kwargs) for _ in range(2)
        )
        self._rhs_by_node_views = tuple(
            rhs.view(self.Np, 4, self.N_tets) for rhs in self._rhs_by_node_buffers
        )
        self._rhs_buffer_index = 0
        self._cuda_step_graphs = {}

    def cache_static_indices(self):
        """Cache immutable flattened indices and boundary normals."""
        self._vmapM = self.vmapM.to(device=self.device, dtype=torch.long)
        self._vmapP = self.vmapP.to(device=self.device, dtype=torch.long)
        self._vmapM_q = self._build_packed_face_indices(self._vmapM)
        self._vmapP_q = self._build_packed_face_indices(self._vmapP)
        self._nx_flat = self.n_xyz[0].reshape(-1)
        self._ny_flat = self.n_xyz[1].reshape(-1)
        self._nz_flat = self.n_xyz[2].reshape(-1)

        for node in self.BCnode:
            node["map"] = node["map"].to(device=self.device, dtype=torch.long)
            node["vmap"] = node["vmap"].to(device=self.device, dtype=torch.long)
            node["vmap_q"] = self._build_packed_face_indices(node["vmap"])
            node["flux_map_q"] = self._build_packed_flux_indices(node["map"])
            node["nx"] = self._nx_flat[node["map"]]
            node["ny"] = self._ny_flat[node["map"]]
            node["nz"] = self._nz_flat[node["map"]]
            node["fscale"] = self.Fscale.reshape(-1)[node["map"]]

    def cache_boundary_parameters(self):
        """Cache boundary parameters as tensors without changing saved BC metadata."""
        self._BC_cache = []
        for index, paras in enumerate(self.BC.BCpara):
            bcvar = self.BC.BCvar[index]
            cache = {
                "RI": torch.as_tensor(
                    paras["RI"], device=self.device, dtype=device_ini.dtype
                ),
                "RI_value": float(paras["RI"]),
                "simple_RI": "RP" not in paras and "CP" not in paras,
                "boundary_q": torch.empty(
                    (4, bcvar["vn"].numel()),
                    device=self.device,
                    dtype=device_ini.dtype,
                ),
                "boundary_temp": torch.empty_like(bcvar["vn"]),
                "boundary_temp2": torch.empty_like(bcvar["vn"]),
                "boundary_flux": torch.empty(
                    (4, bcvar["vn"].numel()),
                    device=self.device,
                    dtype=device_ini.dtype,
                ),
                "incoming_outgoing": torch.empty_like(bcvar["vn"]),
            }
            if "RP" in paras:
                rp = torch.as_tensor(
                    paras["RP"], device=self.device, dtype=device_ini.dtype
                )
                cache["RP_A"] = rp[0].reshape(-1, 1)
                cache["RP_zeta"] = rp[1].reshape(-1, 1)
                cache["RP_terms"] = torch.empty_like(bcvar["phi"])
                cache["RP_sum"] = torch.empty_like(bcvar["vn"])
            if "CP" in paras:
                cp = torch.as_tensor(
                    paras["CP"], device=self.device, dtype=device_ini.dtype
                )
                cache["CP_B"] = cp[0].reshape(-1, 1)
                cache["CP_C"] = cp[1].reshape(-1, 1)
                cache["CP_alpha"] = cp[2].reshape(-1, 1)
                cache["CP_beta"] = cp[3].reshape(-1, 1)
                cache["CP_terms"] = torch.empty_like(bcvar["kexi1"])
                cache["CP_sum"] = torch.empty_like(bcvar["vn"])
                cache["kexi1_temp"] = torch.empty_like(
                    self.BC.BCvar[index]["kexi1"]
                )
            self._BC_cache.append(cache)

    def _compute_jump(self, field: torch.tensor, out: torch.tensor):
        field_flat = field.reshape(-1)
        torch.index_select(field_flat, 0, self._vmapM, out=self._jump_left)
        torch.index_select(field_flat, 0, self._vmapP, out=self._jump_right)
        torch.sub(self._jump_left, self._jump_right, out=out.reshape(-1))

    def _build_packed_face_indices(self, field_indices: torch.tensor):
        node_ids = torch.div(field_indices, self.N_tets, rounding_mode="floor")
        tet_ids = torch.remainder(field_indices, self.N_tets)
        variable_offsets = (
            torch.arange(4, device=self.device, dtype=torch.long).reshape(4, 1)
            * self.N_tets
        )
        packed_indices = (
            node_ids.reshape(1, -1) * (4 * self.N_tets)
            + variable_offsets
            + tet_ids.reshape(1, -1)
        )
        return packed_indices.reshape(-1)

    def _build_packed_flux_indices(self, face_indices: torch.tensor):
        face_ids = torch.div(face_indices, self.N_tets, rounding_mode="floor")
        tet_ids = torch.remainder(face_indices, self.N_tets)
        variable_offsets = (
            torch.arange(4, device=self.device, dtype=torch.long).reshape(4, 1)
            * self.N_tets
        )
        packed_indices = (
            face_ids.reshape(1, -1) * (4 * self.N_tets)
            + variable_offsets
            + tet_ids.reshape(1, -1)
        )
        return packed_indices.reshape(-1)

    def _next_rhs_buffers(self):
        self._rhs_buffer_index = 1 - self._rhs_buffer_index
        return (
            self._rhs_by_node_buffers[self._rhs_buffer_index],
            self._rhs_by_node_views[self._rhs_buffer_index],
        )

    def _pack_fields_by_node(
        self,
        P: torch.tensor,
        Vx: torch.tensor,
        Vy: torch.tensor,
        Vz: torch.tensor,
    ):
        q_view = self._q_by_node_view
        q_view[:, 0, :].copy_(P)
        q_view[:, 1, :].copy_(Vx)
        q_view[:, 2, :].copy_(Vy)
        q_view[:, 3, :].copy_(Vz)
        return self._q_by_node

    def _compute_packed_derivatives(self, q_by_node: torch.tensor):
        if self._use_batched_derivatives:
            torch.bmm(
                self._D_stack,
                q_by_node.unsqueeze(0).expand(3, -1, -1),
                out=self._dQ_by_derivative,
            )
            return

        torch.mm(self.Dr, q_by_node, out=self._dQdr_by_node)
        torch.mm(self.Ds, q_by_node, out=self._dQds_by_node)
        torch.mm(self.Dt, q_by_node, out=self._dQdt_by_node)

    def _compute_packed_jump(self, q_by_node: torch.tensor):
        q_flat = q_by_node.reshape(-1)
        torch.index_select(
            q_flat, 0, self._vmapM_q, out=self._face_left_packed.reshape(-1)
        )
        torch.index_select(
            q_flat, 0, self._vmapP_q, out=self._face_right_packed.reshape(-1)
        )
        torch.sub(
            self._face_left_packed[1],
            self._face_right_packed[1],
            out=self._dVx.reshape(-1),
        )
        torch.sub(
            self._face_left_packed[2],
            self._face_right_packed[2],
            out=self._dVy.reshape(-1),
        )
        torch.sub(
            self._face_left_packed[3],
            self._face_right_packed[3],
            out=self._dVz.reshape(-1),
        )
        torch.sub(
            self._face_left_packed[0],
            self._face_right_packed[0],
            out=self._dP.reshape(-1),
        )

    def _compute_interior_flux(self, q_by_node: torch.tensor):
        if self._use_triton_interior_flux:
            total_faces = 4 * self.Nfp * self.N_tets
            block_size = 256
            interior_flux_kernel[(triton.cdiv(total_faces, block_size),)](
                q_by_node.reshape(-1),
                self._vmapM_q,
                self._vmapP_q,
                self.flux.cn1s,
                self.flux.cn2s,
                self.flux.cn3s,
                self.flux.cn1n2,
                self.flux.cn1n3,
                self.flux.cn2n3,
                self.flux.n1rho,
                self.flux.n2rho,
                self.flux.n3rho,
                self.flux.csn1rho,
                self.flux.csn2rho,
                self.flux.csn3rho,
                self.Fscale.reshape(-1),
                self._flux_by_face.reshape(-1),
                total_faces,
                self.N_tets,
                4 * self.N_tets,
                self.c0,
                self._use_scaled_flux_kernels,
                BLOCK_SIZE=block_size,
            )
            return

        self._compute_packed_jump(q_by_node)
        flux_view = self._flux_by_face_view
        self.flux.compute_all(
            self._dVx,
            self._dVy,
            self._dVz,
            self._dP,
            flux_view[:, 1, :],
            flux_view[:, 2, :],
            flux_view[:, 3, :],
            flux_view[:, 0, :],
        )

    def _combine_derivative(
        self, variable_index: int, coordinate_index: int, out: torch.tensor
    ):
        torch.mul(
            self.rst_xyz[0, coordinate_index],
            self._dQdr_view[:, variable_index, :],
            out=out,
        )
        out.addcmul_(
            self.rst_xyz[1, coordinate_index],
            self._dQds_view[:, variable_index, :],
        )
        out.addcmul_(
            self.rst_xyz[2, coordinate_index],
            self._dQdt_view[:, variable_index, :],
        )

    def _compute_volume_derivatives(self):
        self._combine_derivative(0, 0, self._dPdx)
        self._combine_derivative(0, 1, self._dPdy)
        self._combine_derivative(0, 2, self._dPdz)

        rx = self.rst_xyz[0, 0]
        sx = self.rst_xyz[1, 0]
        tx = self.rst_xyz[2, 0]
        ry = self.rst_xyz[0, 1]
        sy = self.rst_xyz[1, 1]
        ty = self.rst_xyz[2, 1]
        rz = self.rst_xyz[0, 2]
        sz = self.rst_xyz[1, 2]
        tz = self.rst_xyz[2, 2]

        torch.mul(rx, self._dQdr_view[:, 1, :], out=self._divV)
        self._divV.addcmul_(sx, self._dQds_view[:, 1, :])
        self._divV.addcmul_(tx, self._dQdt_view[:, 1, :])
        self._divV.addcmul_(ry, self._dQdr_view[:, 2, :])
        self._divV.addcmul_(sy, self._dQds_view[:, 2, :])
        self._divV.addcmul_(ty, self._dQdt_view[:, 2, :])
        self._divV.addcmul_(rz, self._dQdr_view[:, 3, :])
        self._divV.addcmul_(sz, self._dQds_view[:, 3, :])
        self._divV.addcmul_(tz, self._dQdt_view[:, 3, :])

    def _compute_volume_rhs(
        self,
        rhs_view: torch.tensor,
        surface_by_node: torch.tensor | None = None,
        q_update: torch.tensor | None = None,
        coefficient: float = 0.0,
    ):
        if self._use_triton_volume_rhs and surface_by_node is not None:
            total_nodes = self.Np * self.N_tets
            block_size = 256
            volume_surface_rhs_kernel[(triton.cdiv(total_nodes, block_size),)](
                self._dQdr_by_node,
                self._dQds_by_node,
                self._dQdt_by_node,
                self._metric_p,
                self._metric_v,
                surface_by_node,
                rhs_view.reshape(self.Np, 4 * self.N_tets),
                q_update if q_update is not None else rhs_view.reshape(self.Np, 4 * self.N_tets),
                total_nodes,
                self.N_tets,
                4 * self.N_tets,
                coefficient,
                q_update is not None,
                BLOCK_SIZE=block_size,
            )
            return

        if self._use_triton_volume_rhs:
            total_nodes = self.Np * self.N_tets
            block_size = 256
            volume_rhs_kernel[(triton.cdiv(total_nodes, block_size),)](
                self._dQdr_by_node,
                self._dQds_by_node,
                self._dQdt_by_node,
                self._metric_p,
                self._metric_v,
                rhs_view.reshape(self.Np, 4 * self.N_tets),
                total_nodes,
                self.N_tets,
                4 * self.N_tets,
                BLOCK_SIZE=block_size,
            )
            return

        rhs_p = rhs_view[:, 0, :]
        rhs_vx = rhs_view[:, 1, :]
        rhs_vy = rhs_view[:, 2, :]
        rhs_vz = rhs_view[:, 3, :]

        torch.mul(self._metric_v[0, 0], self._dQdr_view[:, 0, :], out=rhs_vx)
        rhs_vx.addcmul_(self._metric_v[1, 0], self._dQds_view[:, 0, :])
        rhs_vx.addcmul_(self._metric_v[2, 0], self._dQdt_view[:, 0, :])

        torch.mul(self._metric_v[0, 1], self._dQdr_view[:, 0, :], out=rhs_vy)
        rhs_vy.addcmul_(self._metric_v[1, 1], self._dQds_view[:, 0, :])
        rhs_vy.addcmul_(self._metric_v[2, 1], self._dQdt_view[:, 0, :])

        torch.mul(self._metric_v[0, 2], self._dQdr_view[:, 0, :], out=rhs_vz)
        rhs_vz.addcmul_(self._metric_v[1, 2], self._dQds_view[:, 0, :])
        rhs_vz.addcmul_(self._metric_v[2, 2], self._dQdt_view[:, 0, :])

        torch.mul(self._metric_p[0, 0], self._dQdr_view[:, 1, :], out=rhs_p)
        rhs_p.addcmul_(self._metric_p[1, 0], self._dQds_view[:, 1, :])
        rhs_p.addcmul_(self._metric_p[2, 0], self._dQdt_view[:, 1, :])
        rhs_p.addcmul_(self._metric_p[0, 1], self._dQdr_view[:, 2, :])
        rhs_p.addcmul_(self._metric_p[1, 1], self._dQds_view[:, 2, :])
        rhs_p.addcmul_(self._metric_p[2, 1], self._dQdt_view[:, 2, :])
        rhs_p.addcmul_(self._metric_p[0, 2], self._dQdr_view[:, 3, :])
        rhs_p.addcmul_(self._metric_p[1, 2], self._dQds_view[:, 3, :])
        rhs_p.addcmul_(self._metric_p[2, 2], self._dQdt_view[:, 3, :])

    def _compute_derivative_volume_rhs(
        self,
        q_by_node: torch.tensor,
        rhs_view: torch.tensor,
        surface_by_node: torch.tensor,
        q_update: torch.tensor | None = None,
        coefficient: float = 0.0,
    ):
        total_nodes = self.Np * self.N_tets
        block_size = 128
        derivative_volume_surface_rhs_kernel[(triton.cdiv(total_nodes, block_size),)](
            q_by_node,
            self.Dr,
            self.Ds,
            self.Dt,
            self._metric_p,
            self._metric_v,
            surface_by_node,
            rhs_view.reshape(self.Np, 4 * self.N_tets),
            q_update if q_update is not None else rhs_view.reshape(self.Np, 4 * self.N_tets),
            total_nodes,
            self.N_tets,
            4 * self.N_tets,
            self.Np,
            coefficient,
            q_update is not None,
            BLOCK_SIZE=block_size,
        )

    def _compute_lift_surface(self):
        if self._use_triton_lift_surface:
            total_outputs = self.Np * 4 * self.N_tets
            block_size = 128
            lift_surface_kernel[(triton.cdiv(total_outputs, block_size),)](
                self.lift,
                self._flux_by_face,
                self._surface_by_node,
                total_outputs,
                4 * self.N_tets,
                4 * self.Nfp,
                BLOCK_SIZE=block_size,
            )
            return

        torch.mm(self.lift, self._flux_by_face, out=self._surface_by_node)

    def _compute_boundary_flux(
        self,
        bc_cache: dict,
        node: dict,
        bcvar: dict,
        q_flat: torch.tensor,
        flux_flat: torch.tensor,
    ):
        n_boundary = bcvar["vn"].numel()
        block_size = 256
        if self._use_triton_boundary_ri and bc_cache["simple_RI"]:
            boundary_ri_flux_kernel[(triton.cdiv(n_boundary, block_size),)](
                q_flat,
                node["vmap_q"],
                node["flux_map_q"],
                node["nx"],
                node["ny"],
                node["nz"],
                node["fscale"],
                flux_flat,
                bcvar["vn"],
                bcvar["ou"],
                bcvar["in"],
                n_boundary,
                self.rho0,
                self.c0,
                bc_cache["RI_value"],
                self._use_scaled_flux_kernels,
                BLOCK_SIZE=block_size,
            )
            return

        if self._use_triton_boundary_ade and "CP_B" in bc_cache:
            boundary_rp_cp_flux_kernel[(triton.cdiv(n_boundary, block_size),)](
                q_flat,
                node["vmap_q"],
                node["flux_map_q"],
                node["nx"],
                node["ny"],
                node["nz"],
                node["fscale"],
                bc_cache["RP_A"],
                bc_cache["RP_zeta"],
                bc_cache["CP_B"],
                bc_cache["CP_C"],
                bc_cache["CP_alpha"],
                bc_cache["CP_beta"],
                flux_flat,
                bcvar["vn"],
                bcvar["ou"],
                bcvar["in"],
                bcvar["phi"],
                bcvar["kexi1"],
                bcvar["kexi2"],
                n_boundary,
                self.rho0,
                self.c0,
                bc_cache["RI_value"],
                bcvar["phi"].shape[0],
                bcvar["kexi1"].shape[0],
                self._use_scaled_flux_kernels,
                BLOCK_SIZE=block_size,
            )
            return

        if self._use_triton_boundary_ade and "RP_A" in bc_cache:
            boundary_rp_flux_kernel[(triton.cdiv(n_boundary, block_size),)](
                q_flat,
                node["vmap_q"],
                node["flux_map_q"],
                node["nx"],
                node["ny"],
                node["nz"],
                node["fscale"],
                bc_cache["RP_A"],
                bc_cache["RP_zeta"],
                flux_flat,
                bcvar["vn"],
                bcvar["ou"],
                bcvar["in"],
                bcvar["phi"],
                n_boundary,
                self.rho0,
                self.c0,
                bc_cache["RI_value"],
                bcvar["phi"].shape[0],
                self._use_scaled_flux_kernels,
                BLOCK_SIZE=block_size,
            )
            return

        boundary_q = bc_cache["boundary_q"]
        torch.index_select(
            q_flat,
            0,
            node["vmap_q"],
            out=boundary_q.reshape(-1),
        )
        boundary_p = boundary_q[0]
        boundary_temp = bc_cache["boundary_temp"]
        boundary_flux = bc_cache["boundary_flux"]
        incoming_outgoing = bc_cache["incoming_outgoing"]
        torch.mul(node["nx"], boundary_q[1], out=bcvar["vn"])
        bcvar["vn"].addcmul_(node["ny"], boundary_q[2])
        bcvar["vn"].addcmul_(node["nz"], boundary_q[3])
        torch.mul(boundary_p, 1.0 / (self.rho0 * self.c0), out=boundary_temp)
        torch.add(bcvar["vn"], boundary_temp, out=bcvar["ou"])
        torch.mul(bcvar["ou"], bc_cache["RI"], out=bcvar["in"])

        if "RP_A" in bc_cache:
            phi = bcvar["phi"]
            torch.mul(bc_cache["RP_A"], phi, out=bc_cache["RP_terms"])
            torch.sum(bc_cache["RP_terms"], dim=0, out=bc_cache["RP_sum"])
            bcvar["in"].add_(bc_cache["RP_sum"])
            phi.copy_(bcvar["ou"].unsqueeze(0) - bc_cache["RP_zeta"] * phi)

        if "CP_B" in bc_cache:
            kexi1 = bcvar["kexi1"]
            kexi2 = bcvar["kexi2"]
            torch.mul(bc_cache["CP_B"], kexi1, out=bc_cache["CP_terms"])
            bc_cache["CP_terms"].addcmul_(bc_cache["CP_C"], kexi2)
            torch.sum(bc_cache["CP_terms"], dim=0, out=bc_cache["CP_sum"])
            bcvar["in"].add_(bc_cache["CP_sum"])
            bc_cache["kexi1_temp"].copy_(kexi1)
            kexi1.copy_(
                bcvar["ou"].unsqueeze(0)
                - bc_cache["CP_alpha"] * kexi1
                - bc_cache["CP_beta"] * kexi2
            )
            kexi2.copy_(
                -bc_cache["CP_alpha"] * kexi2
                + bc_cache["CP_beta"] * bc_cache["kexi1_temp"]
            )

        torch.add(bcvar["ou"], bcvar["in"], out=incoming_outgoing)
        torch.mul(boundary_p, 1.0 / self.rho0, out=boundary_temp)
        boundary_temp.add_(incoming_outgoing, alpha=-0.5 * self.c0)
        torch.mul(node["nx"], boundary_temp, out=boundary_flux[1])
        torch.mul(node["ny"], boundary_temp, out=boundary_flux[2])
        torch.mul(node["nz"], boundary_temp, out=boundary_flux[3])

        boundary_temp.copy_(bcvar["vn"])
        boundary_temp.add_(bcvar["ou"], alpha=-0.5)
        boundary_temp.add_(bcvar["in"], alpha=0.5)
        boundary_temp.mul_((self.c0**2) * self.rho0)
        boundary_flux[0].copy_(boundary_temp)
        flux_flat[node["flux_map_q"]] = boundary_flux.reshape(-1)

    def _snapshot_time_state(self):
        return (
            self.Q_flat.clone(),
            [
                {
                    key: value.clone()
                    for key, value in state.items()
                    if torch.is_tensor(value)
                }
                for state in self.BC.BCvar
            ],
        )

    def _restore_time_state(self, snapshot):
        q_snapshot, bc_snapshot = snapshot
        self.Q_flat.copy_(q_snapshot)
        for state, state_snapshot in zip(self.BC.BCvar, bc_snapshot):
            for key, value in state_snapshot.items():
                state[key].copy_(value)

    def _ensure_cuda_step_graph(
        self, chunk_steps: int = 1, record_receivers: bool = False
    ):
        key = (chunk_steps, record_receivers)
        if key in self._cuda_step_graphs:
            return self._cuda_step_graphs[key]
        if not torch.cuda.is_available() or self.Q_flat.device.type != "cuda":
            raise RuntimeError("CUDA graph time stepping requires CUDA tensors.")

        sample_chunk = None
        if record_receivers:
            sample_chunk = torch.empty(
                (chunk_steps, self.rec.shape[1]),
                device=self.device,
                dtype=device_ini.dtype,
            )

        snapshot = self._snapshot_time_state()
        for step_index in range(chunk_steps):
            self.time_integrator.step_dt_packed(self.Q_flat, self.BC)
            if sample_chunk is not None:
                self._sample_receivers(sample_chunk[step_index])
        torch.cuda.synchronize()
        self._restore_time_state(snapshot)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for step_index in range(chunk_steps):
                self.time_integrator.step_dt_packed(self.Q_flat, self.BC)
                if sample_chunk is not None:
                    self._sample_receivers(sample_chunk[step_index])
        self._restore_time_state(snapshot)
        self._cuda_step_graphs[key] = (graph, sample_chunk)
        return self._cuda_step_graphs[key]

    # Static methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def compute_Np(Nx: int):
        """Computes the number of collocation nodes for basis of polynomial degree :attr:`Nx`.

        Args:
            Nx (int): see :attr:`Nx`.

        Returns:
            Np (int): see :attr:`Np`.
        """

        return int((Nx + 1) * (Nx + 2) * (Nx + 3) / 6)

    @staticmethod
    def compute_Nfp(Nx: int):
        """Computes the number of collocation nodes lying on a face of the elements for basis of polynomial degree :attr:`Nx`.

        Args:
            Nx (int): see :attr:`Nx`.

        Returns:
            Nfp (int): see :attr:`Nfp`.
        """

        return int((Nx + 1) * (Nx + 2) / 2)

    @staticmethod
    def compute_Nx_from_Np(Np: int):
        """Computes the  polynomial degree :attr:`Nx` of basis from the number of collocation points :attr:`Np`.

        Args:
            Np (int): see :attr:`Np`.


        Returns:
            Nx (int): see :attr:`Nx`.
        """
        # Since Np is given by:
        #   Np = (Nx + 1)*(Nx + 2)*(Nx + 3)/6
        # to compute Nx from Np we need to solve a third order polynomial equation:
        # Nx^3 + 6Nx^2 + 11Nx + 6(1 - Np) = 0
        polynominal = numpy.polynomial.Polynomial(
            [(6 * (1 - Np)), 11, 6, 1]
        )  # we setup the polynomial
        Nx = int(
            round(polynominal.roots()[-1].real)
        )  # then we just get the roots and extract the root with the largest real component

        return Nx

    @staticmethod
    def check_BC_list(BC_list: dict[str, int], mesh: edg_acoustics.Mesh):
        """Check if BC_list is compatible with mesh.
        Given a mesh with a set of boundary conditions specified in mesh.BC_triangles, check if the list of boundary
        conditions specification, BC_list, is compatible. By compatible we mean that all boundary conditions (keys) in
        BC_list exist in mesh.BC_labels, and vice-versa.

        Args:
            mesh (edg_acoustics.Mesh): see :attr:`mesh`.
            BC_list (dict[str, int]): see :attr:`BC_list`.

        Returns:
            is_compatible (bool): a flag specifying if BC_list is compatible with the mesh or not.
        """
        return BC_list.keys() == mesh.BC_triangles.keys()

    @staticmethod
    def compute_collocation_nodes(
        EToV: torch.tensor, vertices: torch.tensor, Nx: int, dim: int = 3
    ):
        """Compute reference element :math:`(r, s, t)` coordinates of collocation points :attr:`rst` and the physical domain :attr:`xyz` coordinates
        for each element.

        Args:
            EToV (torch.tensor): See :any:`edg_acoustics.Mesh.EToV`.
            vertices (torch.tensor): see :any:`edg_acoustics.Mesh.vertices`.
            Nx (int): see :attr:`Nx`.
            dim (int): see :attr:`dim`.

        Returns:
            rst (torch.tensor): see :attr:`rst`.

            xyz (torch.tensor): see :attr:`xyz`.
        """
        # These are so called Fekete points (low, close to optimal, Lebesgue constant) on simplices of dimension
        # self.dim and maximum polynomial degree to interpolate over these nodes (determines the number of nodes).
        rst = modepy.warp_and_blend_nodes(dim, Nx)
        rst = torch.from_numpy(rst).to(device_ini.device)

        # Compute the xyz coordinates for each element from the reference domain coordinates and the element's vertices

        # Start by allocating space for the xyz-coordinates for collocation points on each element
        xyz = torch.zeros(
            [dim, rst.shape[1], EToV.shape[1]],
            device=device_ini.device,
            dtype=device_ini.dtype,
        )

        # Then get shorter references for the indices of the references for each of the nodes of the elements
        # get the indices of the first node of each element
        vertex_0_idx = EToV[0, :]
        # get the indices of the second node of each element
        vertex_1_idx = EToV[1, :]
        # get the indices of the third node of each element
        vertex_2_idx = EToV[2, :]
        # get the indices of the fourth node of each element
        vertex_3_idx = EToV[3, :]

        # Get shorter references for the r, s, and t coordinates, and the sum r + s + t
        rst_sum = rst.sum(axis=0).reshape([-1, 1])
        r = rst[0, :].reshape([-1, 1])
        s = rst[1, :].reshape([-1, 1])
        t = rst[2, :].reshape([-1, 1])

        # Compute the xyz coordinates
        xyz[0] = 0.5 * (
            -(1.0 + rst_sum) * vertices[0, vertex_0_idx]
            + (1.0 + r) * vertices[0, vertex_1_idx]
            + (1.0 + s) * vertices[0, vertex_2_idx]
            + (1.0 + t) * vertices[0, vertex_3_idx]
        )
        xyz[1] = 0.5 * (
            -(1.0 + rst_sum) * vertices[1, vertex_0_idx]
            + (1.0 + r) * vertices[1, vertex_1_idx]
            + (1.0 + s) * vertices[1, vertex_2_idx]
            + (1.0 + t) * vertices[1, vertex_3_idx]
        )
        xyz[2] = 0.5 * (
            -(1.0 + rst_sum) * vertices[2, vertex_0_idx]
            + (1.0 + r) * vertices[2, vertex_1_idx]
            + (1.0 + s) * vertices[2, vertex_2_idx]
            + (1.0 + t) * vertices[2, vertex_3_idx]
        )

        # Return the computed coordinates
        return rst.cpu().numpy(), xyz

    @staticmethod
    def compute_van_der_monde_matrix(
        Nx: int, rst: numpy.ndarray, dim: int = 3
    ) -> torch.tensor:
        """Compute vandermonde matrix of the orthonormal basis functions on the reference simplex element. Polynomial basis
            can exactly represent polynomials up to degree :attr:`Nx`. Consider the set of :attr:`~.AcousticsSimulation.Np` 3D nodes, with the coordinates of each node :math:`i` equal to
            :math:`(r_{i}, s_{i}, t_{i})`, in :attr:`.rst`, and the set of :math:`m` orthonormal basis functions,
            the vandermonde matrix will be :math:`V_{i,j} = f_{j}(r_{i}, s_{i}, t_{i})`.

        Args:
            Nx (int): see :attr:`Nx`.
            rst (torch.tensor): see :attr:`rst`.
            dim (int): see :attr:`dim`.

        Returns:
            V (torch.tensor): see :attr:`V`.
        """
        # Compute the orthonormal polynomial basis of degree Nx and geometric dimension dim
        simplex_basis = modepy.simplex_onb(dim, Nx)

        # Compute van der Monde matrix of simplex_basis over the nodes in rst
        return torch.as_tensor(
            modepy.vandermonde(simplex_basis, rst),
            device=device_ini.device,
            dtype=device_ini.dtype,
        )  # type: ignore

    @staticmethod
    def compute_derivative_matrix(Nx: int, rst: numpy.ndarray, dim: int = 3):
        """Compute the derivative matrix on the reference element.

        Args:
            Nx (int): see :attr:`Nx`.
            rst (torch.tensor): see :attr:`rst`.
            dim (int): see :attr:`dim`.
        Returns:
            Dr (torch.tensor): see :attr:`Dr`.
            Ds (torch.tensor): see :attr:`Ds`.
            Dt (torch.tensor): see :attr:`Dt`.
        """
        # Compute the orthonormal polynomial basis of degree Nx and geometric dimension dim
        simplex_basis = modepy.simplex_onb(dim, Nx)

        # Compute the grad of orthonormal polynomial basis of degree Nx and geometric dimension dim
        grad_simplex_basis = modepy.grad_simplex_onb(dim, Nx)

        # Compute differentiation matrix of simplex_basis over the nodes in rst, return a tuple D
        D = modepy.differentiation_matrices(simplex_basis, grad_simplex_basis, rst)

        # Return d/dr, d/ds and d/dt matrices
        return (
            torch.as_tensor(D[0], device=device, dtype=device_ini.dtype),
            torch.as_tensor(D[1], device=device, dtype=device_ini.dtype),
            torch.as_tensor(D[2], device=device, dtype=device_ini.dtype),
        )

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def compute_Fmask(rst: torch.tensor, node_tol: float):
        """Find all the :attr:``Nfp`` face nodes that lie on each surface.

        Args:
            rst (torch.tensor): see :attr:`rst`.
            node_tol (float): see :attr:`node_tolerance`.

        Returns:
            Fmask (torch.tensor): see :attr:`Fmask`.
        """
        Np = rst.shape[1]
        Nx = AcousticsSimulation.compute_Nx_from_Np(
            Np
        )  # get the polynomial degree of approximation
        Nfp = AcousticsSimulation.compute_Nfp(
            Nx
        )  # get the number of collocation points per face

        Fmask = torch.zeros([4, Nfp], device=device, dtype=torch.int32)

        # Find all the nodes that lie on each surface
        Fmask[0] = torch.nonzero(torch.abs(1 + rst[2]) < node_tol).flatten()
        Fmask[1] = torch.nonzero(torch.abs(1 + rst[1]) < node_tol).flatten()
        Fmask[2] = torch.nonzero(torch.abs(1 + rst.sum(axis=0)) < node_tol).flatten()
        Fmask[3] = torch.nonzero(torch.abs(1 + rst[0]) < node_tol).flatten()

        return Fmask

    @staticmethod
    def compute_lift(V: torch.tensor, rst: torch.tensor, Fmask: torch.tensor):
        """Compute the lift matrix.

        Args:
            V (torch.tensor): see :attr:`V`.
            rst (torch.tensor): see :attr:`rst`.
            Fmask (torch.tensor): see :attr:`Fmask`.
        Returns:
            lift (torch.tensor): see :attr:`lift`.
        """
        Np = V.shape[1]  # get the number of collocation points
        Nx = AcousticsSimulation.compute_Nx_from_Np(
            Np
        )  # get the polynomial degree of approximation
        Nfp = AcousticsSimulation.compute_Nfp(
            Nx
        )  # the number of nodes per surface for basis of polynomial degree Nx

        Emat = torch.zeros(
            [Np, Nfp * 4], device=device_ini.device, dtype=device_ini.dtype
        )
        faceR = torch.zeros([1, Nfp], device=device_ini.device, dtype=device_ini.dtype)
        faceS = torch.zeros([1, Nfp], device=device_ini.device, dtype=device_ini.dtype)

        for face in range(4):
            if face == 0:
                faceR = rst[0, Fmask[0]]
                faceS = rst[1, Fmask[0]]

            elif face == 1:
                faceR = rst[0, Fmask[1]]
                faceS = rst[2, Fmask[1]]

            elif face == 2:
                faceR = rst[1, Fmask[2]]
                faceS = rst[2, Fmask[2]]

            else:
                faceR = rst[1, Fmask[3]]
                faceS = rst[2, Fmask[3]]

            simplex_basis = modepy.simplex_onb(2, Nx)
            vandermondeFace = modepy.vandermonde(
                simplex_basis, numpy.vstack((faceR.cpu().numpy(), faceS.cpu().numpy()))
            )
            vandermondeFace = numpy.asarray(
                vandermondeFace
            )  # just to make sure it is a numpy array to avoid errors
            massFace = numpy.linalg.inv(vandermondeFace @ (vandermondeFace.transpose()))

            Emat[Fmask[face], face * Nfp : (face + 1) * Nfp] += (
                torch.as_tensor(
                    massFace, device=device_ini.device, dtype=device_ini.dtype
                )
            )

        return V @ (V.t() @ Emat)

    @staticmethod
    def geometric_factors_3d(
        xyz: torch.tensor, Dr: torch.tensor, Ds: torch.tensor, Dt: torch.tensor
    ):
        """Compute the metric elements for the local mappings of the elements.

        Args:
            xyz (torch.tensor): see :attr:`xyz`.
            Dr (torch.tensor): see :attr:`Dr`.
            Ds (torch.tensor): see :attr:`Ds`.
            Dt (torch.tensor): see :attr:`Dt`.

        Returns:
            rst_xyz (torch.tensor): see :attr:`rst_xyz`.
            J (torch.tensor): see :attr:`J`.
        """
        Np = xyz.shape[1]  # the number of collocation points
        N_tets = xyz.shape[2]  # the number of elements

        # Compute the derivatives of the physical coordinates at the nodal points
        # x
        xr = Dr @ xyz[0]
        xs = Ds @ xyz[0]
        xt = Dt @ xyz[0]
        # y
        yr = Dr @ xyz[1]
        ys = Ds @ xyz[1]
        yt = Dt @ xyz[1]
        # z
        zr = Dr @ xyz[2]
        zs = Ds @ xyz[2]
        zt = Dt @ xyz[2]

        # Compute the Jacobian determinant of the coordinate transformation
        J = (
            xr * (ys * zt - zs * yt)
            - yr * (xs * zt - zs * xt)
            + zr * (xs * yt - ys * xt)
        )

        # Compute the derivates of the local coordinates at the nodal points
        rst_xyz = torch.zeros(
            [3, 3, Np, N_tets], device=device, dtype=device_ini.dtype
        )  # pre-allocate memory space

        # r
        rst_xyz[0, 0] = (ys * zt - zs * yt) / J
        rst_xyz[0, 1] = -(xs * zt - zs * xt) / J
        rst_xyz[0, 2] = (xs * yt - ys * xt) / J
        # s
        rst_xyz[1, 0] = -(yr * zt - zr * yt) / J
        rst_xyz[1, 1] = (xr * zt - zr * xt) / J
        rst_xyz[1, 2] = -(xr * yt - yr * xt) / J
        # t
        rst_xyz[2, 0] = (yr * zs - zr * ys) / J
        rst_xyz[2, 1] = -(xr * zs - zr * xs) / J
        rst_xyz[2, 2] = (xr * ys - yr * xs) / J

        return rst_xyz, J

    @staticmethod
    def normals_3d(
        xyz: torch.tensor,
        rst_xyz: torch.tensor,
        J: torch.tensor,
        Fmask: torch.tensor,
    ):
        """Compute outward pointing normals at element's faces as well as surface Jacobians.

        Args:
            xyz (torch.tensor): see :attr:`xyz`.
            rst_xyz (torch.tensor): see :attr:`rst_xyz`.
            J (torch.tensor): see :attr:`J`.
            Fmask (torch.tensor): see :attr:`Fmask`.

        Returns:
            n_xyz (torch.tensor): see :attr:`n_xyz`.
            sJ (torch.tensor): see :attr:`sJ`.
        """
        N_tets = xyz.shape[2]  # number of elements
        Np = xyz.shape[1]  # number of collocation points
        Nx = AcousticsSimulation.compute_Nx_from_Np(
            Np
        )  # get the polynomial degree of approximation
        Nfp = AcousticsSimulation.compute_Nfp(
            Nx
        )  # the number of nodes per surface for basis of polynomial degree Nx

        # Extract the transformation derivatives over the faces
        # the structure is the same as for rst_xyz
        frst_xyz = rst_xyz[:, :, Fmask.reshape(-1), :]

        # Construct the normals
        n_xyz = torch.zeros(
            [3, 4 * Nfp, N_tets],
            device=device_ini.device,
            dtype=device_ini.dtype,
        )  # allocate memory space

        face_0_idx = torch.arange(0, Nfp, device=device_ini.device)
        face_1_idx = torch.arange(Nfp, 2 * Nfp, device=device_ini.device)
        face_2_idx = torch.arange(2 * Nfp, 3 * Nfp, device=device_ini.device)
        face_3_idx = torch.arange(3 * Nfp, 4 * Nfp, device=device_ini.device)

        # Face 0
        n_xyz[0, face_0_idx, :] = -frst_xyz[2, 0, face_0_idx, :]  # nx
        n_xyz[1, face_0_idx, :] = -frst_xyz[2, 1, face_0_idx, :]  # ny
        n_xyz[2, face_0_idx, :] = -frst_xyz[2, 2, face_0_idx, :]  # nz

        # Face 1
        n_xyz[0, face_1_idx, :] = -frst_xyz[1, 0, face_1_idx, :]  # nx
        n_xyz[1, face_1_idx, :] = -frst_xyz[1, 1, face_1_idx, :]  # ny
        n_xyz[2, face_1_idx, :] = -frst_xyz[1, 2, face_1_idx, :]  # nz

        # Face 2
        n_xyz[0, face_2_idx, :] = (
            frst_xyz[0, 0, face_3_idx, :]
            + frst_xyz[1, 0, face_3_idx, :]
            + frst_xyz[2, 0, face_3_idx, :]
        )  # nx
        n_xyz[1, face_2_idx, :] = (
            frst_xyz[0, 1, face_3_idx, :]
            + frst_xyz[1, 1, face_3_idx, :]
            + frst_xyz[2, 1, face_3_idx, :]
        )  # ny
        n_xyz[2, face_2_idx, :] = (
            frst_xyz[0, 2, face_3_idx, :]
            + frst_xyz[1, 2, face_3_idx, :]
            + frst_xyz[2, 2, face_3_idx, :]
        )  # nz

        # Face 3
        n_xyz[0, face_3_idx, :] = -frst_xyz[0, 0, face_3_idx, :]  # nx
        n_xyz[1, face_3_idx, :] = -frst_xyz[0, 1, face_3_idx, :]  # ny
        n_xyz[2, face_3_idx, :] = -frst_xyz[0, 2, face_3_idx, :]  # nz

        # Normalized the normal vectors
        norm_n = torch.sqrt(
            (n_xyz[0, :, :] ** 2) + (n_xyz[1, :, :] ** 2) + (n_xyz[2, :, :] ** 2)
        )  # vector norm of normal vectors
        n_xyz = (
            n_xyz / norm_n
        )  # this does broadcasting over all the 3 direction x, y, z

        # Compute the face Jacobians
        sJ = norm_n * J[Fmask.reshape(-1)]

        return n_xyz, sJ

    @staticmethod
    @profile
    def build_maps_3d(
        xyz: torch.tensor,
        EToE: torch.tensor,
        EToF: torch.tensor,
        Fmask: torch.tensor,
        node_tol: float,
    ):
        """Find connectivity for nodes given per surface in all elements

        Args:
            xyz (torch.tensor): see :attr:`xyz`.
            EToE (torch.tensor): see :any:`edg_acoustics.Mesh.EToE`.
            EToF (torch.tensor): see :any:`edg_acoustics.Mesh.EToF`.
            Fmask (torch.tensor): see :attr:`Fmask`.
            node_tol (float): see :attr:`node_tolerance`.
        Returns:
            vmapM (torch.tensor): see :attr:`vmapM`.
            vmapP (torch.tensor): see :attr:`vmapP`.
        """

        N_tets = xyz.shape[2]  # number of elements
        Np = xyz.shape[1]  # number of collocation points
        Nx = AcousticsSimulation.compute_Nx_from_Np(
            Np
        )  # get the polynomial degree of approximation
        Nfp = AcousticsSimulation.compute_Nfp(
            Nx
        )  # the number of nodes per surface for basis of polynomial degree Nx
        nodeids = torch.arange(N_tets * Np).reshape(Np, N_tets).to(device_ini.device)

        vmapM = torch.zeros(
            [4, Nfp, N_tets], device=device_ini.device, dtype=torch.int32
        )
        vmapP = torch.zeros(
            [4, Nfp, N_tets], device=device_ini.device, dtype=torch.int32
        )
        tmp = torch.ones(Nfp, device=device_ini.device, dtype=torch.int32)
        D = torch.zeros([Nfp, Nfp], device=device_ini.device, dtype=device_ini.dtype)

        xV = xyz[0].reshape(-1)  # flatten the x-coordinates
        yV = xyz[1].reshape(-1)
        zV = xyz[2].reshape(-1)

        # 配置grid和block大小
        BLOCK_SIZE = 32
        grid = lambda meta: (triton.cdiv(N_tets, BLOCK_SIZE),)

        # 调用triton kernel
        # build_maps_kernel[grid](
        #     nodeids.contiguous(),
        #     Fmask.contiguous(),
        #     vmapM.contiguous(),
        #     N_tets,
        #     Nfp,
        #     BLOCK_SIZE,
        # )
        for ke in range(N_tets):
            for face in range(4):
                vmapM[face, :, ke] = nodeids[
                    Fmask[face], ke
                ]  # find index of face nodes with respect to volume node ordering

        for ke in range(N_tets):
            for face in range(4):
                # find neighbor
                ke2 = EToE[face, ke]
                face2 = EToF[face, ke]

                # find find volume node numbers of left and right nodes
                vidM = vmapM[face, :, ke]
                vidP = vmapM[face2, :, ke2]

                xM = torch.outer(xV[vidM], tmp)  # viewing
                yM = torch.outer(yV[vidM], tmp)  # viewing
                zM = torch.outer(zV[vidM], tmp)  # viewing

                xP = torch.outer(xV[vidP], tmp).t()  # viewing
                yP = torch.outer(yV[vidP], tmp).t()  # viewing
                zP = torch.outer(zV[vidP], tmp).t()  # viewing

                D = (xM - xP) ** 2 + (yM - yP) ** 2 + (zM - zP) ** 2

                idMP = torch.nonzero(torch.abs(D) < node_tol).t()

                vmapP[face, idMP[0], ke] = vmapM[face2, idMP[1], ke2]

        return vmapM.reshape(-1), vmapP.reshape(-1)

    @staticmethod
    def ismember_col(a: torch.tensor, b: torch.tensor):
        """find the indices of the columns of a that are in b

        Args:
            a (torch.tensor): matrix a to be checked
            b (torch.tensor): matrix b to be checked against

        Returns:
            indices (torch.tensor): boolean indices of the columns of a that are in b
        """
        _, rev = torch.unique(
            torch.concatenate((a, b.t()), dim=1), dim=1, return_inverse=True
        )  # The indices to reconstruct the original array from the unique array
        # Split the index
        b_rev = rev[a.shape[1] :]
        a_rev = rev[: a.shape[1]]
        # Return the result:
        return torch.isin(a_rev, b_rev)

    @staticmethod
    def build_BCmaps_3d(
        BC_list: dict[str, int],
        EToV: torch.tensor,
        vmapM: torch.tensor,
        BC_triangles: dict[str, torch.tensor],
        Nx: int,
    ):
        """Build specialized nodal maps for various types of boundary conditions,specified in BC_list


        Args:
            BC_list (dict[str, int]): see :attr:`BC_list`.
            EToV (torch.tensor): see :any:`edg_acoustics.Mesh.EToV`.
            vmapM (torch.tensor): see :attr:`vmapM`.
            BC_triangles (dict[str, torch.tensor]): see :attr:`edg_acoustics.mesh.Mesh.BC_triangles`.
            Nx (int): see :attr:`Nx`.
        Returns:
            BCnode (list[dict]): see :attr:`BCnode`.
        """
        Nfp = AcousticsSimulation.compute_Nfp(
            Nx
        )  # the number of nodes per surface for basis of polynomial degree Nx
        N_tets = EToV.shape[1]
        BCType = torch.zeros([4, N_tets], dtype=torch.int32)
        VNUM = torch.tensor([[1, 2, 3], [1, 2, 4], [2, 3, 4], [1, 3, 4]]) - 1
        BCnode = []
        for BCname, BClabel in BC_list.items():
            BCnode.append({"label": BClabel})
            tri, _ = BC_triangles[BCname].sort(axis=1)

            for indexl in range(4):
                Face, _ = torch.sort(EToV[VNUM[indexl]], axis=0)
                K_ = AcousticsSimulation.ismember_col(Face, tri)
                BCType[indexl, K_] = BClabel
        # BCType = BCType.repeat(Nfp)
        BCType = BCType.repeat_interleave(Nfp, dim=0)

        for i in range(len(BC_list)):
            BCnode[i]["map"] = (
                torch.nonzero(BCType.reshape(-1) == BCnode[i]["label"])
                .t()[0]
                .to(device_ini.device)
            )
            BCnode[i]["vmap"] = vmapM[BCnode[i]["map"]]

        return BCnode

    @staticmethod
    def diameter_3d(Fscale: torch.tensor):
        """Compute the minimum diameter of the inscribed spheres in all elements.

        Args:
            Fscale (torch.tensor): see :attr:`Fscale`.
        Returns:
            diameter (float): minimum diameter of the inscribed spheres in all elements.
        """
        Nfp = int(Fscale.shape[0] / 4)
        AtoV = 3 / 2 * Fscale[[0, Nfp, 2 * Nfp, 3 * Nfp]].sum(axis=0)
        diameter = 6 / AtoV
        return diameter.min().item()

    @profile
    def grad_3d(self, U: torch.tensor, axis: str):
        """Compute partial derivative dU/dx, dU/dy, dU/dz, or gradient dU/dx + dU/dy + dU/dz

        Args:
            U (torch.tensor): ``[Np, N_tets]`` the acoustic variables that needs to be differentiated.
            axis (str): the axis to be differentiated w.r.t, e.g. 'x', 'y', 'z', 'xyz'

        Returns:
            dUdx (torch.tensor): ``[Np, N_tets]`` derivatives :math:`\\frac{\\partial U}{\\partial x}` at every nodal point, if axis is 'x'.
            dUdy (torch.tensor): ``[Np, N_tets]`` derivatives :math:`\\frac{\\partial U}{\\partial y}` at every nodal point, if axis is 'y'.
            dUdz (torch.tensor): ``[Np, N_tets]`` derivatives :math:`\\frac{\\partial U}{\\partial z}` at every nodal point, if axis is 'z'.
            Tuple of gradient (torch.tensor): ``[Np, N_tets]`` gradient (:math:`\\frac{\\partial U}{\\partial x}, \\frac{\\partial U}{\\partial y}, \\frac{\\partial U}{\\partial z}`), if axis is 'xyz'.
        """
        dUdr = self.Dr @ U
        dUds = self.Ds @ U
        dUdt = self.Dt @ U
        if axis == "x":
            return (
                self.rst_xyz[0, 0] * dUdr
                + self.rst_xyz[1, 0] * dUds
                + self.rst_xyz[2, 0] * dUdt
            )
        elif axis == "y":
            return (
                self.rst_xyz[0, 1] * dUdr
                + self.rst_xyz[1, 1] * dUds
                + self.rst_xyz[2, 1] * dUdt
            )
        elif axis == "z":
            return (
                self.rst_xyz[0, 2] * dUdr
                + self.rst_xyz[1, 2] * dUds
                + self.rst_xyz[2, 2] * dUdt
            )
        elif axis == "xyz":
            return (
                self.rst_xyz[0, 0] * dUdr
                + self.rst_xyz[1, 0] * dUds
                + self.rst_xyz[2, 0] * dUdt,
                self.rst_xyz[0, 1] * dUdr
                + self.rst_xyz[1, 1] * dUds
                + self.rst_xyz[2, 1] * dUdt,
                self.rst_xyz[0, 2] * dUdr
                + self.rst_xyz[1, 2] * dUds
                + self.rst_xyz[2, 2] * dUdt,
            )
        else:
            raise ValueError(f"Invalid axis: {axis}")

    @staticmethod
    def locate_simplex(
        node_coordinates: numpy.ndarray,
        EToV: numpy.ndarray,
        rec: numpy.ndarray,
        methodLocate: str = "scipy",
    ):
        """Locate the simplices containing the sample points.

        Args:
            node_coordinates (numpy.ndarray): see :attr:`edg_acoustics.Mesh.vertices`.
            EToV (torch.tensor): see :attr:`edg_acoustics.Mesh.EToV`.
            rec (torch.tensor): An (N_rec x 3) array containing the (x, y, z) coordinates of N_rec microphone locations.
            methodLocate (str): search method to locate the simplices containing the sample points. Available methods are 'scipy' and 'brute_force'.
                brutal force approach, adopted from https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not/60745339#60745339


        Returns:
            nodeindex (torch.tensor): indices of simplices containing the N_rec microphone points
        """
        if methodLocate == "scipy":
            tri = Delaunay(node_coordinates.T, qhull_options="QJ")
            tri.simplices = EToV.T  # type: ignore
            tri.nsimplex = EToV.shape[1]  # type: ignore

            nodeindex = tri.find_simplex(rec.T)  # type: ignore

        elif methodLocate == "brute_force":
            EToV = EToV.T
            ori = node_coordinates.T[EToV[:, 0], :]
            v1 = node_coordinates.T[EToV[:, 1], :] - ori
            v2 = node_coordinates.T[EToV[:, 2], :] - ori
            v3 = node_coordinates.T[EToV[:, 3], :] - ori
            n_tet = len(EToV)
            v1r = v1.T.reshape((3, 1, n_tet))
            v2r = v2.T.reshape((3, 1, n_tet))
            v3r = v3.T.reshape((3, 1, n_tet))
            mat = numpy.concatenate((v1r, v2r, v3r), axis=1)
            inv_mat = numpy.linalg.inv(
                mat.T
            ).T  # https://stackoverflow.com/a/41851137/12056867
            N_rec = rec.shape[1]
            orir = numpy.repeat(ori[:, :, numpy.newaxis], N_rec, axis=2)
            newp = numpy.einsum("imk,kmj->kij", inv_mat, rec - orir)
            val = (
                numpy.all(newp >= 0, axis=1)
                & numpy.all(newp <= 1, axis=1)
                & (numpy.sum(newp, axis=1) <= 1)
            )
            id_tet, id_p = numpy.nonzero(val)
            nodeindex = -numpy.ones(N_rec, dtype=id_tet.dtype)  # Sentinel value
            nodeindex[id_p] = id_tet
        else:
            raise ValueError(
                f"{methodLocate} is not an available search method, see documentation for available methods"
            )

        return nodeindex

    # instance method -------------------------------------------------------------------------------------------------------

    def sample3D(self, methodLocate: str = "scipy"):
        """Compute interpolation weights required to interpolate the nodal data to the sample (i.e., microphone location)

        Args:
            methodLocate (str): search method to locate the simplices containing the sample points. Available methods are 'scipy' and 'brute_force'.
                brutal force approach, adopted from https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not/60745339#60745339

        Returns:
            sampleWeight (torch.tensor): ``[N_rec, Np]`` interpolation weights required to interpolate the nodal data to the sample (i.e., microphone location).
            nodeindex (torch.tensor): ``[N_rec, ]`` index of simplices that contatin the sample (microphone) points.
        """
        nodeindex = AcousticsSimulation.locate_simplex(
            self.mesh.vertices,
            torch.Tensor.numpy(self.mesh.EToV.cpu()),
            self.rec,
            methodLocate,
        )

        old_nodes = self.xyz[
            :, :, nodeindex
        ]  # old_nodes.shape = (3, Np, N_rec) #type: ignore
        simplex_basis = modepy.simplex_onb(self.dim, self.Nx)
        v_new = modepy.vandermonde(simplex_basis, self.rec)
        sampleWeight = numpy.zeros([self.rec.shape[1], len(simplex_basis)])

        for i in range(old_nodes.shape[2]):
            v_old = modepy.vandermonde(
                simplex_basis, torch.Tensor.numpy(old_nodes[:, :, i].cpu())
            )
            sampleWeight[i] = v_new[i] @ numpy.linalg.inv(v_old)  # type: ignore

        return (
            torch.from_numpy(sampleWeight).to(self.device).to(device_ini.dtype),
            nodeindex,
        )

    def init_IC(self, IC: edg_acoustics.InitialCondition):
        """setup the initial condition and save it to the :class:`AcousticsSimulation` class.

        Args:
            IC (edg_acoustics.InitialCondition): the initial condition object.
        """
        self.IC = IC
        P = self.IC.Pinit(self.xyz).to(device=self.device, dtype=device_ini.dtype)
        Vx = self.IC.VXinit(self.xyz).to(device=self.device, dtype=device_ini.dtype)
        Vy = self.IC.VYinit(self.xyz).to(device=self.device, dtype=device_ini.dtype)
        Vz = self.IC.VZinit(self.xyz).to(device=self.device, dtype=device_ini.dtype)
        self.Q_flat = torch.empty(
            (self.Np, 4 * self.N_tets), device=self.device, dtype=device_ini.dtype
        )
        self.Q = self.Q_flat.view(self.Np, 4, self.N_tets)
        self.P = self.Q[:, 0, :]
        self.Vx = self.Q[:, 1, :]
        self.Vy = self.Q[:, 2, :]
        self.Vz = self.Q[:, 3, :]
        self.P.copy_(P)
        self.Vx.copy_(Vx)
        self.Vy.copy_(Vy)
        self.Vz.copy_(Vz)

    def init_BC(self, BC):
        """load the boundary condition and save it to the :class:`AcousticsSimulation` class.

        Args:
            BC (edg_acoustics.BoundaryCondition): the boundary condition object.
        """
        self.BC = BC
        self.cache_boundary_parameters()

    def init_rec(self, rec: numpy.ndarray, methodLocate: str = "scipy"):
        """load the receiver locations and save it to the :class:`AcousticsSimulation` class.
        Then compute the interpolation weights required to interpolate the nodal data to the sample (i.e., microphone location).

        Args:
            rec (torch.tensor): An ``[3, N_rec]`` array containing the (x, y, z) coordinates of N_rec microphone locations.
            methodLocate (str): search method to locate the simplices containing the sample points. Available methods are 'scipy' and 'brute_force'.
                brutal force approach, adopted from https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not/60745339
        """
        self.rec = rec
        self.sampleWeight, self.nodeindex = self.sample3D(methodLocate)
        normalized_nodeindex = numpy.mod(self.nodeindex, self.N_tets)
        self._nodeindex_tensor = torch.as_tensor(
            normalized_nodeindex, device=self.device, dtype=torch.long
        )
        self._sample_values = torch.empty(
            (self.Np, self.rec.shape[1]), device=self.device, dtype=device_ini.dtype
        )
        self._sample_output = torch.empty(
            (self.rec.shape[1],), device=self.device, dtype=device_ini.dtype
        )

    def _sample_receivers(self, out: torch.tensor):
        torch.index_select(self.P, 1, self._nodeindex_tensor, out=self._sample_values)
        torch.sum(self.sampleWeight * self._sample_values.T, dim=1, out=out)

    def init_Flux(self, Flux):
        """load the interior flux  calculation and save it to the :class:`AcousticsSimulation` class.

        Args:
            Flux (edg_acoustics.Flux): the flux object."""
        self.flux = Flux

    def init_TimeIntegrator(self, time_integrator: edg_acoustics.TimeIntegrator):
        """load the time integrator to be used to and save it to the :class:`AcousticsSimulation` class."""
        self.time_integrator = time_integrator

    @profile
    def RHS_operator(
        self,
        P: torch.tensor,
        Vx: torch.tensor,
        Vy: torch.tensor,
        Vz: torch.tensor,
        BCvar: list[dict],
    ):
        """Compute the right-hand side of the linear acoustic equations with the DG discretization.

        Args:
            P (torch.tensor): Pressure field.
            Vx (torch.tensor): Velocity field in the x-direction.
            Vy (torch.tensor): Velocity field in the y-direction.
            Vz (torch.tensor): Velocity field in the z-direction.
            BCvar (list[dict]): see :data:`edg_acoustics.AbsorbBC.BCvar`.

        Returns:
            RHS_P (torch.tensor): Right-hand side values for pressure.
            RHS_Vx (torch.tensor): Right-hand side values for velocity in the x-direction.
            RHS_Vy (torch.tensor): Right-hand side values for velocity in the y-direction.
            RHS_Vz (torch.tensor): Right-hand side values for velocity in the z-direction.
            BCvar (list[dict]): updated boundary condition variables.
        """

        q_by_node = self._pack_fields_by_node(P, Vx, Vy, Vz)
        rhs_by_node, BCvar = self.RHS_operator_packed(q_by_node, BCvar)
        rhs_view = rhs_by_node.view(self.Np, 4, self.N_tets)
        return (
            rhs_view[:, 0, :],
            rhs_view[:, 1, :],
            rhs_view[:, 2, :],
            rhs_view[:, 3, :],
            BCvar,
        )

    def RHS_operator_packed(
        self,
        q_by_node: torch.tensor,
        BCvar: list[dict],
        q_accumulate: torch.tensor | None = None,
        accumulate_coefficient: float = 0.0,
    ):
        """Compute packed RHS for ``q_by_node`` shaped ``[Np, 4 * N_tets]``."""

        RHS_Q, RHS_Q_view = self._next_rhs_buffers()
        RHS_P = RHS_Q_view[:, 0, :]
        RHS_Vx = RHS_Q_view[:, 1, :]
        RHS_Vy = RHS_Q_view[:, 2, :]
        RHS_Vz = RHS_Q_view[:, 3, :]
        if not self._use_triton_derivative_volume:
            self._compute_packed_derivatives(q_by_node)
        self._compute_interior_flux(q_by_node)

        flux_flat = self._flux_by_face.reshape(-1)
        flux_view = self._flux_by_face_view

        q_flat = q_by_node.reshape(-1)
        for index, bc_cache in enumerate(self._BC_cache):
            self._compute_boundary_flux(
                bc_cache,
                self.BCnode[index],
                BCvar[index],
                q_flat,
                flux_flat,
            )

        if not self._use_scaled_flux_kernels:
            flux_view.mul_(self.Fscale.unsqueeze(1))

        self._compute_lift_surface()
        if self._use_triton_volume_surface_rhs and self._use_triton_volume_rhs:
            if self._use_triton_derivative_volume:
                self._compute_derivative_volume_rhs(
                    q_by_node,
                    RHS_Q_view,
                    self._surface_by_node,
                    q_accumulate,
                    accumulate_coefficient,
                )
            else:
                self._compute_volume_rhs(
                    RHS_Q_view,
                    self._surface_by_node,
                    q_accumulate,
                    accumulate_coefficient,
                )
        else:
            self._compute_volume_rhs(RHS_Q_view)
            surface = self._surface_view
            RHS_P.add_(surface[:, 0, :])
            RHS_Vx.add_(surface[:, 1, :])
            RHS_Vy.add_(surface[:, 2, :])
            RHS_Vz.add_(surface[:, 3, :])
            if q_accumulate is not None:
                q_accumulate.add_(RHS_Q, alpha=accumulate_coefficient)

        return RHS_Q, BCvar

    def RHS_operator_packed_accumulate(
        self,
        q_by_node: torch.tensor,
        BCvar: list[dict],
        q_accumulate: torch.tensor,
        coefficient: float,
    ):
        return self.RHS_operator_packed(
            q_by_node, BCvar, q_accumulate, coefficient
        )

    def time_integration(self, **kwargs):
        """Perform time integration for the acoustics simulation. Additional keyword arguments are optional and can vary:

        Args:
            n_time_steps (int): number of time steps to be performed.
            total_time (float): total simulation time to be performed, determines the number of time steps given the current time step.
            delta_step (int): print solution every delta_step time steps.
            save_step (int): save solution every save_step time steps.
            format (str): the format of the file to save the results. Can be either 'mat' or 'npy'. The default format is 'mat'.
            progress (bool): print progress and timing information. The default is True.
            synchronize_timing (bool): synchronize CUDA before final timing. The default is False.
            use_cuda_graph (bool): replay a captured CUDA graph for each packed time step. The default is False.
            cuda_graph_chunk_steps (int): number of consecutive steps captured in one graph replay. The default is 1.
            record_receivers (bool): sample receiver pressure every time step. The default is True.

        Returns:
            prec (torch.tensor): Pressure field at the microphone locations.
        """

        # Process optional input arguments
        if "n_time_steps" in kwargs and "total_time" in kwargs:
            # Not possible to input both number of steps and total simulation time
            raise ValueError("Set only n_time_steps or total_time, do not set both...")

        elif "n_time_steps" in kwargs:
            # Directly use the number of timesteps
            self.Ntimesteps = kwargs["n_time_steps"]
            total_time = self.Ntimesteps * self.time_integrator.dt

        elif "total_time" in kwargs:
            # Compute the number of time steps from the total simulation time and the time step size of the time integrator
            total_time = kwargs["total_time"]
            self.Ntimesteps = math.floor(total_time / self.time_integrator.dt)

        else:
            raise ValueError("You need to set n_time_steps or total_time...")

        progress = kwargs.get("progress", True)
        synchronize_timing = kwargs.get("synchronize_timing", False)
        record_receivers = kwargs.get("record_receivers", True)
        cuda_graph_chunk_steps = max(1, int(kwargs.get("cuda_graph_chunk_steps", 1)))
        use_cuda_graph = (
            kwargs.get("use_cuda_graph", False)
            and torch.cuda.is_available()
            and hasattr(self, "Q_flat")
            and self.Q_flat.device.type == "cuda"
            and hasattr(self.time_integrator, "step_dt_packed")
        )
        if use_cuda_graph and cuda_graph_chunk_steps > 1 and (
            progress or "save_step" in kwargs
        ):
            cuda_graph_chunk_steps = 1
        if progress:
            print(f"Total simulation time is {total_time}")
            print(f"Total number of simulation steps is {self.Ntimesteps}")

        self.prec = torch.zeros(
            [self.rec.shape[1], self.Ntimesteps],
            device=device_ini.device,
            dtype=device_ini.dtype,
        )

        cuda_step_graph = None
        cuda_sample_chunk = None
        if use_cuda_graph:
            cuda_step_graph, cuda_sample_chunk = self._ensure_cuda_step_graph(
                cuda_graph_chunk_steps, record_receivers
            )
        startTime = time.time()
        curTime = startTime
        prevEstimated = 0
        # Step the solution
        StepIndex = 0
        while StepIndex < self.Ntimesteps:
            # for StepIndex in range(1):

            if (
                cuda_step_graph is not None
                and cuda_graph_chunk_steps > 1
                and StepIndex + cuda_graph_chunk_steps <= self.Ntimesteps
            ):
                cuda_step_graph.replay()
                if cuda_sample_chunk is not None:
                    self.prec[
                        :, StepIndex : StepIndex + cuda_graph_chunk_steps
                    ].copy_(cuda_sample_chunk.T)
                StepIndex += cuda_graph_chunk_steps
                continue

            receiver_sampled_by_graph = False
            if cuda_step_graph is not None:
                cuda_step_graph.replay()
                if cuda_sample_chunk is not None:
                    self.prec[:, StepIndex].copy_(cuda_sample_chunk[0])
                    receiver_sampled_by_graph = True
            elif hasattr(self, "Q_flat") and hasattr(
                self.time_integrator, "step_dt_packed"
            ):
                self.time_integrator.step_dt_packed(self.Q_flat, self.BC)
            else:
                self.time_integrator.step_dt(
                    self.P,
                    self.Vx,
                    self.Vy,
                    self.Vz,
                    self.BC,
                )  # by changing the value in place, the ID of the object is not changed (no new object is created), but the previous value is lost, which is not important here, because the previous value is not used anymore``
            if record_receivers and not receiver_sampled_by_graph:
                self._sample_receivers(self._sample_output)
                self.prec[:, StepIndex].copy_(self._sample_output)

            if (
                progress
                and "delta_step" in kwargs
                and StepIndex % kwargs["delta_step"] == 0
            ):
                newTime = time.time()
                elapsed = newTime - curTime
                if prevEstimated == 0:
                    estimated = (self.Ntimesteps - (StepIndex + 1)) * elapsed / 10
                else:
                    estimated = (
                        0.99 * prevEstimated
                        + 0.01 * (self.Ntimesteps - (StepIndex + 1)) * elapsed / 10
                    )

                minutes = math.floor(estimated / 60)
                seconds = math.floor(estimated - 60 * minutes)
                print(f"Estimated time left: {minutes} minutes {seconds} seconds")
                print(
                    f"Percentage done: {round(100*(StepIndex + 1)/self.Ntimesteps)} %"
                )
                curTime = newTime
                prevEstimated = estimated
                print(f"Current/Total step {StepIndex+1}/{self.Ntimesteps}")
                print(
                    f"Current/Total time {self.time_integrator.dt * StepIndex}/{total_time}"
                )
                print(f"P at mic locations {self.prec[:,StepIndex]}")

            if "save_step" in kwargs and StepIndex % kwargs["save_step"] == 0:
                self.save_results_on_the_run(format=kwargs.get("format", "mat"))
            StepIndex += 1
        if synchronize_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        if progress:
            print(f"time: {end_time-startTime} s")
        return self.prec

    def save_results_on_the_run(self, format: str = "mat"):
        """Save the temporary simulation results to a file.

        Args:
            format (str): the format of the file to save the results. Can be either 'mat' or 'npy'. The default format is 'mat'.
        """
        if format == "mat":
            scipy.io.savemat(
                "results_on_the_run.mat",
                {
                    "BCpara": self.BC.BCpara,
                    "prec": self.prec.cpu().numpy(),
                    "rec": self.rec,
                    "dt": self.time_integrator.dt,
                    "Ntimesteps": self.Ntimesteps,
                    "total_time": self.Ntimesteps * self.time_integrator.dt,
                    "Np": self.Np,
                    "N_tets": self.N_tets,
                    "rho0": self.rho0,
                    "c0": self.c0,
                    "mesh_filename": self.mesh.filename,
                    "source_xyz": self.IC.source_xyz,
                    "halfwidth": self.IC.halfwidth,
                    "Nx": self.Nx,
                    "Nt": self.time_integrator.Nt,
                    "CFL": self.time_integrator.CFL,
                },
            )
        elif format == "npy":
            numpy.savez(
                "results_on_the_run.npz",
                BCpara=self.BC.BCpara,
                prec=self.prec.cpu().numpy(),
                rec=self.rec,
                dt=self.time_integrator.dt,
                Ntimesteps=self.Ntimesteps,
                total_time=self.Ntimesteps * self.time_integrator.dt,
                Np=self.Np,
                N_tets=self.mesh.N_tets,
                rho0=self.rho0,
                c0=self.c0,
                mesh_filename=self.mesh.filename,
                source_xyz=self.IC.source_xyz,
                halfwidth=self.IC.halfwidth,
                Nx=self.Nx,
                Nt=self.time_integrator.Nt,
                CFL=self.time_integrator.CFL,
            )
        else:
            raise ValueError(
                "Invalid format, the format should be either 'mat' or 'npy'."
            )
