"""Large-mesh Triton helpers for the 2D extended-reaction RHS."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is optional at runtime
    triton = None
    tl = None


TRITON_AVAILABLE = triton is not None


if TRITON_AVAILABLE:

    @triton.jit
    def er_rhs_nonporous_post_2d_kernel(
        q_ptr,
        d_q_ptr,
        surface_ptr,
        metric_x_ptr,
        metric_y_ptr,
        metric_dx_ptr,
        metric_dy_ptr,
        k_inf_ptr,
        inv_rho_inf_ptr,
        sponge_sigma_ptr,
        rhs_ptr,
        q_accumulate_ptr,
        coefficient,
        n_elements: tl.constexpr,
        n_var_elements: tl.constexpr,
        n_active_cols: tl.constexpr,
        porous_start: tl.constexpr,
        n_porous: tl.constexpr,
        n_nonporous: tl.constexpr,
        NP: tl.constexpr,
        ACCUMULATE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        node = tl.program_id(1)
        mask = offsets < n_nonporous
        element = tl.where(offsets < porous_start, offsets, offsets + n_porous)

        q_base = node * n_var_elements + element
        dq_dr_base = node * n_active_cols + element
        dq_ds_base = (NP + node) * n_active_cols + element
        surface_base = node * n_active_cols + element
        metric_base = node * n_elements + element

        pressure = tl.load(q_ptr + q_base, mask=mask, other=0.0)
        velocity_x = tl.load(q_ptr + q_base + n_elements, mask=mask, other=0.0)
        velocity_y = tl.load(q_ptr + q_base + 2 * n_elements, mask=mask, other=0.0)

        d_p_dr = tl.load(d_q_ptr + dq_dr_base, mask=mask, other=0.0)
        d_p_ds = tl.load(d_q_ptr + dq_ds_base, mask=mask, other=0.0)
        d_vx_dr = tl.load(d_q_ptr + dq_dr_base + n_elements, mask=mask, other=0.0)
        d_vx_ds = tl.load(d_q_ptr + dq_ds_base + n_elements, mask=mask, other=0.0)
        d_vy_dr = tl.load(d_q_ptr + dq_dr_base + 2 * n_elements, mask=mask, other=0.0)
        d_vy_ds = tl.load(d_q_ptr + dq_ds_base + 2 * n_elements, mask=mask, other=0.0)

        metric_x = tl.load(metric_x_ptr + metric_base, mask=mask, other=0.0)
        metric_y = tl.load(metric_y_ptr + metric_base, mask=mask, other=0.0)
        metric_dx = tl.load(metric_dx_ptr + metric_base, mask=mask, other=0.0)
        metric_dy = tl.load(metric_dy_ptr + metric_base, mask=mask, other=0.0)

        d_p_dx = metric_x * d_p_dr + metric_y * d_p_ds
        d_p_dy = metric_dx * d_p_dr + metric_dy * d_p_ds
        d_vx_dx = metric_x * d_vx_dr + metric_y * d_vx_ds
        d_vy_dy = metric_dx * d_vy_dr + metric_dy * d_vy_ds
        div_v = d_vx_dx + d_vy_dy

        k_inf = tl.load(k_inf_ptr + element, mask=mask, other=0.0)
        inv_rho_inf = tl.load(inv_rho_inf_ptr + element, mask=mask, other=0.0)
        sponge_sigma = tl.load(sponge_sigma_ptr + element, mask=mask, other=0.0)

        rhs_p = -k_inf * div_v
        rhs_vx = -inv_rho_inf * d_p_dx
        rhs_vy = -inv_rho_inf * d_p_dy
        rhs_p += tl.load(surface_ptr + surface_base, mask=mask, other=0.0)
        rhs_vx += tl.load(
            surface_ptr + surface_base + n_elements, mask=mask, other=0.0
        )
        rhs_vy += tl.load(
            surface_ptr + surface_base + 2 * n_elements, mask=mask, other=0.0
        )
        rhs_p -= sponge_sigma * pressure
        rhs_vx -= sponge_sigma * velocity_x
        rhs_vy -= sponge_sigma * velocity_y

        tl.store(rhs_ptr + q_base, rhs_p, mask=mask)
        tl.store(rhs_ptr + q_base + n_elements, rhs_vx, mask=mask)
        tl.store(rhs_ptr + q_base + 2 * n_elements, rhs_vy, mask=mask)
        if not ACCUMULATE:
            tl.store(rhs_ptr + q_base + 3 * n_elements, 0.0, mask=mask)

        if ACCUMULATE:
            tl.store(
                q_accumulate_ptr + q_base,
                tl.load(q_accumulate_ptr + q_base, mask=mask, other=0.0)
                + coefficient * rhs_p,
                mask=mask,
            )
            tl.store(
                q_accumulate_ptr + q_base + n_elements,
                tl.load(
                    q_accumulate_ptr + q_base + n_elements, mask=mask, other=0.0
                )
                + coefficient * rhs_vx,
                mask=mask,
            )
            tl.store(
                q_accumulate_ptr + q_base + 2 * n_elements,
                tl.load(
                    q_accumulate_ptr + q_base + 2 * n_elements, mask=mask, other=0.0
                )
                + coefficient * rhs_vy,
                mask=mask,
            )


    @triton.jit
    def er_rhs_porous_post_2d_kernel(
        q_ptr,
        d_q_ptr,
        surface_ptr,
        metric_x_ptr,
        metric_y_ptr,
        metric_dx_ptr,
        metric_dy_ptr,
        sponge_sigma_ptr,
        beta_ca_ptr,
        rho_ca_ptr,
        beta_diag_ptr,
        beta_b_ptr,
        rho_diag_ptr,
        rho_b_ptr,
        z_beta_ptr,
        z_rho_x_ptr,
        z_rho_y_ptr,
        z_beta_work_ptr,
        z_rho_x_work_ptr,
        z_rho_y_work_ptr,
        rhs_ptr,
        q_accumulate_ptr,
        coefficient,
        neg_inv_beta_d,
        neg_beta_cb_over_d,
        neg_inv_rho_d,
        neg_rho_cb_over_d,
        n_elements: tl.constexpr,
        n_var_elements: tl.constexpr,
        n_active_cols: tl.constexpr,
        porous_start: tl.constexpr,
        n_porous: tl.constexpr,
        NP: tl.constexpr,
        NSTATES: tl.constexpr,
        ACCUMULATE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        porous_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        node = tl.program_id(1)
        mask = porous_offsets < n_porous
        element = porous_start + porous_offsets

        q_base = node * n_var_elements + element
        dq_dr_base = node * n_active_cols + element
        dq_ds_base = (NP + node) * n_active_cols + element
        surface_base = node * n_active_cols + element
        metric_base = node * n_elements + element

        pressure = tl.load(q_ptr + q_base, mask=mask, other=0.0)
        velocity_x = tl.load(q_ptr + q_base + n_elements, mask=mask, other=0.0)
        velocity_y = tl.load(q_ptr + q_base + 2 * n_elements, mask=mask, other=0.0)

        d_p_dr = tl.load(d_q_ptr + dq_dr_base, mask=mask, other=0.0)
        d_p_ds = tl.load(d_q_ptr + dq_ds_base, mask=mask, other=0.0)
        d_vx_dr = tl.load(d_q_ptr + dq_dr_base + n_elements, mask=mask, other=0.0)
        d_vx_ds = tl.load(d_q_ptr + dq_ds_base + n_elements, mask=mask, other=0.0)
        d_vy_dr = tl.load(d_q_ptr + dq_dr_base + 2 * n_elements, mask=mask, other=0.0)
        d_vy_ds = tl.load(d_q_ptr + dq_ds_base + 2 * n_elements, mask=mask, other=0.0)

        metric_x = tl.load(metric_x_ptr + metric_base, mask=mask, other=0.0)
        metric_y = tl.load(metric_y_ptr + metric_base, mask=mask, other=0.0)
        metric_dx = tl.load(metric_dx_ptr + metric_base, mask=mask, other=0.0)
        metric_dy = tl.load(metric_dy_ptr + metric_base, mask=mask, other=0.0)

        d_p_dx = metric_x * d_p_dr + metric_y * d_p_ds
        d_p_dy = metric_dx * d_p_dr + metric_dy * d_p_ds
        d_vx_dx = metric_x * d_vx_dr + metric_y * d_vx_ds
        d_vy_dy = metric_dx * d_vy_dr + metric_dy * d_vy_ds
        div_v = d_vx_dx + d_vy_dy

        beta_memory = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        rho_memory_x = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        rho_memory_y = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)

        for state_index in range(NSTATES):
            state_base = state_index * NP * n_porous + node * n_porous + porous_offsets

            z_beta_work = tl.load(z_beta_work_ptr + state_base, mask=mask, other=0.0)
            z_rho_x_work = tl.load(z_rho_x_work_ptr + state_base, mask=mask, other=0.0)
            z_rho_y_work = tl.load(z_rho_y_work_ptr + state_base, mask=mask, other=0.0)

            beta_ca = tl.load(beta_ca_ptr + state_index)
            rho_ca = tl.load(rho_ca_ptr + state_index)
            beta_diag = tl.load(beta_diag_ptr + state_index)
            beta_b = tl.load(beta_b_ptr + state_index)
            rho_diag = tl.load(rho_diag_ptr + state_index)
            rho_b = tl.load(rho_b_ptr + state_index)

            beta_memory += beta_ca * z_beta_work
            rho_memory_x += rho_ca * z_rho_x_work
            rho_memory_y += rho_ca * z_rho_y_work

            beta_rhs = beta_diag * z_beta_work + beta_b * pressure
            rho_x_rhs = rho_diag * z_rho_x_work + rho_b * velocity_x
            rho_y_rhs = rho_diag * z_rho_y_work + rho_b * velocity_y

            tl.store(z_beta_work_ptr + state_base, beta_rhs, mask=mask)
            tl.store(z_rho_x_work_ptr + state_base, rho_x_rhs, mask=mask)
            tl.store(z_rho_y_work_ptr + state_base, rho_y_rhs, mask=mask)

            tl.store(
                z_beta_ptr + state_base,
                tl.load(z_beta_ptr + state_base, mask=mask, other=0.0)
                + coefficient * beta_rhs,
                mask=mask,
            )
            tl.store(
                z_rho_x_ptr + state_base,
                tl.load(z_rho_x_ptr + state_base, mask=mask, other=0.0)
                + coefficient * rho_x_rhs,
                mask=mask,
            )
            tl.store(
                z_rho_y_ptr + state_base,
                tl.load(z_rho_y_ptr + state_base, mask=mask, other=0.0)
                + coefficient * rho_y_rhs,
                mask=mask,
            )

        sponge_sigma = tl.load(sponge_sigma_ptr + element, mask=mask, other=0.0)

        rhs_p = neg_inv_beta_d * (div_v + beta_memory) + neg_beta_cb_over_d * pressure
        rhs_vx = neg_inv_rho_d * (d_p_dx + rho_memory_x) + neg_rho_cb_over_d * velocity_x
        rhs_vy = neg_inv_rho_d * (d_p_dy + rho_memory_y) + neg_rho_cb_over_d * velocity_y

        rhs_p += tl.load(surface_ptr + surface_base, mask=mask, other=0.0)
        rhs_vx += tl.load(surface_ptr + surface_base + n_elements, mask=mask, other=0.0)
        rhs_vy += tl.load(
            surface_ptr + surface_base + 2 * n_elements, mask=mask, other=0.0
        )

        rhs_p -= sponge_sigma * pressure
        rhs_vx -= sponge_sigma * velocity_x
        rhs_vy -= sponge_sigma * velocity_y

        tl.store(rhs_ptr + q_base, rhs_p, mask=mask)
        tl.store(rhs_ptr + q_base + n_elements, rhs_vx, mask=mask)
        tl.store(rhs_ptr + q_base + 2 * n_elements, rhs_vy, mask=mask)
        if not ACCUMULATE:
            tl.store(rhs_ptr + q_base + 3 * n_elements, 0.0, mask=mask)

        if ACCUMULATE:
            tl.store(
                q_accumulate_ptr + q_base,
                tl.load(q_accumulate_ptr + q_base, mask=mask, other=0.0)
                + coefficient * rhs_p,
                mask=mask,
            )
            tl.store(
                q_accumulate_ptr + q_base + n_elements,
                tl.load(
                    q_accumulate_ptr + q_base + n_elements, mask=mask, other=0.0
                )
                + coefficient * rhs_vx,
                mask=mask,
            )
            tl.store(
                q_accumulate_ptr + q_base + 2 * n_elements,
                tl.load(
                    q_accumulate_ptr + q_base + 2 * n_elements, mask=mask, other=0.0
                )
                + coefficient * rhs_vy,
                mask=mask,
            )


def _require_triton():
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available in this environment.")


def _is_v100(tensor: torch.Tensor) -> bool:
    if tensor.device.type != "cuda":
        return False
    return (
        torch.cuda.get_device_capability(tensor.device) == (7, 0)
        and "V100" in torch.cuda.get_device_name(tensor.device)
    )


def launch_er_rhs_nonporous_post_2d(
    *,
    q_by_node: torch.Tensor,
    d_q_by_derivative: torch.Tensor,
    surface_by_node: torch.Tensor,
    metric_x: torch.Tensor,
    metric_y: torch.Tensor,
    metric_dx: torch.Tensor,
    metric_dy: torch.Tensor,
    k_inf: torch.Tensor,
    inv_rho_inf: torch.Tensor,
    sponge_sigma: torch.Tensor,
    porous_start: int,
    n_porous: int,
    rhs_by_node: torch.Tensor,
    q_accumulate: torch.Tensor | None,
    coefficient: float,
    block_size: int | None = None,
    num_warps: int | None = None,
):
    _require_triton()
    n_elements = int(q_by_node.shape[1] // 4)
    n_nonporous = n_elements - int(n_porous)
    if n_nonporous <= 0:
        return
    if block_size is None:
        block_size = 256 if _is_v100(q_by_node) else 128
    if num_warps is None:
        num_warps = 8 if block_size >= 256 else 4
    grid = (triton.cdiv(n_nonporous, block_size), int(q_by_node.shape[0]))
    er_rhs_nonporous_post_2d_kernel[grid](
        q_by_node,
        d_q_by_derivative,
        surface_by_node,
        metric_x,
        metric_y,
        metric_dx,
        metric_dy,
        k_inf,
        inv_rho_inf,
        sponge_sigma,
        rhs_by_node,
        rhs_by_node if q_accumulate is None else q_accumulate,
        coefficient,
        n_elements,
        4 * n_elements,
        int(d_q_by_derivative.shape[1]),
        porous_start,
        int(n_porous),
        n_nonporous,
        NP=int(q_by_node.shape[0]),
        ACCUMULATE=q_accumulate is not None,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def launch_er_rhs_porous_post_2d(
    *,
    q_by_node: torch.Tensor,
    d_q_by_derivative: torch.Tensor,
    surface_by_node: torch.Tensor,
    metric_x: torch.Tensor,
    metric_y: torch.Tensor,
    metric_dx: torch.Tensor,
    metric_dy: torch.Tensor,
    sponge_sigma: torch.Tensor,
    beta_ca: torch.Tensor,
    rho_ca: torch.Tensor,
    beta_diag: torch.Tensor,
    beta_b: torch.Tensor,
    rho_diag: torch.Tensor,
    rho_b: torch.Tensor,
    z_beta: torch.Tensor,
    z_rho_x: torch.Tensor,
    z_rho_y: torch.Tensor,
    z_beta_work: torch.Tensor,
    z_rho_x_work: torch.Tensor,
    z_rho_y_work: torch.Tensor,
    rhs_by_node: torch.Tensor,
    q_accumulate: torch.Tensor | None,
    coefficient: float,
    beta_cb: float,
    rho_cb: float,
    beta_d: float,
    rho_d: float,
    porous_start: int,
    n_porous: int,
    block_size: int | None = None,
    num_warps: int | None = None,
):
    _require_triton()
    if int(n_porous) <= 0:
        return
    n_elements = int(q_by_node.shape[1] // 4)
    v100_er_shape = (
        _is_v100(q_by_node)
        and q_by_node.shape[0] == 15
        and int(beta_ca.numel()) == 8
    )
    if block_size is None:
        block_size = 256 if v100_er_shape else 128
    if num_warps is None:
        num_warps = 8 if block_size >= 256 else 4
    grid = (triton.cdiv(int(n_porous), block_size), int(q_by_node.shape[0]))
    er_rhs_porous_post_2d_kernel[grid](
        q_by_node,
        d_q_by_derivative,
        surface_by_node,
        metric_x,
        metric_y,
        metric_dx,
        metric_dy,
        sponge_sigma,
        beta_ca,
        rho_ca,
        beta_diag,
        beta_b,
        rho_diag,
        rho_b,
        z_beta,
        z_rho_x,
        z_rho_y,
        z_beta_work,
        z_rho_x_work,
        z_rho_y_work,
        rhs_by_node,
        rhs_by_node if q_accumulate is None else q_accumulate,
        coefficient,
        -(1.0 / beta_d),
        -(beta_cb / beta_d),
        -(1.0 / rho_d),
        -(rho_cb / rho_d),
        n_elements,
        4 * n_elements,
        int(d_q_by_derivative.shape[1]),
        porous_start,
        int(n_porous),
        NP=int(q_by_node.shape[0]),
        NSTATES=int(beta_ca.numel()),
        ACCUMULATE=q_accumulate is not None,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
