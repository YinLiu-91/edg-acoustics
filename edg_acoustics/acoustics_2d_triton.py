"""Triton kernels and launch helpers for the 2D acoustic solvers."""

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
    def interior_material_flux_2d_kernel(
        q_ptr,
        face_node_ids_ptr,
        vmapP_q_ptr,
        nx_ptr,
        ny_ptr,
        rho_left_ptr,
        rho_right_ptr,
        c_left_ptr,
        c_right_ptr,
        fscale_ptr,
        flux_ptr,
        total_faces: tl.constexpr,
        n_elements: tl.constexpr,
        n_var_elements: tl.constexpr,
        SCALE_FLUX: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_faces
        element = offsets % n_elements
        face_node = offsets // n_elements

        node_m = tl.load(face_node_ids_ptr + face_node, mask=mask, other=0)
        base_m = node_m * n_var_elements + element
        base_p = tl.load(vmapP_q_ptr + offsets, mask=mask, other=0)

        p_m = tl.load(q_ptr + base_m, mask=mask, other=0.0)
        p_p = tl.load(q_ptr + base_p, mask=mask, other=0.0)
        vx_m = tl.load(q_ptr + base_m + n_elements, mask=mask, other=0.0)
        vx_p = tl.load(q_ptr + base_p + n_elements, mask=mask, other=0.0)
        vy_m = tl.load(q_ptr + base_m + 2 * n_elements, mask=mask, other=0.0)
        vy_p = tl.load(q_ptr + base_p + 2 * n_elements, mask=mask, other=0.0)

        nx = tl.load(nx_ptr + offsets, mask=mask, other=0.0)
        ny = tl.load(ny_ptr + offsets, mask=mask, other=0.0)
        rho_left = tl.load(rho_left_ptr + offsets, mask=mask, other=1.0)
        rho_right = tl.load(rho_right_ptr + offsets, mask=mask, other=1.0)
        c_left = tl.load(c_left_ptr + offsets, mask=mask, other=0.0)
        c_right = tl.load(c_right_ptr + offsets, mask=mask, other=0.0)

        dp = p_m - p_p
        dvn = nx * (vx_m - vx_p) + ny * (vy_m - vy_p)
        z_right = rho_right * c_right
        k_left = rho_left * c_left * c_left
        denominator = rho_left * c_left + z_right
        velocity_flux = (c_right * dp - c_left * z_right * dvn) / denominator
        pressure_flux = k_left * (z_right * dvn - dp) / denominator

        if SCALE_FLUX:
            fscale = tl.load(fscale_ptr + offsets, mask=mask, other=0.0)
            velocity_flux *= fscale
            pressure_flux *= fscale

        out_base = face_node * n_var_elements + element
        tl.store(flux_ptr + out_base, pressure_flux, mask=mask)
        tl.store(flux_ptr + out_base + n_elements, nx * velocity_flux, mask=mask)
        tl.store(flux_ptr + out_base + 2 * n_elements, ny * velocity_flux, mask=mask)
        tl.store(flux_ptr + out_base + 3 * n_elements, 0.0, mask=mask)


    @triton.jit
    def boundary_ri_flux_2d_kernel(
        q_ptr,
        vmap_q_ptr,
        flux_map_q_ptr,
        nx_ptr,
        ny_ptr,
        rho_ptr,
        c_ptr,
        z_ptr,
        k_ptr,
        fscale_ptr,
        flux_ptr,
        vn_ptr,
        ou_ptr,
        in_ptr,
        n_boundary: tl.constexpr,
        ri_ptr,
        SCALE_FLUX: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_boundary
        ri = tl.load(ri_ptr)

        idx_p = tl.load(vmap_q_ptr + offsets, mask=mask, other=0)
        idx_vx = tl.load(vmap_q_ptr + n_boundary + offsets, mask=mask, other=0)
        idx_vy = tl.load(vmap_q_ptr + 2 * n_boundary + offsets, mask=mask, other=0)
        p = tl.load(q_ptr + idx_p, mask=mask, other=0.0)
        vx = tl.load(q_ptr + idx_vx, mask=mask, other=0.0)
        vy = tl.load(q_ptr + idx_vy, mask=mask, other=0.0)

        nx = tl.load(nx_ptr + offsets, mask=mask, other=0.0)
        ny = tl.load(ny_ptr + offsets, mask=mask, other=0.0)
        rho = tl.load(rho_ptr + offsets, mask=mask, other=1.0)
        c = tl.load(c_ptr + offsets, mask=mask, other=0.0)
        z = tl.load(z_ptr + offsets, mask=mask, other=1.0)
        k = tl.load(k_ptr + offsets, mask=mask, other=0.0)

        vn = nx * vx + ny * vy
        ou = vn + p / z
        incoming = ri * ou
        velocity_flux = p / rho - 0.5 * c * (ou + incoming)
        pressure_flux = (vn - 0.5 * ou + 0.5 * incoming) * k

        if SCALE_FLUX:
            fscale = tl.load(fscale_ptr + offsets, mask=mask, other=0.0)
            velocity_flux *= fscale
            pressure_flux *= fscale

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
        tl.store(flux_ptr + out_vz, 0.0, mask=mask)


    @triton.jit
    def combined_boundary_ri_flux_2d_kernel(
        q_ptr,
        vmap_q_ptr,
        flux_map_q_ptr,
        nx_ptr,
        ny_ptr,
        rho_ptr,
        c_ptr,
        z_ptr,
        k_ptr,
        fscale_ptr,
        ri_ptr,
        flux_ptr,
        vn_ptr,
        ou_ptr,
        in_ptr,
        n_boundary: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_boundary
        idx_p = tl.load(vmap_q_ptr + offsets, mask=mask, other=0)
        idx_vx = tl.load(
            vmap_q_ptr + n_boundary + offsets, mask=mask, other=0
        )
        idx_vy = tl.load(
            vmap_q_ptr + 2 * n_boundary + offsets, mask=mask, other=0
        )
        p = tl.load(q_ptr + idx_p, mask=mask, other=0.0)
        vx = tl.load(q_ptr + idx_vx, mask=mask, other=0.0)
        vy = tl.load(q_ptr + idx_vy, mask=mask, other=0.0)
        nx = tl.load(nx_ptr + offsets, mask=mask, other=0.0)
        ny = tl.load(ny_ptr + offsets, mask=mask, other=0.0)
        rho = tl.load(rho_ptr + offsets, mask=mask, other=1.0)
        c = tl.load(c_ptr + offsets, mask=mask, other=0.0)
        z = tl.load(z_ptr + offsets, mask=mask, other=1.0)
        k = tl.load(k_ptr + offsets, mask=mask, other=0.0)
        ri = tl.load(ri_ptr + offsets, mask=mask, other=0.0)

        vn = nx * vx + ny * vy
        ou = vn + p / z
        incoming = ri * ou
        velocity_flux = p / rho - 0.5 * c * (ou + incoming)
        pressure_flux = (vn - 0.5 * ou + 0.5 * incoming) * k
        fscale = tl.load(fscale_ptr + offsets, mask=mask, other=0.0)
        velocity_flux *= fscale
        pressure_flux *= fscale

        tl.store(vn_ptr + offsets, vn, mask=mask)
        tl.store(ou_ptr + offsets, ou, mask=mask)
        tl.store(in_ptr + offsets, incoming, mask=mask)
        out_p = tl.load(flux_map_q_ptr + offsets, mask=mask, other=0)
        out_vx = tl.load(
            flux_map_q_ptr + n_boundary + offsets, mask=mask, other=0
        )
        out_vy = tl.load(
            flux_map_q_ptr + 2 * n_boundary + offsets, mask=mask, other=0
        )
        tl.store(flux_ptr + out_p, pressure_flux, mask=mask)
        tl.store(flux_ptr + out_vx, nx * velocity_flux, mask=mask)
        tl.store(flux_ptr + out_vy, ny * velocity_flux, mask=mask)


    @triton.jit
    def fused_acoustic_rhs_2d_kernel(
        q_ptr,
        flux_ptr,
        dr_ptr,
        ds_ptr,
        lift_ptr,
        metric_x_ptr,
        metric_y_ptr,
        metric_dx_ptr,
        metric_dy_ptr,
        rhs_ptr,
        q_accumulate_ptr,
        coefficient,
        n_elements: tl.constexpr,
        n_var_elements: tl.constexpr,
        neg_k,
        neg_inv_rho,
        NP: tl.constexpr,
        NFACE: tl.constexpr,
        ACCUMULATE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        element_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        node = tl.program_id(1)
        mask = element_offsets < n_elements
        node_base = node * n_var_elements + element_offsets

        d_p_dr = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        d_p_ds = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        d_vx_dr = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        d_vx_ds = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        d_vy_dr = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        d_vy_ds = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        surface_p = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        surface_vx = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        surface_vy = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)

        for local_node in range(NP):
            dr = tl.load(dr_ptr + node * NP + local_node)
            ds = tl.load(ds_ptr + node * NP + local_node)
            q_base = local_node * n_var_elements + element_offsets
            pressure = tl.load(q_ptr + q_base, mask=mask, other=0.0)
            velocity_x = tl.load(
                q_ptr + q_base + n_elements, mask=mask, other=0.0
            )
            velocity_y = tl.load(
                q_ptr + q_base + 2 * n_elements, mask=mask, other=0.0
            )
            d_p_dr += dr * pressure
            d_p_ds += ds * pressure
            d_vx_dr += dr * velocity_x
            d_vx_ds += ds * velocity_x
            d_vy_dr += dr * velocity_y
            d_vy_ds += ds * velocity_y

        for face_node in range(NFACE):
            lift = tl.load(lift_ptr + node * NFACE + face_node)
            flux_base = face_node * n_var_elements + element_offsets
            surface_p += lift * tl.load(flux_ptr + flux_base, mask=mask, other=0.0)
            surface_vx += lift * tl.load(
                flux_ptr + flux_base + n_elements, mask=mask, other=0.0
            )
            surface_vy += lift * tl.load(
                flux_ptr + flux_base + 2 * n_elements, mask=mask, other=0.0
            )

        metric_index = node * n_elements + element_offsets
        metric_x = tl.load(metric_x_ptr + metric_index, mask=mask, other=0.0)
        metric_y = tl.load(metric_y_ptr + metric_index, mask=mask, other=0.0)
        metric_dx = tl.load(metric_dx_ptr + metric_index, mask=mask, other=0.0)
        metric_dy = tl.load(metric_dy_ptr + metric_index, mask=mask, other=0.0)

        d_p_dx = metric_x * d_p_dr + metric_y * d_p_ds
        d_p_dy = metric_dx * d_p_dr + metric_dy * d_p_ds
        d_vx_dx = metric_x * d_vx_dr + metric_y * d_vx_ds
        d_vy_dy = metric_dx * d_vy_dr + metric_dy * d_vy_ds
        div_v = d_vx_dx + d_vy_dy

        rhs_p = neg_k * div_v + surface_p
        rhs_vx = neg_inv_rho * d_p_dx + surface_vx
        rhs_vy = neg_inv_rho * d_p_dy + surface_vy
        tl.store(rhs_ptr + node_base, rhs_p, mask=mask)
        tl.store(rhs_ptr + node_base + n_elements, rhs_vx, mask=mask)
        tl.store(rhs_ptr + node_base + 2 * n_elements, rhs_vy, mask=mask)
        if not ACCUMULATE:
            tl.store(rhs_ptr + node_base + 3 * n_elements, 0.0, mask=mask)

        if ACCUMULATE:
            tl.store(
                q_accumulate_ptr + node_base,
                tl.load(q_accumulate_ptr + node_base, mask=mask, other=0.0)
                + coefficient * rhs_p,
                mask=mask,
            )
            tl.store(
                q_accumulate_ptr + node_base + n_elements,
                tl.load(
                    q_accumulate_ptr + node_base + n_elements, mask=mask, other=0.0
                )
                + coefficient * rhs_vx,
                mask=mask,
            )
            tl.store(
                q_accumulate_ptr + node_base + 2 * n_elements,
                tl.load(
                    q_accumulate_ptr + node_base + 2 * n_elements, mask=mask, other=0.0
                )
                + coefficient * rhs_vy,
                mask=mask,
            )


    @triton.jit
    def fused_er_rhs_2d_kernel(
        q_ptr,
        flux_ptr,
        dr_ptr,
        ds_ptr,
        lift_ptr,
        metric_x_ptr,
        metric_y_ptr,
        metric_dx_ptr,
        metric_dy_ptr,
        porous_mask_ptr,
        k_inf_ptr,
        inv_rho_inf_ptr,
        sponge_sigma_ptr,
        beta_ca_ptr,
        rho_ca_ptr,
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
        porous_start: tl.constexpr,
        n_porous: tl.constexpr,
        NP: tl.constexpr,
        NFACE: tl.constexpr,
        NSTATES: tl.constexpr,
        ACCUMULATE: tl.constexpr,
        COMPACT_ADE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        element_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        node = tl.program_id(1)
        mask = element_offsets < n_elements
        node_base = node * n_var_elements + element_offsets

        d_p_dr = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        d_p_ds = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        d_vx_dr = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        d_vx_ds = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        d_vy_dr = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        d_vy_ds = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        surface_p = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        surface_vx = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        surface_vy = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)

        for local_node in range(NP):
            dr = tl.load(dr_ptr + node * NP + local_node)
            ds = tl.load(ds_ptr + node * NP + local_node)
            q_base = local_node * n_var_elements + element_offsets
            pressure_local = tl.load(q_ptr + q_base, mask=mask, other=0.0)
            velocity_x_local = tl.load(
                q_ptr + q_base + n_elements, mask=mask, other=0.0
            )
            velocity_y_local = tl.load(
                q_ptr + q_base + 2 * n_elements, mask=mask, other=0.0
            )
            d_p_dr += dr * pressure_local
            d_p_ds += ds * pressure_local
            d_vx_dr += dr * velocity_x_local
            d_vx_ds += ds * velocity_x_local
            d_vy_dr += dr * velocity_y_local
            d_vy_ds += ds * velocity_y_local

        for face_node in range(NFACE):
            lift = tl.load(lift_ptr + node * NFACE + face_node)
            flux_base = face_node * n_var_elements + element_offsets
            surface_p += lift * tl.load(flux_ptr + flux_base, mask=mask, other=0.0)
            surface_vx += lift * tl.load(
                flux_ptr + flux_base + n_elements, mask=mask, other=0.0
            )
            surface_vy += lift * tl.load(
                flux_ptr + flux_base + 2 * n_elements, mask=mask, other=0.0
            )

        metric_index = node * n_elements + element_offsets
        metric_x = tl.load(metric_x_ptr + metric_index, mask=mask, other=0.0)
        metric_y = tl.load(metric_y_ptr + metric_index, mask=mask, other=0.0)
        metric_dx = tl.load(metric_dx_ptr + metric_index, mask=mask, other=0.0)
        metric_dy = tl.load(metric_dy_ptr + metric_index, mask=mask, other=0.0)

        d_p_dx = metric_x * d_p_dr + metric_y * d_p_ds
        d_p_dy = metric_dx * d_p_dr + metric_dy * d_p_ds
        d_vx_dx = metric_x * d_vx_dr + metric_y * d_vx_ds
        d_vy_dy = metric_dx * d_vy_dr + metric_dy * d_vy_ds
        div_v = d_vx_dx + d_vy_dy

        pressure = tl.load(q_ptr + node_base, mask=mask, other=0.0)
        velocity_x = tl.load(q_ptr + node_base + n_elements, mask=mask, other=0.0)
        velocity_y = tl.load(
            q_ptr + node_base + 2 * n_elements, mask=mask, other=0.0
        )
        porous = tl.load(porous_mask_ptr + element_offsets, mask=mask, other=0) != 0
        porous_mask = mask & porous

        k_inf = tl.load(k_inf_ptr + element_offsets, mask=mask, other=0.0)
        inv_rho_inf = tl.load(
            inv_rho_inf_ptr + element_offsets, mask=mask, other=0.0
        )
        sponge_sigma = tl.load(
            sponge_sigma_ptr + element_offsets, mask=mask, other=0.0
        )
        beta_memory = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        rho_memory_x = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        rho_memory_y = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        for state_index in range(NSTATES):
            if COMPACT_ADE:
                state_offset = (
                    state_index * NP * n_porous
                    + node * n_porous
                    + element_offsets
                    - porous_start
                )
            else:
                state_offset = (
                    state_index * NP * n_elements
                    + node * n_elements
                    + element_offsets
                )
            beta_ca = tl.load(beta_ca_ptr + state_index)
            rho_ca = tl.load(rho_ca_ptr + state_index)
            beta_memory += beta_ca * tl.load(
                z_beta_work_ptr + state_offset, mask=porous_mask, other=0.0
            )
            rho_memory_x += rho_ca * tl.load(
                z_rho_x_work_ptr + state_offset, mask=porous_mask, other=0.0
            )
            rho_memory_y += rho_ca * tl.load(
                z_rho_y_work_ptr + state_offset, mask=porous_mask, other=0.0
            )

        rhs_p_air = -k_inf * div_v
        rhs_vx_air = -inv_rho_inf * d_p_dx
        rhs_vy_air = -inv_rho_inf * d_p_dy

        rhs_p_porous = (
            neg_inv_beta_d * (div_v + beta_memory)
            + neg_beta_cb_over_d * pressure
        )
        rhs_vx_porous = (
            neg_inv_rho_d * (d_p_dx + rho_memory_x)
            + neg_rho_cb_over_d * velocity_x
        )
        rhs_vy_porous = (
            neg_inv_rho_d * (d_p_dy + rho_memory_y)
            + neg_rho_cb_over_d * velocity_y
        )

        rhs_p = tl.where(porous, rhs_p_porous, rhs_p_air) - sponge_sigma * pressure
        rhs_vx = tl.where(porous, rhs_vx_porous, rhs_vx_air) - sponge_sigma * velocity_x
        rhs_vy = tl.where(porous, rhs_vy_porous, rhs_vy_air) - sponge_sigma * velocity_y
        rhs_p += surface_p
        rhs_vx += surface_vx
        rhs_vy += surface_vy

        tl.store(rhs_ptr + node_base, rhs_p, mask=mask)
        tl.store(rhs_ptr + node_base + n_elements, rhs_vx, mask=mask)
        tl.store(rhs_ptr + node_base + 2 * n_elements, rhs_vy, mask=mask)
        if not ACCUMULATE:
            tl.store(rhs_ptr + node_base + 3 * n_elements, 0.0, mask=mask)

        if ACCUMULATE:
            tl.store(
                q_accumulate_ptr + node_base,
                tl.load(q_accumulate_ptr + node_base, mask=mask, other=0.0)
                + coefficient * rhs_p,
                mask=mask,
            )
            tl.store(
                q_accumulate_ptr + node_base + n_elements,
                tl.load(
                    q_accumulate_ptr + node_base + n_elements, mask=mask, other=0.0
                )
                + coefficient * rhs_vx,
                mask=mask,
            )
            tl.store(
                q_accumulate_ptr + node_base + 2 * n_elements,
                tl.load(
                    q_accumulate_ptr + node_base + 2 * n_elements, mask=mask, other=0.0
                )
                + coefficient * rhs_vy,
                mask=mask,
            )


    @triton.jit
    def sample_receivers_2d_kernel(
        pressure_ptr,
        element_ids_ptr,
        weights_ptr,
        output_ptr,
        pressure_node_stride: tl.constexpr,
        pressure_element_stride: tl.constexpr,
        NP: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        receiver = tl.program_id(0)
        nodes = tl.arange(0, BLOCK_SIZE)
        mask = nodes < NP
        element = tl.load(element_ids_ptr + receiver)
        pressure = tl.load(
            pressure_ptr
            + nodes * pressure_node_stride
            + element * pressure_element_stride,
            mask=mask,
            other=0.0,
        )
        weights = tl.load(
            weights_ptr + receiver * NP + nodes,
            mask=mask,
            other=0.0,
        )
        tl.store(output_ptr + receiver, tl.sum(pressure * weights, axis=0))


    @triton.jit
    def fused_er_aux_update_diag_kernel(
        q_ptr,
        porous_element_ids_ptr,
        z_beta_ptr,
        z_rho_x_ptr,
        z_rho_y_ptr,
        z_beta_work_ptr,
        z_rho_x_work_ptr,
        z_rho_y_work_ptr,
        beta_diag_ptr,
        beta_b_ptr,
        rho_diag_ptr,
        rho_b_ptr,
        coefficient,
        n_elements: tl.constexpr,
        n_var_elements: tl.constexpr,
        n_porous: tl.constexpr,
        NP: tl.constexpr,
        NSTATES: tl.constexpr,
        COMPACT_ADE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        porous_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        node = tl.program_id(1)
        state_index = tl.program_id(2)
        mask = porous_offsets < n_porous
        element_offsets = tl.load(
            porous_element_ids_ptr + porous_offsets, mask=mask, other=0
        )

        q_node_base = node * n_var_elements + element_offsets
        pressure = tl.load(q_ptr + q_node_base, mask=mask, other=0.0)
        velocity_x = tl.load(q_ptr + q_node_base + n_elements, mask=mask, other=0.0)
        velocity_y = tl.load(
            q_ptr + q_node_base + 2 * n_elements, mask=mask, other=0.0
        )

        beta_diag = tl.load(beta_diag_ptr + state_index)
        beta_b = tl.load(beta_b_ptr + state_index)
        rho_diag = tl.load(rho_diag_ptr + state_index)
        rho_b = tl.load(rho_b_ptr + state_index)

        if COMPACT_ADE:
            state_offset = (
                state_index * NP * n_porous
                + node * n_porous
                + porous_offsets
            )
        else:
            state_offset = (
                state_index * NP * n_elements
                + node * n_elements
                + element_offsets
            )

        z_beta_work = tl.load(z_beta_work_ptr + state_offset, mask=mask, other=0.0)
        z_rho_x_work = tl.load(z_rho_x_work_ptr + state_offset, mask=mask, other=0.0)
        z_rho_y_work = tl.load(z_rho_y_work_ptr + state_offset, mask=mask, other=0.0)

        beta_rhs = beta_diag * z_beta_work + beta_b * pressure
        rho_x_rhs = rho_diag * z_rho_x_work + rho_b * velocity_x
        rho_y_rhs = rho_diag * z_rho_y_work + rho_b * velocity_y

        tl.store(z_beta_work_ptr + state_offset, beta_rhs, mask=mask)
        tl.store(z_rho_x_work_ptr + state_offset, rho_x_rhs, mask=mask)
        tl.store(z_rho_y_work_ptr + state_offset, rho_y_rhs, mask=mask)

        tl.store(
            z_beta_ptr + state_offset,
            tl.load(z_beta_ptr + state_offset, mask=mask, other=0.0)
            + coefficient * beta_rhs,
            mask=mask,
        )
        tl.store(
            z_rho_x_ptr + state_offset,
            tl.load(z_rho_x_ptr + state_offset, mask=mask, other=0.0)
            + coefficient * rho_x_rhs,
            mask=mask,
        )
        tl.store(
            z_rho_y_ptr + state_offset,
            tl.load(z_rho_y_ptr + state_offset, mask=mask, other=0.0)
            + coefficient * rho_y_rhs,
            mask=mask,
        )


    @triton.jit
    def pml_auxiliary_rhs_2d_kernel(
        q_ptr,
        rhs_ptr,
        q_accumulate_ptr,
        pml_sigma_ptr,
        pml_psi_ptr,
        pml_psi_work_ptr,
        coefficient,
        inv_rho_c2,
        rho_c2,
        n_elements: tl.constexpr,
        n_var_elements: tl.constexpr,
        NP: tl.constexpr,
        ACCUMULATE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        element_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        node = tl.program_id(1)
        mask = element_offsets < n_elements
        node_base = node * n_var_elements + element_offsets

        pressure = tl.load(q_ptr + node_base, mask=mask, other=0.0)
        rhs_p_base = tl.load(rhs_ptr + node_base, mask=mask, other=0.0)
        rhs_vx_base = tl.load(
            rhs_ptr + node_base + n_elements, mask=mask, other=0.0
        )
        rhs_vy_base = tl.load(
            rhs_ptr + node_base + 2 * n_elements, mask=mask, other=0.0
        )

        sigma_offset = node * n_elements + element_offsets
        sigma_x = tl.load(pml_sigma_ptr + sigma_offset, mask=mask, other=0.0)
        sigma_y = tl.load(
            pml_sigma_ptr + NP * n_elements + sigma_offset, mask=mask, other=0.0
        )

        psi_offset = node * 2 * n_elements + element_offsets
        psi_x = tl.load(pml_psi_work_ptr + psi_offset, mask=mask, other=0.0)
        psi_y = tl.load(
            pml_psi_work_ptr + psi_offset + n_elements, mask=mask, other=0.0
        )

        correction_p = -(sigma_x + sigma_y) * pressure
        correction_vx = -inv_rho_c2 * psi_x
        correction_vy = -inv_rho_c2 * psi_y
        rhs_p = rhs_p_base + correction_p
        rhs_vx = rhs_vx_base + correction_vx
        rhs_vy = rhs_vy_base + correction_vy
        psi_rhs_x = rho_c2 * (sigma_x - sigma_y) * rhs_vx - sigma_y * psi_x
        psi_rhs_y = rho_c2 * (sigma_y - sigma_x) * rhs_vy - sigma_x * psi_y

        tl.store(rhs_ptr + node_base, rhs_p, mask=mask)
        tl.store(rhs_ptr + node_base + n_elements, rhs_vx, mask=mask)
        tl.store(rhs_ptr + node_base + 2 * n_elements, rhs_vy, mask=mask)
        tl.store(pml_psi_work_ptr + psi_offset, psi_rhs_x, mask=mask)
        tl.store(pml_psi_work_ptr + psi_offset + n_elements, psi_rhs_y, mask=mask)

        if ACCUMULATE:
            tl.store(
                q_accumulate_ptr + node_base,
                tl.load(q_accumulate_ptr + node_base, mask=mask, other=0.0)
                + coefficient * correction_p,
                mask=mask,
            )
            tl.store(
                q_accumulate_ptr + node_base + n_elements,
                tl.load(
                    q_accumulate_ptr + node_base + n_elements, mask=mask, other=0.0
                )
                + coefficient * correction_vx,
                mask=mask,
            )
            tl.store(
                q_accumulate_ptr + node_base + 2 * n_elements,
                tl.load(
                    q_accumulate_ptr + node_base + 2 * n_elements, mask=mask, other=0.0
                )
                + coefficient * correction_vy,
                mask=mask,
            )
            tl.store(
                pml_psi_ptr + psi_offset,
                tl.load(pml_psi_ptr + psi_offset, mask=mask, other=0.0)
                + coefficient * psi_rhs_x,
                mask=mask,
            )
            tl.store(
                pml_psi_ptr + psi_offset + n_elements,
                tl.load(
                    pml_psi_ptr + psi_offset + n_elements, mask=mask, other=0.0
                )
                + coefficient * psi_rhs_y,
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


def launch_interior_material_flux_2d(
    *,
    q_by_node: torch.Tensor,
    face_node_ids: torch.Tensor,
    vmap_p_q: torch.Tensor,
    nx: torch.Tensor,
    ny: torch.Tensor,
    rho_left: torch.Tensor,
    rho_right: torch.Tensor,
    c_left: torch.Tensor,
    c_right: torch.Tensor,
    fscale: torch.Tensor,
    flux_by_face: torch.Tensor,
    n_elements: int,
    scale_flux: bool,
    block_size: int | None = None,
):
    _require_triton()
    if block_size is None:
        block_size = 128 if _is_v100(q_by_node) else 256
    total_faces = int(face_node_ids.numel() * n_elements)
    interior_material_flux_2d_kernel[(triton.cdiv(total_faces, block_size),)](
        q_by_node.reshape(-1),
        face_node_ids,
        vmap_p_q,
        nx,
        ny,
        rho_left,
        rho_right,
        c_left,
        c_right,
        fscale.reshape(-1),
        flux_by_face.reshape(-1),
        total_faces,
        n_elements,
        4 * n_elements,
        scale_flux,
        BLOCK_SIZE=block_size,
    )


def launch_boundary_ri_flux_2d(
    *,
    q_flat: torch.Tensor,
    vmap_q: torch.Tensor,
    flux_map_q: torch.Tensor,
    nx: torch.Tensor,
    ny: torch.Tensor,
    rho: torch.Tensor,
    c: torch.Tensor,
    z: torch.Tensor,
    k: torch.Tensor,
    fscale: torch.Tensor,
    flux_flat: torch.Tensor,
    vn: torch.Tensor,
    ou: torch.Tensor,
    incoming: torch.Tensor,
    ri_tensor: torch.Tensor,
    scale_flux: bool,
    block_size: int = 256,
):
    _require_triton()
    n_boundary = int(vn.numel())
    boundary_ri_flux_2d_kernel[(triton.cdiv(n_boundary, block_size),)](
        q_flat,
        vmap_q,
        flux_map_q,
        nx,
        ny,
        rho,
        c,
        z,
        k,
        fscale,
        flux_flat,
        vn,
        ou,
        incoming,
        n_boundary,
        ri_tensor,
        scale_flux,
        BLOCK_SIZE=block_size,
    )


def launch_combined_boundary_ri_flux_2d(
    *,
    q_flat: torch.Tensor,
    vmap_q: torch.Tensor,
    flux_map_q: torch.Tensor,
    nx: torch.Tensor,
    ny: torch.Tensor,
    rho: torch.Tensor,
    c: torch.Tensor,
    z: torch.Tensor,
    k: torch.Tensor,
    fscale: torch.Tensor,
    ri: torch.Tensor,
    flux_flat: torch.Tensor,
    vn: torch.Tensor,
    ou: torch.Tensor,
    incoming: torch.Tensor,
    block_size: int = 128,
):
    _require_triton()
    n_boundary = int(vn.numel())
    combined_boundary_ri_flux_2d_kernel[
        (triton.cdiv(n_boundary, block_size),)
    ](
        q_flat,
        vmap_q,
        flux_map_q,
        nx,
        ny,
        rho,
        c,
        z,
        k,
        fscale,
        ri,
        flux_flat,
        vn,
        ou,
        incoming,
        n_boundary,
        BLOCK_SIZE=block_size,
    )


def launch_fused_acoustic_rhs_2d(
    *,
    q_by_node: torch.Tensor,
    flux_by_face: torch.Tensor,
    dr: torch.Tensor,
    ds: torch.Tensor,
    lift: torch.Tensor,
    metric_x: torch.Tensor,
    metric_y: torch.Tensor,
    metric_dx: torch.Tensor,
    metric_dy: torch.Tensor,
    rhs_by_node: torch.Tensor,
    q_accumulate: torch.Tensor | None,
    coefficient: float,
    c0: float,
    rho0: float,
    block_size: int | None = None,
    num_warps: int | None = None,
):
    _require_triton()
    n_elements = int(q_by_node.shape[1] // 4)
    if block_size is None:
        block_size = 64 if _is_v100(q_by_node) and q_by_node.shape[0] == 15 else 128
    if num_warps is None:
        num_warps = 2 if block_size == 64 else 4
    grid = (triton.cdiv(n_elements, block_size), int(q_by_node.shape[0]))
    fused_acoustic_rhs_2d_kernel[grid](
        q_by_node,
        flux_by_face,
        dr,
        ds,
        lift,
        metric_x,
        metric_y,
        metric_dx,
        metric_dy,
        rhs_by_node,
        rhs_by_node if q_accumulate is None else q_accumulate,
        coefficient,
        n_elements,
        4 * n_elements,
        -(c0 * c0) * rho0,
        -(1.0 / rho0),
        NP=int(q_by_node.shape[0]),
        NFACE=int(flux_by_face.shape[0]),
        ACCUMULATE=q_accumulate is not None,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def launch_fused_er_rhs_2d(
    *,
    q_by_node: torch.Tensor,
    flux_by_face: torch.Tensor,
    dr: torch.Tensor,
    ds: torch.Tensor,
    lift: torch.Tensor,
    metric_x: torch.Tensor,
    metric_y: torch.Tensor,
    metric_dx: torch.Tensor,
    metric_dy: torch.Tensor,
    porous_mask: torch.Tensor,
    k_inf: torch.Tensor,
    inv_rho_inf: torch.Tensor,
    sponge_sigma: torch.Tensor,
    beta_ca: torch.Tensor,
    rho_ca: torch.Tensor,
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
    porous_start: int = 0,
    n_porous: int = 0,
    compact_ade: bool = False,
    block_size: int | None = None,
    num_warps: int | None = None,
):
    _require_triton()
    n_elements = int(q_by_node.shape[1] // 4)
    v100_er_shape = (
        _is_v100(q_by_node)
        and q_by_node.shape[0] == 15
        and beta_ca.numel() == 8
    )
    if block_size is None:
        block_size = 64 if v100_er_shape else 128
    if num_warps is None:
        num_warps = 2 if v100_er_shape else 4
    grid = (triton.cdiv(n_elements, block_size), int(q_by_node.shape[0]))
    fused_er_rhs_2d_kernel[grid](
        q_by_node,
        flux_by_face,
        dr,
        ds,
        lift,
        metric_x,
        metric_y,
        metric_dx,
        metric_dy,
        porous_mask,
        k_inf,
        inv_rho_inf,
        sponge_sigma,
        beta_ca,
        rho_ca,
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
        porous_start,
        n_porous,
        NP=int(q_by_node.shape[0]),
        NFACE=int(flux_by_face.shape[0]),
        NSTATES=int(beta_ca.numel()),
        ACCUMULATE=q_accumulate is not None,
        COMPACT_ADE=compact_ade,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def launch_fused_er_aux_update_diag_2d(
    *,
    q_by_node: torch.Tensor,
    porous_element_ids: torch.Tensor,
    z_beta: torch.Tensor,
    z_rho_x: torch.Tensor,
    z_rho_y: torch.Tensor,
    z_beta_work: torch.Tensor,
    z_rho_x_work: torch.Tensor,
    z_rho_y_work: torch.Tensor,
    beta_diag: torch.Tensor,
    beta_b: torch.Tensor,
    rho_diag: torch.Tensor,
    rho_b: torch.Tensor,
    coefficient: float,
    compact_state: bool = False,
    block_size: int | None = None,
    num_warps: int | None = None,
):
    _require_triton()
    n_elements = int(q_by_node.shape[1] // 4)
    v100_er_shape = (
        _is_v100(q_by_node)
        and q_by_node.shape[0] == 15
        and beta_diag.numel() == 8
    )
    if block_size is None:
        block_size = 64 if v100_er_shape else 128
    if num_warps is None:
        num_warps = 4
    grid = (
        triton.cdiv(int(porous_element_ids.numel()), block_size),
        int(q_by_node.shape[0]),
        int(beta_diag.numel()),
    )
    fused_er_aux_update_diag_kernel[grid](
        q_by_node,
        porous_element_ids,
        z_beta,
        z_rho_x,
        z_rho_y,
        z_beta_work,
        z_rho_x_work,
        z_rho_y_work,
        beta_diag,
        beta_b,
        rho_diag,
        rho_b,
        coefficient,
        n_elements,
        4 * n_elements,
        int(porous_element_ids.numel()),
        NP=int(q_by_node.shape[0]),
        NSTATES=int(beta_diag.numel()),
        COMPACT_ADE=compact_state,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def launch_pml_auxiliary_rhs_2d(
    *,
    q_by_node: torch.Tensor,
    rhs_by_node: torch.Tensor,
    q_accumulate: torch.Tensor | None,
    pml_sigma: torch.Tensor,
    pml_psi: torch.Tensor,
    pml_psi_work: torch.Tensor,
    coefficient: float,
    rho_c2: float,
    block_size: int | None = None,
    num_warps: int | None = None,
):
    _require_triton()
    n_elements = int(q_by_node.shape[1] // 4)
    if block_size is None:
        block_size = 64 if _is_v100(q_by_node) and q_by_node.shape[0] == 15 else 128
    if num_warps is None:
        num_warps = 2 if block_size == 64 else 4
    grid = (triton.cdiv(n_elements, block_size), int(q_by_node.shape[0]))
    pml_auxiliary_rhs_2d_kernel[grid](
        q_by_node,
        rhs_by_node,
        rhs_by_node if q_accumulate is None else q_accumulate,
        pml_sigma,
        pml_psi,
        pml_psi_work,
        coefficient,
        1.0 / float(rho_c2),
        float(rho_c2),
        n_elements,
        4 * n_elements,
        NP=int(q_by_node.shape[0]),
        ACCUMULATE=q_accumulate is not None,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def launch_sample_receivers_2d(
    *,
    pressure: torch.Tensor,
    element_ids: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
):
    _require_triton()
    block_size = triton.next_power_of_2(int(pressure.shape[0]))
    sample_receivers_2d_kernel[(int(element_ids.numel()),)](
        pressure,
        element_ids,
        weights,
        output,
        pressure_node_stride=int(pressure.stride(0)),
        pressure_element_stride=int(pressure.stride(1)),
        NP=int(pressure.shape[0]),
        BLOCK_SIZE=block_size,
        num_warps=1,
    )
