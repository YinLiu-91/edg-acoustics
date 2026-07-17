"""Minimal 2D acoustic DG solver using the existing TSI time integrator."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import meshio
import modepy
import numpy
import scipy.io
import torch
from scipy.spatial.qhull import Delaunay

from . import acoustics_2d_triton
import edg_acoustics.device_ini as device_ini
import edg_acoustics.simplex_dg as simplex_dg

if TYPE_CHECKING:
    from .boundary_condition import AbsorbBC
    from .initial_condition import InitialCondition
    from .mesh2d import Mesh2D
    from .preprocessing import Flux
    from .time_integration import TimeIntegrator


_TRITON_AVAILABLE = acoustics_2d_triton.TRITON_AVAILABLE


def _normalize_env_bool_mode(name: str, default: str = "0") -> bool:
    raw = os.environ.get(name, default)
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


class AcousticsSimulation2D:
    """2D first-order acoustic solver on triangular DG meshes."""

    def __init__(
        self,
        rho0: float,
        c0: float,
        Nx: int,
        mesh: Mesh2D,
        BC_list: dict[str, int],
        node_tolerance: float = 1.0e-7,
    ):
        self.device = device_ini.device
        self.rho0 = float(rho0)
        self.c0 = float(c0)
        self.mesh = mesh
        self.Nx = Nx
        self.N_elements = mesh.EToV.shape[1]
        self.N_triangles = self.N_elements
        self.N_tets = self.N_elements
        self.BC_list = BC_list
        self.dim = 2
        self.node_tolerance = node_tolerance
        self._use_fused_state_accumulation = False
        self.IC = None
        self.BC = None
        self.flux = None
        self.time_integrator = None
        self.rec = None

        if set(BC_list) != set(mesh.BC_faces):
            raise ValueError(
                "[edg_acoustics.AcousticsSimulation2D] All BC labels must be present "
                "in the mesh and all mesh labels must be present in BC_list."
            )

        self.Np = simplex_dg.simplex_num_nodes(2, Nx)
        self.Nfp = simplex_dg.simplex_num_face_nodes(2, Nx)
        self._init_local_system()
        self.configure_fast_paths()
        self.init_runtime_buffers()
        self.cache_static_indices()
        self._cuda_step_graphs = {}

    def _clear_cuda_step_graphs(self):
        if hasattr(self, "_cuda_step_graphs"):
            self._cuda_step_graphs.clear()

    def configure_fast_paths(
        self,
        *,
        use_packed_rhs: bool | None = None,
        use_triton_kernels: bool | None = None,
        use_triton_deep_rhs: bool | None = None,
        use_triton_interior_flux: bool | None = None,
        use_triton_boundary_ri: bool | None = None,
    ):
        default_packed_rhs = _normalize_env_bool_mode("EDG_ACOUSTICS_2D_PACKED_RHS", "0")
        default_triton = _normalize_env_bool_mode(
            "EDG_ACOUSTICS_2D_TRITON_KERNELS", "0"
        )
        default_deep_rhs = _normalize_env_bool_mode(
            "EDG_ACOUSTICS_2D_DEEP_FUSED_RHS", "0"
        )
        if use_packed_rhs is None:
            use_packed_rhs = default_packed_rhs
        if use_triton_kernels is None:
            use_triton_kernels = default_triton
        if use_triton_deep_rhs is None:
            use_triton_deep_rhs = default_deep_rhs
        if use_triton_interior_flux is None:
            use_triton_interior_flux = bool(use_triton_kernels)
        if use_triton_boundary_ri is None:
            use_triton_boundary_ri = bool(use_triton_kernels)

        triton_supported = _TRITON_AVAILABLE and self.device.type == "cuda"
        self._use_packed_rhs = bool(use_packed_rhs)
        self._triton_deep_rhs_requested = bool(use_triton_deep_rhs)
        self._use_triton_interior_flux = (
            self._use_packed_rhs and triton_supported and bool(use_triton_interior_flux)
        )
        self._use_triton_boundary_ri = (
            self._use_packed_rhs and triton_supported and bool(use_triton_boundary_ri)
        )
        self._use_triton_deep_rhs = (
            self._use_packed_rhs
            and triton_supported
            and bool(use_triton_deep_rhs)
            and self._supports_triton_deep_rhs()
        )
        self._use_fused_state_accumulation = self._use_packed_rhs
        if getattr(self, "time_integrator", None) is not None:
            self.time_integrator.L_operator_packed = getattr(self, "RHS_operator_packed", None)
            if self._use_fused_state_accumulation:
                self.time_integrator.L_operator_packed_accumulate = getattr(
                    self, "RHS_operator_packed_accumulate", None
                )
            else:
                self.time_integrator.L_operator_packed_accumulate = None
        self._clear_cuda_step_graphs()

    def _supports_triton_deep_rhs(self) -> bool:
        return self.Np > 0 and self.Np <= 32 and 3 * self.Nfp <= 32

    def _has_triton_deep_rhs(self) -> bool:
        return getattr(self, "_use_triton_deep_rhs", False)

    def _can_use_packed_time_integration(self) -> bool:
        return (
            getattr(self, "_use_packed_rhs", False)
            and hasattr(self, "Q_flat")
            and self.time_integrator is not None
            and hasattr(self.time_integrator, "step_dt_packed")
        )

    def _state_view(self, packed: torch.tensor):
        return packed.view(self.Np, 4, self.N_elements)

    def _flux_state_view(self, packed: torch.tensor):
        return packed.view(3 * self.Nfp, 4, self.N_elements)

    def _packed_rhs_buffer_for_view(self, rhs_view: torch.tensor):
        return rhs_view.reshape(self.Np, 4 * self.N_elements)

    def init_runtime_buffers(self):
        face_shape = self.Fscale.shape
        face_node_count = face_shape[0]
        kwargs = {"device": self.device, "dtype": device_ini.dtype}
        self._dVx = torch.empty(face_shape, **kwargs)
        self._dVy = torch.empty(face_shape, **kwargs)
        self._dVz = torch.empty(face_shape, **kwargs)
        self._dP = torch.empty(face_shape, **kwargs)
        self._face_left_packed = torch.empty(
            (4, face_node_count * self.N_elements), **kwargs
        )
        self._face_right_packed = torch.empty_like(self._face_left_packed)
        self._flux_by_face = torch.empty(
            (face_node_count, 4 * self.N_elements), **kwargs
        )
        self._flux_by_face_view = self._flux_state_view(self._flux_by_face)
        self._surface_by_node = torch.empty((self.Np, 4 * self.N_elements), **kwargs)
        self._surface_view = self._state_view(self._surface_by_node)
        self._q_by_node = torch.empty((self.Np, 4 * self.N_elements), **kwargs)
        self._q_by_node_view = self._state_view(self._q_by_node)
        self._dQdr_by_node = torch.empty_like(self._q_by_node)
        self._dQds_by_node = torch.empty_like(self._q_by_node)
        self._dQdr_view = self._state_view(self._dQdr_by_node)
        self._dQds_view = self._state_view(self._dQds_by_node)
        self._surface_metric_x = self.rst_xyz[0, 0]
        self._surface_metric_y = self.rst_xyz[1, 0]
        self._surface_metric_dx = self.rst_xyz[0, 1]
        self._surface_metric_dy = self.rst_xyz[1, 1]
        self._dr_contiguous = self.Dr.contiguous()
        self._ds_contiguous = self.Ds.contiguous()
        self._lift_contiguous = self.lift.contiguous()
        self._surface_metric_x_contiguous = self._surface_metric_x.contiguous()
        self._surface_metric_y_contiguous = self._surface_metric_y.contiguous()
        self._surface_metric_dx_contiguous = self._surface_metric_dx.contiguous()
        self._surface_metric_dy_contiguous = self._surface_metric_dy.contiguous()
        self._dPdx = torch.empty((self.Np, self.N_elements), **kwargs)
        self._dPdy = torch.empty_like(self._dPdx)
        self._dVxdx = torch.empty_like(self._dPdx)
        self._dVydy = torch.empty_like(self._dPdx)
        self._divV = torch.empty_like(self._dPdx)
        self._rhs_by_node_buffers = tuple(
            torch.empty((self.Np, 4 * self.N_elements), **kwargs) for _ in range(2)
        )
        self._rhs_by_node_views = tuple(
            self._state_view(rhs) for rhs in self._rhs_by_node_buffers
        )
        self._rhs_buffer_index = 0

    def _build_packed_face_indices(self, field_indices: torch.tensor):
        flat_indices = field_indices.reshape(-1)
        node_ids = torch.div(flat_indices, self.N_elements, rounding_mode="floor")
        element_ids = torch.remainder(flat_indices, self.N_elements)
        variable_offsets = (
            torch.arange(4, device=self.device, dtype=torch.long).reshape(4, 1)
            * self.N_elements
        )
        packed_indices = (
            node_ids.reshape(1, -1) * (4 * self.N_elements)
            + variable_offsets
            + element_ids.reshape(1, -1)
        )
        return packed_indices.reshape(-1)

    def _build_packed_flux_indices(self, face_indices: torch.tensor):
        flat_indices = face_indices.reshape(-1)
        face_ids = torch.div(flat_indices, self.N_elements, rounding_mode="floor")
        element_ids = torch.remainder(flat_indices, self.N_elements)
        variable_offsets = (
            torch.arange(4, device=self.device, dtype=torch.long).reshape(4, 1)
            * self.N_elements
        )
        packed_indices = (
            face_ids.reshape(1, -1) * (4 * self.N_elements)
            + variable_offsets
            + element_ids.reshape(1, -1)
        )
        return packed_indices.reshape(-1)

    def cache_static_indices(self):
        self._vmapM = self.vmapM.to(device=self.device, dtype=torch.long)
        self._vmapP = self.vmapP.to(device=self.device, dtype=torch.long)
        self._vmapM_q = self._build_packed_face_indices(self._vmapM)
        self._vmapP_q = self._build_packed_face_indices(self._vmapP)
        self._face_node_ids = self.Fmask.reshape(-1).to(device=self.device, dtype=torch.long)
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

    def _cache_flux_coefficients(self):
        if self.flux is None:
            return
        if hasattr(self.flux, "rho_left"):
            rho_left = self.flux.rho_left
            rho_right = self.flux.rho_right
            c_left = self.flux.c_left
            c_right = self.flux.c_right
        else:
            rho_left = torch.full_like(self.Fscale, self.rho0)
            rho_right = torch.full_like(self.Fscale, self.rho0)
            c_left = torch.full_like(self.Fscale, self.c0)
            c_right = torch.full_like(self.Fscale, self.c0)
        self._rho_left_flat = rho_left.reshape(-1).contiguous()
        self._rho_right_flat = rho_right.reshape(-1).contiguous()
        self._c_left_flat = c_left.reshape(-1).contiguous()
        self._c_right_flat = c_right.reshape(-1).contiguous()

    def _use_scaled_flux_kernels(self) -> bool:
        return (
            self._use_triton_interior_flux
            and self._use_triton_boundary_ri
            and all(cache.get("simple_RI", False) for cache in getattr(self, "_BC_cache", []))
        )

    def _next_rhs_buffers(self):
        self._rhs_buffer_index = 1 - self._rhs_buffer_index
        return (
            self._rhs_by_node_buffers[self._rhs_buffer_index],
            self._rhs_by_node_views[self._rhs_buffer_index],
        )

    def _init_local_system(self):
        vertices = torch.from_numpy(self.mesh.vertices).to(
            device=device_ini.device, dtype=device_ini.dtype
        )
        self.rst, self.xyz = simplex_dg.triangle_collocation_nodes(
            self.mesh.EToV,
            vertices,
            self.Nx,
        )
        self.V = simplex_dg.simplex_vandermonde(2, self.Nx, self.rst)
        self.Dr, self.Ds = simplex_dg.simplex_derivative_matrices(
            2, self.Nx, self.rst
        )
        rst_tensor = torch.from_numpy(self.rst).to(device_ini.device)
        self.Fmask = simplex_dg.triangle_fmask(rst_tensor, self.node_tolerance)
        self.lift = simplex_dg.triangle_lift(self.V, rst_tensor, self.Fmask).to(
            self.device
        )
        self.rst_xyz, self.J = simplex_dg.triangle_geometric_factors(
            self.xyz, self.Dr, self.Ds
        )
        n_xy, self.sJ = simplex_dg.triangle_normals(
            vertices, self.mesh.EToV, self.J, self.Fmask
        )
        self.n_xyz = torch.zeros(
            (3, 3 * self.Nfp, self.N_elements),
            device=self.device,
            dtype=device_ini.dtype,
        )
        self.n_xyz[:2] = n_xy.to(self.device, dtype=device_ini.dtype)
        self.sJ = self.sJ.to(self.device, dtype=device_ini.dtype)
        self.Fscale = self.sJ / self.J[self.Fmask.reshape(-1), :]

        nodeids = torch.arange(
            self.N_elements * self.Np, device=self.device, dtype=torch.long
        ).reshape(self.Np, self.N_elements)
        self.vmapM, self.vmapP = simplex_dg.simplex_build_maps(
            nodeids,
            self.xyz,
            self.mesh.EToE,
            self.mesh.EToF,
            self.Fmask,
            self.node_tolerance,
        )
        self.BCnode = simplex_dg.simplex_build_bcmaps(
            self.mesh.face_vertex_ids.to(device_ini.device),
            self.BC_list,
            self.mesh.EToV,
            self.vmapM,
            self.mesh.BC_faces,
            self.Fmask,
        )
        for node in self.BCnode:
            node["map"] = node["map"].to(device=self.device, dtype=torch.long)
            node["vmap"] = node["vmap"].to(device=self.device, dtype=torch.long)
            node["nx"] = self.n_xyz[0].reshape(-1)[node["map"]]
            node["ny"] = self.n_xyz[1].reshape(-1)[node["map"]]
            node["nz"] = self.n_xyz[2].reshape(-1)[node["map"]]
            node["fscale"] = self.Fscale.reshape(-1)[node["map"]]

        self.dtscale = simplex_dg.diameter_2d(self.Fscale) / self.c0 / (
            2 * self.Nx + 1
        )

    def _xyz_for_ic(self):
        xyz_ic = torch.zeros(
            (3, self.Np, self.N_elements),
            device=self.device,
            dtype=device_ini.dtype,
        )
        xyz_ic[:2] = self.xyz
        return xyz_ic

    def init_IC(self, IC: InitialCondition):
        self.IC = IC
        xyz_ic = self._xyz_for_ic()
        P = self.IC.Pinit(xyz_ic).to(device=self.device, dtype=device_ini.dtype)
        Vx = self.IC.VXinit(xyz_ic).to(device=self.device, dtype=device_ini.dtype)
        Vy = self.IC.VYinit(xyz_ic).to(device=self.device, dtype=device_ini.dtype)
        Vz = self.IC.VZinit(xyz_ic).to(device=self.device, dtype=device_ini.dtype)
        self.Q = torch.zeros(
            (self.Np, 4, self.N_elements), device=self.device, dtype=device_ini.dtype
        )
        self.Q_flat = self.Q.reshape(self.Np, 4 * self.N_elements)
        self.P = self.Q[:, 0, :]
        self.Vx = self.Q[:, 1, :]
        self.Vy = self.Q[:, 2, :]
        self.Vz = self.Q[:, 3, :]
        self.P.copy_(P)
        self.Vx.copy_(Vx)
        self.Vy.copy_(Vy)
        self.Vz.copy_(Vz)
        self._clear_cuda_step_graphs()

    def init_BC(self, BC: AbsorbBC):
        self.BC = BC
        self._cache_boundary_parameters()
        self._clear_cuda_step_graphs()

    def init_Flux(self, Flux: Flux):
        self.flux = Flux
        self._cache_flux_coefficients()
        self._clear_cuda_step_graphs()

    def init_TimeIntegrator(self, time_integrator: TimeIntegrator):
        self.time_integrator = time_integrator
        self.time_integrator.L_operator_packed = getattr(self, "RHS_operator_packed", None)
        if self._use_fused_state_accumulation:
            self.time_integrator.L_operator_packed_accumulate = getattr(
                self, "RHS_operator_packed_accumulate", None
            )
        else:
            self.time_integrator.L_operator_packed_accumulate = None
        self._clear_cuda_step_graphs()

    @staticmethod
    def locate_simplex_2d(
        node_coordinates: numpy.ndarray,
        EToV: numpy.ndarray,
        rec: numpy.ndarray,
        methodLocate: str = "scipy",
    ):
        """Locate the triangles containing the requested sample points."""
        rec_xy = rec[:2]
        node_xy = node_coordinates[:2]
        if methodLocate == "scipy":
            tri = Delaunay(node_xy.T, qhull_options="QJ")
            tri.simplices = EToV.T  # type: ignore
            tri.nsimplex = EToV.shape[1]  # type: ignore
            nodeindex = tri.find_simplex(rec_xy.T)  # type: ignore
        else:
            raise ValueError(
                f"{methodLocate} is not an available 2D search method."
            )

        if numpy.any(nodeindex < 0):
            raise ValueError("Some receiver points are outside of the 2D mesh.")
        return nodeindex

    def sample2D(self, methodLocate: str = "scipy"):
        """Compute interpolation weights for 2D receiver sampling."""
        nodeindex = AcousticsSimulation2D.locate_simplex_2d(
            self.mesh.vertices,
            torch.Tensor.numpy(self.mesh.EToV.cpu()),
            self.rec,
            methodLocate,
        )
        old_nodes = self.xyz[:, :, nodeindex]
        simplex_basis = modepy.simplex_onb(self.dim, self.Nx)
        v_new = modepy.vandermonde(simplex_basis, self.rec[:2])
        sampleWeight = numpy.zeros([self.rec.shape[1], len(simplex_basis)])

        for index in range(old_nodes.shape[2]):
            v_old = modepy.vandermonde(
                simplex_basis, torch.Tensor.numpy(old_nodes[:, :, index].cpu())
            )
            sampleWeight[index] = v_new[index] @ numpy.linalg.inv(v_old)

        return (
            torch.from_numpy(sampleWeight).to(self.device).to(device_ini.dtype),
            nodeindex,
        )

    def init_rec(self, rec: numpy.ndarray, methodLocate: str = "scipy"):
        """Initialize receiver interpolation metadata for 2D runs."""
        self.rec = rec
        self.sampleWeight, self.nodeindex = self.sample2D(methodLocate)
        normalized_nodeindex = numpy.mod(self.nodeindex, self.N_elements)
        self._nodeindex_tensor = torch.as_tensor(
            normalized_nodeindex, device=self.device, dtype=torch.long
        )
        self._sample_values = torch.empty(
            (self.Np, self.rec.shape[1]), device=self.device, dtype=device_ini.dtype
        )
        self._sample_output = torch.empty(
            (self.rec.shape[1],), device=self.device, dtype=device_ini.dtype
        )
        self._clear_cuda_step_graphs()

    def _sample_receivers(self, out: torch.tensor):
        torch.index_select(self.P, 1, self._nodeindex_tensor, out=self._sample_values)
        torch.sum(self.sampleWeight * self._sample_values.T, dim=1, out=out)

    def _cache_boundary_parameters(self):
        self._BC_cache = []
        for index, paras in enumerate(self.BC.BCpara):
            bcvar = self.BC.BCvar[index]
            cache = {
                "RI_value": float(paras["RI"]),
                "ri_tensor": torch.tensor(
                    [float(paras["RI"])], device=self.device, dtype=device_ini.dtype
                ),
                "rho_boundary": torch.full_like(bcvar["vn"], self.rho0),
                "c_boundary": torch.full_like(bcvar["vn"], self.c0),
                "z_boundary": torch.full_like(bcvar["vn"], self.rho0 * self.c0),
                "k_boundary": torch.full_like(bcvar["vn"], self.rho0 * self.c0**2),
                "boundary_q": torch.empty(
                    (4, bcvar["vn"].numel()),
                    device=self.device,
                    dtype=device_ini.dtype,
                ),
                "boundary_temp": torch.empty_like(bcvar["vn"]),
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
                cache["kexi1_temp"] = torch.empty_like(bcvar["kexi1"])
            cache["simple_RI"] = "RP_A" not in cache and "CP_B" not in cache
            self._BC_cache.append(cache)

    def _gradient(self, field: torch.Tensor):
        ddr = self.Dr @ field
        dds = self.Ds @ field
        dx = self.rst_xyz[0, 0] * ddr + self.rst_xyz[1, 0] * dds
        dy = self.rst_xyz[0, 1] * ddr + self.rst_xyz[1, 1] * dds
        return dx, dy

    def _jump(self, field: torch.Tensor):
        field_flat = field.reshape(-1)
        return (field_flat[self.vmapM.long()] - field_flat[self.vmapP.long()]).reshape(
            3 * self.Nfp, self.N_elements
        )

    def _compute_boundary_flux(
        self,
        fluxP: torch.Tensor,
        fluxVx: torch.Tensor,
        fluxVy: torch.Tensor,
        fluxVz: torch.Tensor,
        P: torch.Tensor,
        Vx: torch.Tensor,
        Vy: torch.Tensor,
        Vz: torch.Tensor,
        BCvar: list[dict],
    ):
        P_flat = P.reshape(-1)
        Vx_flat = Vx.reshape(-1)
        Vy_flat = Vy.reshape(-1)
        Vz_flat = Vz.reshape(-1)
        fluxP_flat = fluxP.reshape(-1)
        fluxVx_flat = fluxVx.reshape(-1)
        fluxVy_flat = fluxVy.reshape(-1)
        fluxVz_flat = fluxVz.reshape(-1)

        for index, node in enumerate(self.BCnode):
            bcvar = BCvar[index]
            cache = self._BC_cache[index]
            boundary_q = cache["boundary_q"]
            boundary_q[0].copy_(P_flat[node["vmap"]])
            boundary_q[1].copy_(Vx_flat[node["vmap"]])
            boundary_q[2].copy_(Vy_flat[node["vmap"]])
            boundary_q[3].copy_(Vz_flat[node["vmap"]])

            torch.mul(node["nx"], boundary_q[1], out=bcvar["vn"])
            bcvar["vn"].addcmul_(node["ny"], boundary_q[2])
            bcvar["vn"].addcmul_(node["nz"], boundary_q[3])

            boundary_temp = cache["boundary_temp"]
            torch.mul(boundary_q[0], 1.0 / (self.rho0 * self.c0), out=boundary_temp)
            torch.add(bcvar["vn"], boundary_temp, out=bcvar["ou"])
            torch.mul(bcvar["ou"], cache["RI_value"], out=bcvar["in"])

            if "RP_A" in cache:
                phi = bcvar["phi"]
                torch.mul(cache["RP_A"], phi, out=cache["RP_terms"])
                bcvar["in"].add_(torch.sum(cache["RP_terms"], dim=0))
                phi.copy_(bcvar["ou"].unsqueeze(0) - cache["RP_zeta"] * phi)

            if "CP_B" in cache:
                kexi1 = bcvar["kexi1"]
                kexi2 = bcvar["kexi2"]
                torch.mul(cache["CP_B"], kexi1, out=cache["CP_terms"])
                cache["CP_terms"].addcmul_(cache["CP_C"], kexi2)
                bcvar["in"].add_(torch.sum(cache["CP_terms"], dim=0))
                cache["kexi1_temp"].copy_(kexi1)
                kexi1.copy_(
                    bcvar["ou"].unsqueeze(0)
                    - cache["CP_alpha"] * kexi1
                    - cache["CP_beta"] * kexi2
                )
                kexi2.copy_(
                    -cache["CP_alpha"] * kexi2
                    + cache["CP_beta"] * cache["kexi1_temp"]
                )

            torch.add(bcvar["ou"], bcvar["in"], out=cache["incoming_outgoing"])
            torch.mul(boundary_q[0], 1.0 / self.rho0, out=boundary_temp)
            boundary_temp.add_(cache["incoming_outgoing"], alpha=-0.5 * self.c0)

            boundary_flux = cache["boundary_flux"]
            torch.mul(node["nx"], boundary_temp, out=boundary_flux[1])
            torch.mul(node["ny"], boundary_temp, out=boundary_flux[2])
            torch.mul(node["nz"], boundary_temp, out=boundary_flux[3])

            boundary_temp.copy_(bcvar["vn"])
            boundary_temp.add_(bcvar["ou"], alpha=-0.5)
            boundary_temp.add_(bcvar["in"], alpha=0.5)
            boundary_temp.mul_((self.c0**2) * self.rho0)
            boundary_flux[0].copy_(boundary_temp)

            fluxP_flat[node["map"]] = boundary_flux[0]
            fluxVx_flat[node["map"]] = boundary_flux[1]
            fluxVy_flat[node["map"]] = boundary_flux[2]
            fluxVz_flat[node["map"]] = boundary_flux[3]

    def _pack_fields_by_node(
        self,
        P: torch.tensor,
        Vx: torch.tensor,
        Vy: torch.tensor,
        Vz: torch.tensor,
    ):
        if (
            hasattr(self, "Q_flat")
            and P.data_ptr() == self.P.data_ptr()
            and Vx.data_ptr() == self.Vx.data_ptr()
            and Vy.data_ptr() == self.Vy.data_ptr()
            and Vz.data_ptr() == self.Vz.data_ptr()
        ):
            return self.Q_flat
        q_view = self._q_by_node_view
        q_view[:, 0, :].copy_(P)
        q_view[:, 1, :].copy_(Vx)
        q_view[:, 2, :].copy_(Vy)
        q_view[:, 3, :].copy_(Vz)
        return self._q_by_node

    def _compute_packed_derivatives(self, q_by_node: torch.tensor):
        torch.mm(self.Dr, q_by_node, out=self._dQdr_by_node)
        torch.mm(self.Ds, q_by_node, out=self._dQds_by_node)

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

    def _compute_interior_flux_packed(self, q_by_node: torch.tensor):
        flux_view = self._flux_by_face_view
        scale_in_kernels = self._use_scaled_flux_kernels()
        if self._use_triton_interior_flux:
            acoustics_2d_triton.launch_interior_material_flux_2d(
                q_by_node=q_by_node,
                face_node_ids=self._face_node_ids,
                vmap_p_q=self._vmapP_q,
                nx=self._nx_flat,
                ny=self._ny_flat,
                rho_left=self._rho_left_flat,
                rho_right=self._rho_right_flat,
                c_left=self._c_left_flat,
                c_right=self._c_right_flat,
                fscale=self.Fscale,
                flux_by_face=self._flux_by_face,
                n_elements=self.N_elements,
                scale_flux=scale_in_kernels,
            )
            return

        self._compute_packed_jump(q_by_node)
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

    def _compute_boundary_flux_packed(
        self,
        bc_cache: dict,
        node: dict,
        bcvar: dict,
        q_flat: torch.tensor,
        flux_flat: torch.tensor,
    ):
        n_boundary = bcvar["vn"].numel()
        scale_in_kernels = self._use_scaled_flux_kernels()
        if self._use_triton_boundary_ri and bc_cache.get("simple_RI", False):
            acoustics_2d_triton.launch_boundary_ri_flux_2d(
                q_flat=q_flat,
                vmap_q=node["vmap_q"],
                flux_map_q=node["flux_map_q"],
                nx=node["nx"],
                ny=node["ny"],
                rho=bc_cache["rho_boundary"],
                c=bc_cache["c_boundary"],
                z=bc_cache["z_boundary"],
                k=bc_cache["k_boundary"],
                fscale=node["fscale"],
                flux_flat=flux_flat,
                vn=bcvar["vn"],
                ou=bcvar["ou"],
                incoming=bcvar["in"],
                ri_tensor=bc_cache["ri_tensor"],
                scale_flux=scale_in_kernels,
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
        torch.div(boundary_p, bc_cache["z_boundary"], out=boundary_temp)
        torch.add(bcvar["vn"], boundary_temp, out=bcvar["ou"])
        torch.mul(bcvar["ou"], bc_cache["RI_value"], out=bcvar["in"])

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
        torch.div(boundary_p, bc_cache["rho_boundary"], out=boundary_temp)
        boundary_temp.addcmul_(incoming_outgoing, bc_cache["c_boundary"], value=-0.5)

        torch.mul(node["nx"], boundary_temp, out=boundary_flux[1])
        torch.mul(node["ny"], boundary_temp, out=boundary_flux[2])
        torch.mul(node["nz"], boundary_temp, out=boundary_flux[3])

        boundary_temp.copy_(bcvar["vn"])
        boundary_temp.add_(bcvar["ou"], alpha=-0.5)
        boundary_temp.add_(bcvar["in"], alpha=0.5)
        boundary_temp.mul_(bc_cache["k_boundary"])
        boundary_flux[0].copy_(boundary_temp)
        flux_flat[node["flux_map_q"]] = boundary_flux.reshape(-1)

    def _compute_lift_surface(self):
        torch.mm(self.lift, self._flux_by_face, out=self._surface_by_node)

    def _compute_fused_packed_rhs_triton(
        self,
        q_by_node: torch.Tensor,
        rhs_by_node: torch.Tensor,
        q_accumulate: torch.Tensor | None,
        coefficient: float,
    ):
        acoustics_2d_triton.launch_fused_acoustic_rhs_2d(
            q_by_node=q_by_node,
            flux_by_face=self._flux_by_face,
            dr=self._dr_contiguous,
            ds=self._ds_contiguous,
            lift=self._lift_contiguous,
            metric_x=self._surface_metric_x_contiguous,
            metric_y=self._surface_metric_y_contiguous,
            metric_dx=self._surface_metric_dx_contiguous,
            metric_dy=self._surface_metric_dy_contiguous,
            rhs_by_node=rhs_by_node,
            q_accumulate=q_accumulate,
            coefficient=coefficient,
            c0=self.c0,
            rho0=self.rho0,
        )

    def _rhs_operator_packed_pre_lift(
        self,
        q_by_node: torch.tensor,
        BCvar: list[dict],
        *,
        compute_derivatives: bool = True,
    ):
        if self.flux is None:
            raise RuntimeError("Flux must be initialized before time integration.")
        if self.BC is None:
            raise RuntimeError("Boundary conditions must be initialized before time integration.")
        RHS_Q, RHS_Q_view = self._next_rhs_buffers()
        if compute_derivatives:
            self._compute_packed_derivatives(q_by_node)
        self._compute_interior_flux_packed(q_by_node)
        flux_flat = self._flux_by_face.reshape(-1)
        q_flat = q_by_node.reshape(-1)
        for index, bc_cache in enumerate(self._BC_cache):
            self._compute_boundary_flux_packed(
                bc_cache,
                self.BCnode[index],
                BCvar[index],
                q_flat,
                flux_flat,
            )
        if not self._use_scaled_flux_kernels():
            self._flux_by_face_view.mul_(self.Fscale.unsqueeze(1))
        return RHS_Q, RHS_Q_view, BCvar

    def _compute_volume_rhs_packed(
        self,
        q_by_node: torch.tensor,
        rhs_view: torch.tensor,
    ):
        dQdr = self._dQdr_view
        dQds = self._dQds_view
        dPdr = dQdr[:, 0, :]
        dPds = dQds[:, 0, :]
        dVxdr = dQdr[:, 1, :]
        dVxds = dQds[:, 1, :]
        dVydr = dQdr[:, 2, :]
        dVyds = dQds[:, 2, :]

        torch.mul(self._surface_metric_x, dPdr, out=self._dPdx)
        self._dPdx.addcmul_(self._surface_metric_y, dPds)
        torch.mul(self._surface_metric_dx, dPdr, out=self._dPdy)
        self._dPdy.addcmul_(self._surface_metric_dy, dPds)

        torch.mul(self._surface_metric_x, dVxdr, out=self._dVxdx)
        self._dVxdx.addcmul_(self._surface_metric_y, dVxds)
        torch.mul(self._surface_metric_dx, dVydr, out=self._dVydy)
        self._dVydy.addcmul_(self._surface_metric_dy, dVyds)
        torch.add(self._dVxdx, self._dVydy, out=self._divV)

        rhs_P = rhs_view[:, 0, :]
        rhs_Vx = rhs_view[:, 1, :]
        rhs_Vy = rhs_view[:, 2, :]
        rhs_Vz = rhs_view[:, 3, :]
        torch.mul(self._divV, -(self.c0**2) * self.rho0, out=rhs_P)
        torch.mul(self._dPdx, -(1.0 / self.rho0), out=rhs_Vx)
        torch.mul(self._dPdy, -(1.0 / self.rho0), out=rhs_Vy)
        rhs_Vz.zero_()

    def _rhs_operator_packed_post_lift(
        self,
        q_by_node: torch.tensor,
        RHS_Q: torch.tensor,
        RHS_Q_view: torch.tensor,
        q_accumulate: torch.tensor | None = None,
        accumulate_coefficient: float = 0.0,
    ):
        if self._has_triton_deep_rhs():
            self._compute_fused_packed_rhs_triton(
                q_by_node,
                RHS_Q,
                q_accumulate,
                accumulate_coefficient,
            )
            return
        self._compute_volume_rhs_packed(q_by_node, RHS_Q_view)
        RHS_Q.add_(self._surface_by_node)
        if q_accumulate is not None:
            q_accumulate.add_(RHS_Q, alpha=accumulate_coefficient)

    def RHS_operator_packed(
        self,
        q_by_node: torch.tensor,
        BCvar: list[dict],
        q_accumulate: torch.tensor | None = None,
        accumulate_coefficient: float = 0.0,
    ):
        use_deep_rhs = self._has_triton_deep_rhs()
        RHS_Q, RHS_Q_view, BCvar = self._rhs_operator_packed_pre_lift(
            q_by_node,
            BCvar,
            compute_derivatives=not use_deep_rhs,
        )
        if not use_deep_rhs:
            self._compute_lift_surface()
        self._rhs_operator_packed_post_lift(
            q_by_node,
            RHS_Q,
            RHS_Q_view,
            q_accumulate,
            accumulate_coefficient,
        )
        return RHS_Q, BCvar

    def RHS_operator_packed_accumulate(
        self,
        q_by_node: torch.tensor,
        BCvar: list[dict],
        q_accumulate: torch.tensor,
        coefficient: float,
    ):
        return self.RHS_operator_packed(q_by_node, BCvar, q_accumulate, coefficient)

    def _RHS_operator_reference(
        self,
        P: torch.Tensor,
        Vx: torch.Tensor,
        Vy: torch.Tensor,
        Vz: torch.Tensor,
        BCvar: list[dict],
    ):
        if self.flux is None:
            raise RuntimeError("Flux must be initialized before time integration.")
        if self.BC is None:
            raise RuntimeError("Boundary conditions must be initialized before time integration.")

        dVx = self._jump(Vx)
        dVy = self._jump(Vy)
        dVz = self._jump(Vz)
        dP = self._jump(P)

        face_shape = (3 * self.Nfp, self.N_elements)
        fluxVx = torch.empty(face_shape, device=self.device, dtype=device_ini.dtype)
        fluxVy = torch.empty_like(fluxVx)
        fluxVz = torch.empty_like(fluxVx)
        fluxP = torch.empty_like(fluxVx)
        self.flux.compute_all(dVx, dVy, dVz, dP, fluxVx, fluxVy, fluxVz, fluxP)
        self._compute_boundary_flux(fluxP, fluxVx, fluxVy, fluxVz, P, Vx, Vy, Vz, BCvar)

        fluxP.mul_(self.Fscale)
        fluxVx.mul_(self.Fscale)
        fluxVy.mul_(self.Fscale)
        fluxVz.mul_(self.Fscale)

        dPdx, dPdy = self._gradient(P)
        dVxdx, _ = self._gradient(Vx)
        _, dVydy = self._gradient(Vy)
        rhs_P = -(self.c0**2) * self.rho0 * (dVxdx + dVydy)
        rhs_Vx = -(1.0 / self.rho0) * dPdx
        rhs_Vy = -(1.0 / self.rho0) * dPdy
        rhs_Vz = torch.zeros_like(rhs_P)

        rhs_P.add_(self.lift @ fluxP)
        rhs_Vx.add_(self.lift @ fluxVx)
        rhs_Vy.add_(self.lift @ fluxVy)
        rhs_Vz.add_(self.lift @ fluxVz)
        return rhs_P, rhs_Vx, rhs_Vy, rhs_Vz, BCvar

    def RHS_operator(
        self,
        P: torch.Tensor,
        Vx: torch.Tensor,
        Vy: torch.Tensor,
        Vz: torch.Tensor,
        BCvar: list[dict],
    ):
        if not self._use_packed_rhs:
            return self._RHS_operator_reference(P, Vx, Vy, Vz, BCvar)
        q_by_node = self._pack_fields_by_node(P, Vx, Vy, Vz)
        rhs_by_node, BCvar = self.RHS_operator_packed(q_by_node, BCvar)
        rhs_view = self._state_view(rhs_by_node)
        return (
            rhs_view[:, 0, :],
            rhs_view[:, 1, :],
            rhs_view[:, 2, :],
            rhs_view[:, 3, :],
            BCvar,
        )

    def _reference_triangle_vertex_node_indices(self) -> tuple[int, int, int]:
        """Return the local nodal indices of the three triangle vertices."""
        cache_name = "_reference_triangle_vertex_node_indices_cache"
        if hasattr(self, cache_name):
            return getattr(self, cache_name)

        rst = numpy.asarray(self.rst)
        vertex_targets = numpy.array(
            (
                (-1.0, -1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
            ),
            dtype=rst.dtype,
        )
        vertex_indices: list[int] = []
        for target in vertex_targets:
            distances = numpy.linalg.norm(rst.T - target, axis=1)
            index = int(distances.argmin())
            if not numpy.allclose(rst[:, index], target, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(
                    "Failed to identify the triangle vertex nodes on the reference element."
                )
            vertex_indices.append(index)

        vertex_indices_tuple = tuple(vertex_indices)
        setattr(self, cache_name, vertex_indices_tuple)
        return vertex_indices_tuple

    def _build_vertex_visualization_point_data(self) -> dict[str, numpy.ndarray]:
        """Project DG nodal state to mesh vertices for visualization snapshots."""
        vertex_node_indices = self._reference_triangle_vertex_node_indices()
        element_vertices = torch.Tensor.numpy(self.mesh.EToV.cpu()).T
        n_vertices = self.mesh.N_vertices
        point_data: dict[str, numpy.ndarray] = {}

        for field_name in ("P", "Vx", "Vy", "Vz"):
            field = getattr(self, field_name).detach().cpu().numpy()
            accumulated = numpy.zeros(n_vertices, dtype=field.dtype)
            counts = numpy.zeros(n_vertices, dtype=numpy.int64)
            for local_vertex_position, local_node_index in enumerate(
                vertex_node_indices
            ):
                global_vertex_ids = element_vertices[:, local_vertex_position]
                local_values = field[local_node_index, :]
                numpy.add.at(accumulated, global_vertex_ids, local_values)
                numpy.add.at(counts, global_vertex_ids, 1)
            counts = numpy.maximum(counts, 1)
            point_data[field_name] = accumulated / counts

        return point_data

    def save_mesh_results_on_the_run(
        self,
        *,
        output_dir: str | None = None,
        step_index: int | None = None,
        real_time: float | None = None,
        file_format: str = "gmsh22",
    ):
        """Save a mesh snapshot with vertex visualization fields in gmsh format."""
        if output_dir is None:
            output_dir = "results_on_the_run_msh"
        os.makedirs(output_dir, exist_ok=True)

        if step_index is None:
            step_index = 0
        if real_time is None:
            real_time = 0.0

        mesh_snapshot = meshio.read(self.mesh.filename)
        mesh_snapshot.point_data = self._build_vertex_visualization_point_data()
        file_name = (
            f"{os.path.splitext(os.path.basename(self.mesh.filename))[0]}_"
            f"step{step_index:06d}_t{real_time:.6e}.msh"
        )
        mesh_path = os.path.join(output_dir, file_name)
        meshio.write(mesh_path, mesh_snapshot, file_format=file_format)

    def save_results_on_the_run(
        self,
        *,
        output_dir: str | Path | None = None,
        format: str = "mat",
        step_index: int | None = None,
    ):
        """Save the temporary 2D simulation results to a file."""
        if output_dir is None:
            output_dir = Path(".")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if step_index is None:
            step_index = getattr(self, "Ntimesteps", 0)

        data = {
            "BCpara": self.BC.BCpara,
            "P": self.P.detach().cpu().numpy(),
            "Vx": self.Vx.detach().cpu().numpy(),
            "Vy": self.Vy.detach().cpu().numpy(),
            "Vz": self.Vz.detach().cpu().numpy(),
            "xyz": self.xyz.detach().cpu().numpy(),
            "dt": self.time_integrator.dt,
            "current_step": step_index,
            "current_time": step_index * self.time_integrator.dt,
            "Ntimesteps": getattr(self, "Ntimesteps", step_index),
            "total_time": getattr(self, "Ntimesteps", step_index) * self.time_integrator.dt,
            "Np": self.Np,
            "N_elements": self.N_elements,
            "N_triangles": self.N_triangles,
            "rho0": self.rho0,
            "c0": self.c0,
            "mesh_filename": self.mesh.filename,
            "source_xyz": self.IC.source_xyz,
            "halfwidth": self.IC.halfwidth,
            "Nx": self.Nx,
            "Nt": self.time_integrator.Nt,
            "CFL": self.time_integrator.CFL,
        }
        if hasattr(self, "extra_results_metadata"):
            data.update(self.extra_results_metadata)
        if self.rec is not None:
            time_steps = int(step_index)
            data["rec"] = self.rec
            data["prec"] = self.prec[:, :time_steps].detach().cpu().numpy()
            data["time"] = (
                numpy.arange(1, time_steps + 1) * self.time_integrator.dt
            )
        if format == "mat":
            scipy.io.savemat(output_dir / "results_on_the_run.mat", data)
        elif format == "npy":
            numpy.savez(output_dir / "results_on_the_run.npz", **data)
        else:
            raise ValueError(
                "Invalid format, the format should be either 'mat' or 'npy'."
            )

    @staticmethod
    def _tensor_state_matches(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: float,
        atol: float,
    ) -> bool:
        if actual.is_floating_point() or actual.is_complex():
            return torch.allclose(actual, expected, rtol=rtol, atol=atol)
        return torch.equal(actual, expected)

    def _snapshot_time_state(self):
        auxiliary_state = {}
        for name in getattr(self, "_aux_state_names", ()):
            if hasattr(self, name):
                auxiliary_state[name] = getattr(self, name).clone()
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
            auxiliary_state,
        )

    def _restore_time_state(self, snapshot):
        q_snapshot, bc_snapshot, auxiliary_state = snapshot
        self.Q_flat.copy_(q_snapshot)
        for state, state_snapshot in zip(self.BC.BCvar, bc_snapshot):
            for key, value in state_snapshot.items():
                state[key].copy_(value)
        for name, value in auxiliary_state.items():
            getattr(self, name).copy_(value)

    def _time_state_matches(
        self,
        snapshot,
        *,
        rtol: float = 1.0e-10,
        atol: float = 1.0e-10,
    ) -> bool:
        q_snapshot, bc_snapshot, auxiliary_state = snapshot
        if not self._tensor_state_matches(
            self.Q_flat, q_snapshot, rtol=rtol, atol=atol
        ):
            return False
        for state, state_snapshot in zip(self.BC.BCvar, bc_snapshot):
            for key, value in state_snapshot.items():
                if not self._tensor_state_matches(
                    state[key], value, rtol=rtol, atol=atol
                ):
                    return False
        for name, value in auxiliary_state.items():
            if not self._tensor_state_matches(
                getattr(self, name), value, rtol=rtol, atol=atol
            ):
                return False
        return True

    def _run_cuda_step_chunk(
        self,
        chunk_steps: int,
        sample_chunk: torch.Tensor | None,
    ):
        for step_index in range(chunk_steps):
            if self._can_use_packed_time_integration():
                self.time_integrator.step_dt_packed(self.Q_flat, self.BC)
            else:
                self.time_integrator.step_dt(self.P, self.Vx, self.Vy, self.Vz, self.BC)
            if sample_chunk is not None:
                self._sample_receivers(sample_chunk[step_index])

    @staticmethod
    def _capture_cuda_graph_segment(fn):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        return graph

    def _capture_full_cuda_step_graph(
        self,
        chunk_steps: int,
        sample_chunk: torch.Tensor | None,
    ):
        return self._capture_cuda_graph_segment(
            lambda: self._run_cuda_step_chunk(chunk_steps, sample_chunk)
        )

    def _validate_cuda_step_graph(self, graph, snapshot, expected_snapshot) -> bool:
        self._restore_time_state(snapshot)
        graph.replay()
        torch.cuda.synchronize()
        graph_matches = self._time_state_matches(expected_snapshot)
        self._restore_time_state(snapshot)
        return graph_matches

    def _ensure_cuda_step_graph(
        self,
        chunk_steps: int = 1,
        record_receivers: bool = False,
    ):
        key = (chunk_steps, record_receivers)
        if key in self._cuda_step_graphs:
            return self._cuda_step_graphs[key]
        if (
            not torch.cuda.is_available()
            or not hasattr(self, "Q_flat")
            or self.Q_flat.device.type != "cuda"
        ):
            raise RuntimeError("CUDA graph time stepping requires CUDA tensors.")

        sample_chunk = None
        if record_receivers:
            sample_chunk = torch.empty(
                (chunk_steps, self.rec.shape[1]),
                device=self.device,
                dtype=device_ini.dtype,
            )

        snapshot = self._snapshot_time_state()
        self._run_cuda_step_chunk(chunk_steps, sample_chunk)
        torch.cuda.synchronize()
        expected_snapshot = self._snapshot_time_state()
        self._restore_time_state(snapshot)

        graph = self._capture_full_cuda_step_graph(chunk_steps, sample_chunk)
        if not self._validate_cuda_step_graph(graph, snapshot, expected_snapshot):
            raise RuntimeError("CUDA graph replay validation failed.")

        self._cuda_step_graphs[key] = (graph, sample_chunk, "full")
        return self._cuda_step_graphs[key]

    def time_integration(self, **kwargs):
        """Perform 2D time integration.

        Optional keyword arguments include ``n_time_steps``, ``total_time``,
        ``progress``, save controls, ``use_cuda_graph`` and
        ``cuda_graph_chunk_steps``. ``synchronize_timing`` optionally forces a
        CUDA synchronize before reading the final wall time. When
        ``use_cuda_graph`` is true and the state tensors live on CUDA, a
        validated CUDA graph is replayed for the time-step loop.
        """
        if self.time_integrator is None:
            raise RuntimeError("Time integrator must be initialized before time integration.")

        n_time_steps = kwargs.get("n_time_steps")
        total_time = kwargs.get("total_time")
        progress = kwargs.get("progress", False)
        synchronize_timing = kwargs.get("synchronize_timing", False)
        save_step = int(kwargs.get("save_step", 0) or 0)
        save_steps = {int(step) for step in kwargs.get("save_steps", [])}
        save_results_dir = kwargs.get("save_results_dir", None)
        save_format = kwargs.get("format", "mat")
        save_mesh_step = int(kwargs.get("save_mesh_step", 0) or 0)
        save_mesh_steps = {int(step) for step in kwargs.get("save_mesh_steps", [])}
        save_mesh_dir = kwargs.get("save_mesh_dir", None)
        save_mesh_format = kwargs.get("save_mesh_format", "gmsh22")
        cuda_graph_chunk_steps = max(1, int(kwargs.get("cuda_graph_chunk_steps", 1)))
        use_packed_rhs = self._can_use_packed_time_integration()
        use_cuda_graph = (
            kwargs.get("use_cuda_graph", False)
            and torch.cuda.is_available()
            and hasattr(self, "Q_flat")
            and self.Q_flat.device.type == "cuda"
        )
        if use_cuda_graph and cuda_graph_chunk_steps > 1 and (
            progress
            or save_step > 0
            or save_steps
            or save_mesh_step > 0
            or save_mesh_steps
        ):
            cuda_graph_chunk_steps = 1
        if n_time_steps is None:
            if total_time is None:
                raise ValueError("Set n_time_steps or total_time.")
            n_time_steps = math.floor(total_time / self.time_integrator.dt)
        self.Ntimesteps = int(n_time_steps)
        if self.Ntimesteps <= 0:
            use_cuda_graph = False
        elif use_cuda_graph:
            cuda_graph_chunk_steps = min(cuda_graph_chunk_steps, self.Ntimesteps)
        if self.rec is not None:
            self.prec = torch.zeros(
                (self.rec.shape[1], self.Ntimesteps),
                device=self.device,
                dtype=device_ini.dtype,
            )
        simulated_total_time = self.Ntimesteps * self.time_integrator.dt

        cuda_step_graph = None
        cuda_sample_chunk = None
        cuda_graph_mode = "disabled"
        if use_cuda_graph:
            (
                cuda_step_graph,
                cuda_sample_chunk,
                cuda_graph_mode,
            ) = self._ensure_cuda_step_graph(
                cuda_graph_chunk_steps,
                self.rec is not None,
            )

        start_time = time.time()
        step = 0
        while step < self.Ntimesteps:
            if (
                cuda_step_graph is not None
                and cuda_graph_chunk_steps > 1
                and step + cuda_graph_chunk_steps <= self.Ntimesteps
            ):
                cuda_step_graph.replay()
                if cuda_sample_chunk is not None:
                    self.prec[
                        :, step : step + cuda_graph_chunk_steps
                    ].copy_(cuda_sample_chunk.T)
                step += cuda_graph_chunk_steps
                continue

            receiver_sampled_by_graph = False
            if cuda_step_graph is not None and cuda_graph_chunk_steps == 1:
                cuda_step_graph.replay()
                if cuda_sample_chunk is not None:
                    self.prec[:, step].copy_(cuda_sample_chunk[0])
                    receiver_sampled_by_graph = True
            elif use_packed_rhs:
                self.time_integrator.step_dt_packed(self.Q_flat, self.BC)
            else:
                self.time_integrator.step_dt(self.P, self.Vx, self.Vy, self.Vz, self.BC)

            step_index = step + 1
            if self.rec is not None and not receiver_sampled_by_graph:
                self._sample_receivers(self._sample_output)
                self.prec[:, step].copy_(self._sample_output)
            if progress and (step % 10 == 0 or step + 1 == self.Ntimesteps):
                print(f"2D acoustic step {step_index}/{self.Ntimesteps}")
            should_save_results = (
                (save_step > 0 and step_index % save_step == 0)
                or step_index in save_steps
            )
            if should_save_results:
                self.save_results_on_the_run(
                    output_dir=save_results_dir,
                    format=save_format,
                    step_index=step_index,
                )
            should_save_mesh = (
                (save_mesh_step > 0 and step_index % save_mesh_step == 0)
                or step_index in save_mesh_steps
            )
            if should_save_mesh:
                self.save_mesh_results_on_the_run(
                    output_dir=save_mesh_dir,
                    step_index=step_index,
                    real_time=step_index * self.time_integrator.dt,
                    file_format=save_mesh_format,
                )
            step += 1

        if synchronize_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        self.last_time_integration_elapsed_s = end_time - start_time
        self.last_time_integration_steps = self.Ntimesteps
        self.last_time_integration_total_time = simulated_total_time
        self.last_time_integration_used_packed_rhs = use_packed_rhs
        self.last_time_integration_used_cuda_graph = cuda_step_graph is not None
        self.last_time_integration_cuda_graph_mode = cuda_graph_mode
        self.last_time_integration_cuda_graph_chunk_steps = (
            cuda_graph_chunk_steps if cuda_step_graph is not None else 0
        )
        if progress:
            print(f"time: {self.last_time_integration_elapsed_s} s")
        return self.P, self.Vx, self.Vy, self.Vz

    def write_snapshot(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "P": self.P.detach().cpu().numpy(),
            "Vx": self.Vx.detach().cpu().numpy(),
            "Vy": self.Vy.detach().cpu().numpy(),
            "Vz": self.Vz.detach().cpu().numpy(),
            "rho0": self.rho0,
            "c0": self.c0,
            "Nx": self.Nx,
            "N_elements": self.N_elements,
            "mesh_filename": self.mesh.filename,
        }
        if self.time_integrator is not None:
            data["dt"] = self.time_integrator.dt
        if path.suffix == ".mat":
            scipy.io.savemat(path, data)
        else:
            numpy.savez(path, **data)
