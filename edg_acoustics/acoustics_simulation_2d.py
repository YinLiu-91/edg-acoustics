"""Minimal 2D acoustic DG solver using the existing TSI time integrator."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import meshio
import modepy
import numpy
import scipy.io
import torch
from scipy.spatial.qhull import Delaunay

import edg_acoustics.device_ini as device_ini
import edg_acoustics.simplex_dg as simplex_dg

if TYPE_CHECKING:
    from .boundary_condition import AbsorbBC
    from .initial_condition import InitialCondition
    from .mesh2d import Mesh2D
    from .preprocessing import Flux
    from .time_integration import TimeIntegrator


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

        if set(BC_list) != set(mesh.BC_faces):
            raise ValueError(
                "[edg_acoustics.AcousticsSimulation2D] All BC labels must be present "
                "in the mesh and all mesh labels must be present in BC_list."
            )

        self.Np = simplex_dg.simplex_num_nodes(2, Nx)
        self.Nfp = simplex_dg.simplex_num_face_nodes(2, Nx)
        self._init_local_system()
        self.IC = None
        self.BC = None
        self.flux = None
        self.time_integrator = None
        self.rec = None

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

    def init_BC(self, BC: AbsorbBC):
        self.BC = BC
        self._cache_boundary_parameters()

    def init_Flux(self, Flux: Flux):
        self.flux = Flux

    def init_TimeIntegrator(self, time_integrator: TimeIntegrator):
        self.time_integrator = time_integrator

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

    def _sample_receivers(self, out: torch.tensor):
        torch.index_select(self.P, 1, self._nodeindex_tensor, out=self._sample_values)
        torch.sum(self.sampleWeight * self._sample_values.T, dim=1, out=out)

    def _cache_boundary_parameters(self):
        self._BC_cache = []
        for index, paras in enumerate(self.BC.BCpara):
            bcvar = self.BC.BCvar[index]
            cache = {
                "RI_value": float(paras["RI"]),
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
            if "CP" in paras:
                cp = torch.as_tensor(
                    paras["CP"], device=self.device, dtype=device_ini.dtype
                )
                cache["CP_B"] = cp[0].reshape(-1, 1)
                cache["CP_C"] = cp[1].reshape(-1, 1)
                cache["CP_alpha"] = cp[2].reshape(-1, 1)
                cache["CP_beta"] = cp[3].reshape(-1, 1)
                cache["CP_terms"] = torch.empty_like(bcvar["kexi1"])
                cache["kexi1_temp"] = torch.empty_like(bcvar["kexi1"])
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

    def RHS_operator(
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

    def time_integration(self, **kwargs):
        if self.time_integrator is None:
            raise RuntimeError("Time integrator must be initialized before time integration.")

        n_time_steps = kwargs.get("n_time_steps")
        total_time = kwargs.get("total_time")
        progress = kwargs.get("progress", False)
        save_step = int(kwargs.get("save_step", 0) or 0)
        save_steps = {int(step) for step in kwargs.get("save_steps", [])}
        save_results_dir = kwargs.get("save_results_dir", None)
        save_format = kwargs.get("format", "mat")
        save_mesh_step = int(kwargs.get("save_mesh_step", 0) or 0)
        save_mesh_steps = {int(step) for step in kwargs.get("save_mesh_steps", [])}
        save_mesh_dir = kwargs.get("save_mesh_dir", None)
        save_mesh_format = kwargs.get("save_mesh_format", "gmsh22")
        if n_time_steps is None:
            if total_time is None:
                raise ValueError("Set n_time_steps or total_time.")
            n_time_steps = math.floor(total_time / self.time_integrator.dt)
        self.Ntimesteps = int(n_time_steps)
        if self.rec is not None:
            self.prec = torch.zeros(
                (self.rec.shape[1], self.Ntimesteps),
                device=self.device,
                dtype=device_ini.dtype,
            )

        for step in range(self.Ntimesteps):
            self.time_integrator.step_dt(self.P, self.Vx, self.Vy, self.Vz, self.BC)
            step_index = step + 1
            if self.rec is not None:
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
