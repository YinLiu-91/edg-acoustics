"""2D extended-reaction acoustic solver with vector-fitted porous material ADEs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy
import scipy.io
import torch

import edg_acoustics.device_ini as device_ini
from .acoustics_simulation_2d import AcousticsSimulation2D
from .preprocessing import MaterialUpwindFlux2D


@dataclass
class VectorFittedSISO:
    """Real state-space realization of a fitted scalar transfer function."""

    A: numpy.ndarray
    B: numpy.ndarray
    C: numpy.ndarray
    D: float
    rmserr: float | None = None

    @staticmethod
    def _ensure_real(name: str, value: numpy.ndarray):
        array = numpy.real_if_close(numpy.asarray(value), tol=1000)
        if numpy.iscomplexobj(array):
            max_imag = float(numpy.max(numpy.abs(array.imag)))
            if max_imag > 1.0e-10:
                raise ValueError(f"{name} contains non-negligible imaginary parts.")
            array = array.real
        return numpy.asarray(array, dtype=float)

    @classmethod
    def from_mat(cls, data: dict[str, numpy.ndarray], prefix: str):
        A = cls._ensure_real(f"A_{prefix}", data[f"A_{prefix}"])
        B = cls._ensure_real(f"B_{prefix}", data[f"B_{prefix}"]).reshape(-1, 1)
        C = cls._ensure_real(f"C_{prefix}", data[f"C_{prefix}"]).reshape(1, -1)
        D_array = cls._ensure_real(f"D_{prefix}", data[f"D_{prefix}"]).reshape(-1)
        if D_array.size != 1:
            raise ValueError(f"D_{prefix} must be a scalar.")
        rmserr = None
        rmserr_key = f"rmserr_{prefix}"
        if rmserr_key in data:
            rmserr = float(numpy.asarray(data[rmserr_key]).reshape(-1)[0])
        return cls(A=A, B=B, C=C, D=float(D_array[0]), rmserr=rmserr)


@dataclass
class ExtendedReactionMaterialFit:
    """Vector-fitted compressibility and density transfer functions."""

    beta: VectorFittedSISO
    rho: VectorFittedSISO

    @classmethod
    def from_mat(cls, path: str | Path):
        path = Path(path)
        data = scipy.io.loadmat(path, squeeze_me=False)
        return cls(
            beta=VectorFittedSISO.from_mat(data, "beta"),
            rho=VectorFittedSISO.from_mat(data, "rho"),
        )


class ExtendedReactionSimulation2D(AcousticsSimulation2D):
    """2D acoustics with an ER porous subdomain and sponge absorbing layers."""

    def __init__(
        self,
        rho0: float,
        c0: float,
        Nx: int,
        mesh,
        BC_list: dict[str, int],
        domain_labels: dict[str, int],
        material_fit: ExtendedReactionMaterialFit,
        *,
        physical_bbox: tuple[float, float, float, float],
        sponge_thickness: float,
        sponge_sigma_max: float = 2500.0,
        node_tolerance: float = 1.0e-7,
    ):
        self.domain_labels = dict(domain_labels)
        self.material_fit = material_fit
        self.physical_bbox = tuple(float(value) for value in physical_bbox)
        self.sponge_thickness = float(sponge_thickness)
        self.sponge_sigma_max = float(sponge_sigma_max)
        super().__init__(
            rho0,
            c0,
            Nx,
            mesh,
            BC_list,
            node_tolerance=node_tolerance,
        )
        self._build_material_model()
        self.extra_results_metadata = {
            "domain_label_names": numpy.asarray(
                [label for label, _ in sorted(self.domain_labels.items())],
                dtype=object,
            ),
            "domain_label_values": numpy.asarray(
                [value for _, value in sorted(self.domain_labels.items())],
                dtype=int,
            ),
            "element_physical_labels": self.mesh.element_physical_labels.detach()
            .cpu()
            .numpy(),
            "beta_fit_rmserr": -1.0
            if self.material_fit.beta.rmserr is None
            else self.material_fit.beta.rmserr,
            "rho_fit_rmserr": -1.0
            if self.material_fit.rho.rmserr is None
            else self.material_fit.rho.rmserr,
            "physical_bbox": numpy.asarray(self.physical_bbox, dtype=float),
            "sponge_sigma_max": self.sponge_sigma_max,
            "sponge_thickness": self.sponge_thickness,
        }

    def _build_material_model(self):
        self.air_label = self.domain_labels["Air"]
        self.porous_label = self.domain_labels["Porous"]
        self.sponge_label = self.domain_labels["Sponge"]
        element_labels = self.mesh.element_physical_labels.to(
            device=self.device, dtype=torch.long
        )
        self._air_mask = element_labels == self.air_label
        self._porous_mask = element_labels == self.porous_label
        self._sponge_mask = element_labels == self.sponge_label
        self._porous_mask_2d = self._porous_mask.reshape(1, -1)

        beta_air = 1.0 / (self.rho0 * self.c0**2)
        rho_air = self.rho0
        beta_inf = torch.full(
            (self.N_elements,), beta_air, device=self.device, dtype=device_ini.dtype
        )
        rho_inf = torch.full(
            (self.N_elements,), rho_air, device=self.device, dtype=device_ini.dtype
        )
        beta_porous = float(self.material_fit.beta.D)
        rho_porous = float(self.material_fit.rho.D)
        if beta_porous <= 0.0 or rho_porous <= 0.0:
            raise ValueError("The ER material constant terms must be positive.")
        beta_inf[self._porous_mask] = beta_porous
        rho_inf[self._porous_mask] = rho_porous

        self.beta_inf_vector = beta_inf
        self.rho_inf_vector = rho_inf
        self.k_inf_vector = 1.0 / self.beta_inf_vector
        self.c_inf_vector = torch.sqrt(self.k_inf_vector / self.rho_inf_vector)
        self.z_inf_vector = self.rho_inf_vector * self.c_inf_vector
        self.inv_rho_inf_vector = 1.0 / self.rho_inf_vector

        self.beta_inf_elements = self.beta_inf_vector.reshape(1, -1)
        self.k_inf_elements = self.k_inf_vector.reshape(1, -1)
        self.inv_rho_inf_elements = self.inv_rho_inf_vector.reshape(1, -1)

        neighbor_elements = self.mesh.EToE.to(device=self.device, dtype=torch.long)
        self.flux = MaterialUpwindFlux2D(
            self.n_xyz,
            self._facewise_property(self.rho_inf_vector, neighbor_elements, "left"),
            self._facewise_property(self.rho_inf_vector, neighbor_elements, "right"),
            self._facewise_property(self.c_inf_vector, neighbor_elements, "left"),
            self._facewise_property(self.c_inf_vector, neighbor_elements, "right"),
        )
        self._build_sponge_sigma()
        self._init_material_state_space()

    def _facewise_property(
        self,
        property_vector: torch.Tensor,
        neighbor_elements: torch.Tensor,
        side: str,
    ):
        if side == "left":
            values = property_vector.unsqueeze(0).repeat(3, 1)
        elif side == "right":
            values = property_vector[neighbor_elements]
        else:
            raise ValueError(f"Unknown face side {side}.")
        return values.repeat_interleave(self.Nfp, dim=0)

    def _build_sponge_sigma(self):
        xmin, xmax, _, ymax = self.physical_bbox
        vertices = torch.as_tensor(
            self.mesh.vertices[:2], device=self.device, dtype=device_ini.dtype
        )
        centroids = vertices[:, self.mesh.EToV.to(dtype=torch.long)].mean(dim=1)
        x = centroids[0]
        y = centroids[1]
        side_depth = torch.maximum(
            torch.maximum(torch.zeros_like(x), xmin - x),
            torch.maximum(torch.zeros_like(x), x - xmax),
        )
        top_depth = torch.maximum(torch.zeros_like(y), y - ymax)
        depth = torch.maximum(side_depth, top_depth)
        normalized_depth = torch.clamp(depth / self.sponge_thickness, min=0.0, max=1.0)
        sigma = self.sponge_sigma_max * normalized_depth**2
        sigma = torch.where(self._sponge_mask, sigma, torch.zeros_like(sigma))
        self.sponge_sigma = sigma.reshape(1, -1)

    def _init_material_state_space(self):
        self.beta_A = torch.as_tensor(
            self.material_fit.beta.A, device=self.device, dtype=device_ini.dtype
        )
        self.beta_B = torch.as_tensor(
            self.material_fit.beta.B.reshape(-1),
            device=self.device,
            dtype=device_ini.dtype,
        )
        self.beta_CA = torch.as_tensor(
            (self.material_fit.beta.C @ self.material_fit.beta.A).reshape(-1),
            device=self.device,
            dtype=device_ini.dtype,
        )
        self.beta_CB = float(
            numpy.asarray(self.material_fit.beta.C @ self.material_fit.beta.B).reshape(
                -1
            )[0]
        )
        self.beta_D = float(self.material_fit.beta.D)

        self.rho_A = torch.as_tensor(
            self.material_fit.rho.A, device=self.device, dtype=device_ini.dtype
        )
        self.rho_B = torch.as_tensor(
            self.material_fit.rho.B.reshape(-1),
            device=self.device,
            dtype=device_ini.dtype,
        )
        self.rho_CA = torch.as_tensor(
            (self.material_fit.rho.C @ self.material_fit.rho.A).reshape(-1),
            device=self.device,
            dtype=device_ini.dtype,
        )
        self.rho_CB = float(
            numpy.asarray(self.material_fit.rho.C @ self.material_fit.rho.B).reshape(-1)[
                0
            ]
        )
        self.rho_D = float(self.material_fit.rho.D)

        self._aux_state_names = ["z_beta", "z_rho_x", "z_rho_y"]

    def init_IC(self, IC):
        super().init_IC(IC)
        self._allocate_auxiliary_state()

    def _allocate_auxiliary_state(self):
        beta_state_count = int(self.beta_A.shape[0])
        rho_state_count = int(self.rho_A.shape[0])
        self.z_beta = torch.zeros(
            (beta_state_count, self.Np, self.N_elements),
            device=self.device,
            dtype=device_ini.dtype,
        )
        self.z_rho_x = torch.zeros(
            (rho_state_count, self.Np, self.N_elements),
            device=self.device,
            dtype=device_ini.dtype,
        )
        self.z_rho_y = torch.zeros_like(self.z_rho_x)
        self._taylor_aux_work = {
            name: torch.empty_like(getattr(self, name)) for name in self._aux_state_names
        }
        self._taylor_aux_rhs = {
            name: torch.empty_like(getattr(self, name)) for name in self._aux_state_names
        }

    def _prepare_taylor_auxiliary_state(self):
        for name in self._aux_state_names:
            self._taylor_aux_work[name].copy_(getattr(self, name))

    def _accumulate_taylor_auxiliary_state(self, coefficient: float):
        for name in self._aux_state_names:
            state = getattr(self, name)
            rhs = self._taylor_aux_rhs[name]
            state.add_(rhs, alpha=coefficient)
            self._taylor_aux_work[name].copy_(rhs)

    def _active_auxiliary_state(self, name: str):
        if hasattr(self, "_taylor_aux_work") and name in self._taylor_aux_work:
            return self._taylor_aux_work[name]
        return getattr(self, name)

    def _material_rhs(self, A: torch.Tensor, B: torch.Tensor, state: torch.Tensor, field: torch.Tensor):
        if A.numel() == 0:
            return torch.zeros_like(state)
        rhs = torch.einsum("ij,jkn->ikn", A, state)
        rhs.add_(B.reshape(-1, 1, 1) * field.unsqueeze(0))
        return rhs

    def _collapse_memory(self, coefficients: torch.Tensor, state: torch.Tensor, like: torch.Tensor):
        if coefficients.numel() == 0:
            return torch.zeros_like(like)
        return torch.tensordot(coefficients, state, dims=([0], [0]))

    def _cache_boundary_parameters(self):
        super()._cache_boundary_parameters()
        for index, node in enumerate(self.BCnode):
            cache = self._BC_cache[index]
            element_ids = torch.remainder(
                node["vmap"].to(device=self.device, dtype=torch.long),
                self.N_elements,
            )
            cache["rho_boundary"] = self.rho_inf_vector[element_ids]
            cache["c_boundary"] = self.c_inf_vector[element_ids]
            cache["k_boundary"] = self.k_inf_vector[element_ids]
            cache["z_boundary"] = self.z_inf_vector[element_ids]

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
            boundary_temp.copy_(boundary_q[0] / cache["z_boundary"])
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
            boundary_temp.copy_(boundary_q[0] / cache["rho_boundary"])
            boundary_temp.addcmul_(
                cache["incoming_outgoing"],
                cache["c_boundary"],
                value=-0.5,
            )

            boundary_flux = cache["boundary_flux"]
            torch.mul(node["nx"], boundary_temp, out=boundary_flux[1])
            torch.mul(node["ny"], boundary_temp, out=boundary_flux[2])
            torch.mul(node["nz"], boundary_temp, out=boundary_flux[3])

            boundary_temp.copy_(bcvar["vn"])
            boundary_temp.add_(bcvar["ou"], alpha=-0.5)
            boundary_temp.add_(bcvar["in"], alpha=0.5)
            boundary_temp.mul_(cache["k_boundary"])
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
        divV = dVxdx + dVydy

        rhs_P = -self.k_inf_elements * divV
        rhs_Vx = -self.inv_rho_inf_elements * dPdx
        rhs_Vy = -self.inv_rho_inf_elements * dPdy
        rhs_Vz = torch.zeros_like(rhs_P)

        z_beta = self._active_auxiliary_state("z_beta")
        z_rho_x = self._active_auxiliary_state("z_rho_x")
        z_rho_y = self._active_auxiliary_state("z_rho_y")

        beta_memory = self._collapse_memory(self.beta_CA, z_beta, P)
        rho_memory_x = self._collapse_memory(self.rho_CA, z_rho_x, Vx)
        rho_memory_y = self._collapse_memory(self.rho_CA, z_rho_y, Vy)
        rhs_porous_P = -(divV + beta_memory + self.beta_CB * P) / self.beta_D
        rhs_porous_Vx = -(dPdx + rho_memory_x + self.rho_CB * Vx) / self.rho_D
        rhs_porous_Vy = -(dPdy + rho_memory_y + self.rho_CB * Vy) / self.rho_D
        rhs_P = torch.where(self._porous_mask_2d, rhs_porous_P, rhs_P)
        rhs_Vx = torch.where(self._porous_mask_2d, rhs_porous_Vx, rhs_Vx)
        rhs_Vy = torch.where(self._porous_mask_2d, rhs_porous_Vy, rhs_Vy)

        masked_P = P * self._porous_mask_2d
        masked_Vx = Vx * self._porous_mask_2d
        masked_Vy = Vy * self._porous_mask_2d
        self._taylor_aux_rhs["z_beta"].copy_(
            self._material_rhs(self.beta_A, self.beta_B, z_beta, masked_P)
        )
        self._taylor_aux_rhs["z_rho_x"].copy_(
            self._material_rhs(self.rho_A, self.rho_B, z_rho_x, masked_Vx)
        )
        self._taylor_aux_rhs["z_rho_y"].copy_(
            self._material_rhs(self.rho_A, self.rho_B, z_rho_y, masked_Vy)
        )

        rhs_P.add_(self.lift @ fluxP)
        rhs_Vx.add_(self.lift @ fluxVx)
        rhs_Vy.add_(self.lift @ fluxVy)
        rhs_Vz.add_(self.lift @ fluxVz)

        rhs_P.addcmul_(self.sponge_sigma, P, value=-1.0)
        rhs_Vx.addcmul_(self.sponge_sigma, Vx, value=-1.0)
        rhs_Vy.addcmul_(self.sponge_sigma, Vy, value=-1.0)

        return rhs_P, rhs_Vx, rhs_Vy, rhs_Vz, BCvar
