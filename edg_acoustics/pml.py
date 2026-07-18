"""Reusable region, damping, and auxiliary-state helpers for acoustic PMLs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


__all__ = ["PMLAugmentation", "PMLDamping", "PMLRegion"]


_PROFILE_IDS = {
    "constant": 0,
    "linear": 1,
    "quadratic": 2,
    "cubic": 3,
    "sine-linear": 4,
    "sine_linear": 4,
}


@dataclass(frozen=True)
class PMLRegion:
    """Select complete PML elements using a mesh region or Cartesian cutoffs."""

    cpml: tuple[float, ...]
    mode: str = "centered"
    region_name: str | None = None

    def element_mask(
        self,
        xyz: torch.Tensor,
        element_regions: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return a boolean mask with one entry per element."""
        if xyz.ndim != 3:
            raise ValueError("PML coordinates must have shape [dim, Np, K].")
        if len(self.cpml) != xyz.shape[0]:
            raise ValueError("PML cpml dimensionality must match the coordinates.")
        if self.mode not in {"centered", "ground"}:
            raise ValueError("PML region mode must be 'centered' or 'ground'.")

        if self.region_name is not None and element_regions is not None:
            if self.region_name not in element_regions:
                raise ValueError(f"PML region '{self.region_name}' is not in the mesh.")
            mask = torch.zeros(xyz.shape[2], device=xyz.device, dtype=torch.bool)
            element_ids = element_regions[self.region_name].to(
                device=xyz.device, dtype=torch.long
            )
            mask[element_ids] = True
            return mask

        centers = xyz.mean(dim=1)
        mask = torch.zeros(centers.shape[1], device=xyz.device, dtype=torch.bool)
        for axis, limit in enumerate(self.cpml):
            coordinate = centers[axis]
            if self.mode == "ground" and axis == 1:
                mask |= coordinate > float(limit)
            else:
                mask |= torch.abs(coordinate) > float(limit)
        return mask

    def interior_mask(
        self,
        xyz: torch.Tensor,
        element_regions: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return the complement of :meth:`element_mask`."""
        return ~self.element_mask(xyz, element_regions)


@dataclass(frozen=True)
class PMLDamping:
    """Build nodewise directional damping from Cartesian PML coordinates."""

    amp_sigma: float = 1000.0
    profile: str | int = "quadratic"
    scale_factor: float | tuple[float, ...] | None = None

    def _profile_id(self) -> int:
        if isinstance(self.profile, str):
            if self.profile not in _PROFILE_IDS:
                raise ValueError(f"Unsupported PML damping profile: {self.profile}")
            return _PROFILE_IDS[self.profile]
        profile_id = int(self.profile)
        if profile_id not in set(_PROFILE_IDS.values()):
            raise ValueError(f"Unsupported PML damping profile id: {profile_id}")
        return profile_id

    @staticmethod
    def _profile_value(profile_id: int, normalized: torch.Tensor) -> torch.Tensor:
        if profile_id == 0:
            return torch.ones_like(normalized)
        if profile_id == 1:
            return normalized
        if profile_id == 2:
            return normalized.square()
        if profile_id == 3:
            return normalized.pow(3)
        if profile_id == 4:
            return normalized - torch.sin(2.0 * torch.pi * normalized) / (
                2.0 * torch.pi
            )
        raise ValueError(f"Unsupported PML damping profile id: {profile_id}")

    def _scale_for_axis(self, axis: int, profile_id: int) -> float:
        if self.scale_factor is None:
            if profile_id == 0:
                return 1.0
            if profile_id == 4:
                return 2.0
            return float(profile_id + 1)
        if isinstance(self.scale_factor, tuple):
            if len(self.scale_factor) <= axis:
                raise ValueError("PML scale_factor dimensionality is too small.")
            return float(self.scale_factor[axis])
        return float(self.scale_factor)

    def compute(
        self,
        xyz: torch.Tensor,
        region: PMLRegion,
        element_regions: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return directional damping with shape ``[dim, Np, K]``."""
        if float(self.amp_sigma) < 0.0:
            raise ValueError("PML amp_sigma must be non-negative.")
        if xyz.ndim != 3 or len(region.cpml) != xyz.shape[0]:
            raise ValueError("PML cpml dimensionality must match the coordinates.")

        profile_id = self._profile_id()
        sigma = torch.zeros_like(xyz)
        for axis, limit in enumerate(region.cpml):
            coordinate = xyz[axis]
            outer = torch.max(torch.abs(coordinate))
            delta = outer - float(limit)
            if float(delta.item()) <= 0.0:
                raise ValueError("PML outer extent must be larger than cpml.")
            if region.mode == "ground" and axis == 1:
                distance = coordinate - float(limit)
            else:
                distance = torch.abs(coordinate) - float(limit)
            normalized = torch.clamp(distance / delta, min=0.0, max=1.0)
            values = self._profile_value(profile_id, normalized)
            values = torch.where(distance > 0.0, values, torch.zeros_like(values))
            scale = (
                float(self.amp_sigma)
                * self._scale_for_axis(axis, profile_id)
                / float(delta.item())
            )
            sigma[axis] = scale * values

        if region.region_name is not None and element_regions is not None:
            mask = region.element_mask(xyz, element_regions).reshape(1, 1, -1)
            sigma = torch.where(mask, sigma, torch.zeros_like(sigma))
        return sigma


class PMLAugmentation:
    """Two-dimensional first-order acoustic PML auxiliary-memory equations."""

    def __init__(
        self,
        sigma: torch.Tensor,
        rho0: float | torch.Tensor,
        c0: float | torch.Tensor,
    ):
        if sigma.ndim != 3 or sigma.shape[0] != 2:
            raise ValueError("The 2D PML sigma tensor must have shape [2, Np, K].")
        self.sigma = sigma
        self.rho0 = self._coefficient(rho0)
        self.c0 = self._coefficient(c0)

    def _coefficient(self, value: float | torch.Tensor):
        if torch.is_tensor(value):
            return value.to(device=self.sigma.device, dtype=self.sigma.dtype)
        return float(value)

    def apply_in_place(
        self,
        q: torch.Tensor,
        memory: torch.Tensor,
        acoustic_rhs: torch.Tensor,
        memory_rhs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append PML terms to an already assembled acoustic DG RHS."""
        if q.ndim != 3 or q.shape[1] != 3:
            raise ValueError("The 2D primary state must use [pressure, vx, vy].")
        if memory.shape != (q.shape[0], 2, q.shape[2]):
            raise ValueError("The 2D PML memory must use [psi_x, psi_y].")
        if acoustic_rhs.shape != q.shape:
            raise ValueError("The acoustic RHS shape must match the primary state.")
        if memory_rhs is None:
            memory_rhs = torch.empty_like(memory)
        elif memory_rhs.shape != memory.shape:
            raise ValueError("The PML memory RHS shape must match the memory state.")

        sigma_x = self.sigma[0]
        sigma_y = self.sigma[1]
        rho_c2 = self.rho0 * self.c0 * self.c0

        acoustic_rhs[:, 0, :].addcmul_(sigma_x + sigma_y, q[:, 0, :], value=-1.0)
        acoustic_rhs[:, 1, :].add_(memory[:, 0, :] / rho_c2, alpha=-1.0)
        acoustic_rhs[:, 2, :].add_(memory[:, 1, :] / rho_c2, alpha=-1.0)

        memory_rhs[:, 0, :].copy_(
            rho_c2 * (sigma_x - sigma_y) * acoustic_rhs[:, 1, :]
            - sigma_y * memory[:, 0, :]
        )
        memory_rhs[:, 1, :].copy_(
            rho_c2 * (sigma_y - sigma_x) * acoustic_rhs[:, 2, :]
            - sigma_x * memory[:, 1, :]
        )
        return acoustic_rhs, memory_rhs

    def rhs(
        self,
        q: torch.Tensor,
        memory: torch.Tensor,
        acoustic_rhs: torch.Tensor,
    ) -> torch.Tensor:
        """Return the combined ``[p, vx, vy, psi_x, psi_y]`` RHS."""
        rhs_q, rhs_memory = self.apply_in_place(q, memory, acoustic_rhs)
        return torch.cat((rhs_q, rhs_memory), dim=1)
