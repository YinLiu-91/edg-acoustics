"""edg_acoustics package."""

from __future__ import annotations

from edg_acoustics.acoustics_simulation import AcousticsSimulation
from edg_acoustics.boundary_condition import BoundaryCondition, AbsorbBC
from edg_acoustics.preprocessing import Flux, UpwindFlux
from edg_acoustics.initial_condition import InitialCondition, Monopole_IC
from edg_acoustics.time_integration import TimeIntegrator, RungeKutta
from edg_acoustics.mesh import Mesh
import torch  # 添加PyTorch导入

__all__ = [
    "AcousticsSimulation",
    "BoundaryCondition",
    "AbsorbBC",
    "Flux",
    "UpwindFlux",
    "InitialCondition",
    "Monopole_IC",
    "TimeIntegrator",
    "RungeKutta",
    "Mesh",
]
