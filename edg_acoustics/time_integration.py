"""This module contains the abstract base class for time integrators and the implementation of the Taylor-series time integration scheme.

The edg_acoustics.time_integration provide more necessary functionalities
(based upon :mod:`edg_acoustics.acoustics_simulation`) to setup time integration.
"""

from __future__ import annotations

import typing
import abc
import math
import numpy
import edg_acoustics
import torch

__all__ = ["TimeIntegrator", "TSI_TI", "CFL_Default"]

CFL_Default = 0.5
"""float: Default value of the CFL number for time integration schemes."""


class TimeIntegrator(abc.ABC):
    """Base class for time integrators.

    Args:
        L_operator (typing.Callable[[torch.tensor, torch.tensor, torch.tensor, torch.tensor, list]]): the function in AcousticSimulation that enables the computation of Lq, given q = [P, Vx, Vy, Vz]
        dtscale (float): the time step scale based on the mesh size measure
        CFL (float): the CFL number

    Attributes:
        L_operator (typing.Callable[[torch.tensor, torch.tensor, torch.tensor, torch.tensor, list]]): the function in AcousticSimulation that enables the computation of Lq, given q = [P, Vx, Vy, Vz]
        CFL (float): the CFL number
        dt (float): the time step size
    """

    def __init__(
        self,
        L_operator: typing.Callable[
            [torch.tensor, torch.tensor, torch.tensor, torch.tensor, list]
        ],
        dtscale: float,
        CFL: float = CFL_Default,
    ):
        self.L_operator = L_operator
        self.CFL = CFL
        self.dt = CFL * dtscale

    @abc.abstractmethod
    def step_dt(
        self,
        P: None,
        Vx: None,
        Vy: None,
        Vz: None,
        BC: None,
    ):
        """Takes the pressure, velocity at the time T and evolves it to time T + dt."""
        # P0 := P(T)
        # P := P(T + dt)
        # the same for all the other variables


class TSI_TI(TimeIntegrator):
    """Class for time integrator of Taylor-series time integration scheme.

    :class:`.TSI_TI` is used to evolve the pressure and velocity at the time T to time T + dt, based on the Taylor-series time integration scheme.

    Args:
        L_operator (typing.Callable[[torch.tensor, torch.tensor, torch.tensor, torch.tensor, list]]): the function in AcousticSimulation that enables the computation of Lq, given q = [P, Vx, Vy, Vz]
        dtscale (float): the time step scale based on the mesh size measure
        CFL (float): the CFL number
        Nt (int): the order of the time integration scheme.

    Attributes:
        L_operator (typing.Callable[[torch.tensor, torch.tensor, torch.tensor, torch.tensor, list]]): the function in AcousticSimulation that enables the computation of Lq, given q = [P, Vx, Vy, Vz]
        CFL (float): the CFL number
        dt (float): the time step size
        Nt (int): the order of the time integration scheme.
    """

    def __init__(
        self,
        L_operator: typing.Callable[
            [torch.tensor, torch.tensor, torch.tensor, torch.tensor, list]
        ],
        dtscale: float,
        CFL: float = CFL_Default,
        **kwargs,
    ):
        super().__init__(L_operator, dtscale, CFL)
        self.Nt = kwargs["Nt"]
        self.taylor_coefficients = [
            self.dt**Tind / math.factorial(Tind) for Tind in range(1, self.Nt + 1)
        ]
        owner = getattr(L_operator, "__self__", None)
        self.L_operator_packed = getattr(owner, "RHS_operator_packed", None)
        self._state_buffers = None
        self._packed_state_buffer = None
        print("TSI_TI initialized.")

    def _copy_state_to_buffers(
        self,
        P: torch.tensor,
        Vx: torch.tensor,
        Vy: torch.tensor,
        Vz: torch.tensor,
    ):
        if (
            self._state_buffers is None
            or self._state_buffers[0].shape != P.shape
            or self._state_buffers[0].device != P.device
            or self._state_buffers[0].dtype != P.dtype
        ):
            self._state_buffers = tuple(
                torch.empty_like(state) for state in (P, Vx, Vy, Vz)
            )

        P0, Vx0, Vy0, Vz0 = self._state_buffers
        P0.copy_(P)
        Vx0.copy_(Vx)
        Vy0.copy_(Vy)
        Vz0.copy_(Vz)
        return P0, Vx0, Vy0, Vz0

    def _copy_packed_state_to_buffer(self, Q_flat: torch.tensor):
        if (
            self._packed_state_buffer is None
            or self._packed_state_buffer.shape != Q_flat.shape
            or self._packed_state_buffer.device != Q_flat.device
            or self._packed_state_buffer.dtype != Q_flat.dtype
        ):
            self._packed_state_buffer = torch.empty_like(Q_flat)

        self._packed_state_buffer.copy_(Q_flat)
        return self._packed_state_buffer

    @staticmethod
    def _copy_boundary_derivatives(BC: edg_acoustics.AbsorbBC):
        for index, paras in enumerate(BC.BCpara):
            for polekey in paras:
                if polekey == "RP":
                    BC.BCvar[index]["phi"].copy_(BC.BCvar[index]["PHI"])
                elif polekey == "CP":
                    BC.BCvar[index]["kexi1"].copy_(BC.BCvar[index]["KEXI1"])
                    BC.BCvar[index]["kexi2"].copy_(BC.BCvar[index]["KEXI2"])

    def _accumulate_boundary_derivatives(
        self, BC: edg_acoustics.AbsorbBC, coefficient: float
    ):
        for index, paras in enumerate(BC.BCpara):
            for polekey in paras:
                if polekey == "RP":
                    BC.BCvar[index]["PHI"].add_(
                        BC.BCvar[index]["phi"], alpha=coefficient
                    )
                elif polekey == "CP":
                    BC.BCvar[index]["KEXI1"].add_(
                        BC.BCvar[index]["kexi1"], alpha=coefficient
                    )
                    BC.BCvar[index]["KEXI2"].add_(
                        BC.BCvar[index]["kexi2"], alpha=coefficient
                    )

    def step_dt_packed(
        self,
        Q_flat: torch.tensor,
        BC: edg_acoustics.AbsorbBC,
    ):
        """Evolve a packed ``[Np, 4 * N_tets]`` state by one time step."""
        if self.L_operator_packed is None:
            raise RuntimeError("Packed RHS operator is not available.")

        Q0 = self._copy_packed_state_to_buffer(Q_flat)
        self._copy_boundary_derivatives(BC)

        for coefficient in self.taylor_coefficients:
            Q0, BC.BCvar = self.L_operator_packed(Q0, BC.BCvar)
            Q_flat.add_(Q0, alpha=coefficient)
            self._accumulate_boundary_derivatives(BC, coefficient)

    def step_dt(
        self,
        P: torch.tensor,
        Vx: torch.tensor,
        Vy: torch.tensor,
        Vz: torch.tensor,
        BC: edg_acoustics.AbsorbBC,
    ):
        """Takes the pressure, velocity at the time T and evolves/updates them to time T + dt.

        Args:
            P (torch.tensor): the pressure at the time T, will be updated to the pressure at the time T + dt
            Vx (torch.tensor): the x-component of the velocity at the time T, will be updated to the x-component of the velocity at the time T + dt
            Vy (torch.tensor): the y-component of the velocity at the time T, will be updated to the y-component of the velocity at the time T + dt
            Vz (torch.tensor): the z-component of the velocity at the time T, will be updated to the z-component of the velocity at the time T + dt
            BC (edg_acoustics.AbsorbBC): the boundary condition object
        """
        # Takes the pressure, velocity, and BCvar at the time T and evolves it to time T + dt
        # BC(edg_acoustics.BoundaryCondition) object neesds to be passed as a reference since we need to access the BCpara attribute
        #   P0 contains the (high-order) derivative values
        #   P := P(T) and P(T + dt)
        # the same for all the other variables
        ##########################
        P0, Vx0, Vy0, Vz0 = self._copy_state_to_buffers(P, Vx, Vy, Vz)
        self._copy_boundary_derivatives(BC)

        ##########################
        for Tind, coefficient in enumerate(self.taylor_coefficients, start=1):
            # Compute L (L^{Tind-1} q)
            P0, Vx0, Vy0, Vz0, BC.BCvar = self.L_operator(P0, Vx0, Vy0, Vz0, BC.BCvar)

            # Add the Taylor term \frac{dt^{Tind}}{Tind!}L^{Tind}q
            Vx.add_(Vx0, alpha=coefficient)
            Vy.add_(Vy0, alpha=coefficient)
            Vz.add_(Vz0, alpha=coefficient)
            P.add_(P0, alpha=coefficient)

            self._accumulate_boundary_derivatives(BC, coefficient)
