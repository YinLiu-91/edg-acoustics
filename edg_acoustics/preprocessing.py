"""This module provides preprocessing functionalities for the edg_acoustics package."""

from __future__ import annotations
import abc
import numpy
import torch

__all__ = ["Flux", "UpwindFlux"]


class Flux(abc.ABC):
    """abstract base class for fluxes."""

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def FluxP(self):
        """abstract method for pressure flux."""

    @abc.abstractmethod
    def FluxVx(self):
        """abstract method for flux of velocity in x-direction."""

    @abc.abstractmethod
    def FluxVy(self):
        """abstract method for flux of velocity in y-direction."""

    @abc.abstractmethod
    def FluxVz(self):
        """abstract method for flux of velocity in z-direction."""


class UpwindFlux(Flux):
    """Calculation of upwind fluxes.
    Args:
        rho0 (float): The reference density.
        c0 (float): The reference speed of sound.
        n_xyz (torch.tensor): The array representing the normal vector of the face.

    Attributes:
        rho0 (float): see :attr:`edg_acoustics.AcousticsSimulation.rho0`.
        c0 (float): see :attr:`edg_acoustics.AcousticsSimulation.c0`.
        n_xyz (torch.tensor): see :attr:`edg_acoustics.AcousticsSimulation.n_xyz`.
        cn1s (torch.tensor): precalculated constant for the calculation of the flux of velocity in x-direction.
        cn2s (torch.tensor): precalculated constant for the calculation of the flux of velocity in y-direction.
        cn3s (torch.tensor): precalculated constant for the calculation of the flux of velocity in z-direction.
        cn1n2 (torch.tensor): precalculated constant for the calculation of the flux of velocity.
        cn1n3 (torch.tensor): precalculated constant for the calculation of the flux of velocity.
        cn2n3 (torch.tensor): precalculated constant for the calculation of the flux of velocity.
        n1rho (torch.tensor): precalculated constant for the calculation of the flux of velocity in x-direction.
        n2rho (torch.tensor): precalculated constant for the calculation of the flux of velocity in y-direction.
        n3rho (torch.tensor): precalculated constant for the calculation of the flux of velocity in z-direction.
        csn1rho (torch.tensor): precalculated constant for the calculation of the pressure flux.
        csn2rho (torch.tensor): precalculated constant for the calculation of the pressure flux.
        csn3rho (torch.tensor): precalculated constant for the calculation of the pressure flux.


    """

    def __init__(self, rho0: float, c0: float, n_xyz: torch.tensor):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rho0 = rho0
        self.c0 = c0
        self.n_xyz = n_xyz
        self.cn1s = -c0 * n_xyz[0] ** 2 / 2
        self.cn2s = -c0 * n_xyz[1] ** 2 / 2
        self.cn3s = -c0 * n_xyz[2] ** 2 / 2
        self.cn1n2 = -c0 * n_xyz[0] * n_xyz[1] / 2
        self.cn1n3 = -c0 * n_xyz[0] * n_xyz[2] / 2
        self.cn2n3 = -c0 * n_xyz[1] * n_xyz[2] / 2
        self.n1rho = n_xyz[0] / 2 / rho0
        self.n2rho = n_xyz[1] / 2 / rho0
        self.n3rho = n_xyz[2] / 2 / rho0
        self.csn1rho = c0**2 * rho0 * n_xyz[0] / 2
        self.csn2rho = c0**2 * rho0 * n_xyz[1] / 2
        self.csn3rho = c0**2 * rho0 * n_xyz[2] / 2

    def FluxP(
        self,
        dvx: torch.tensor,
        dvy: torch.tensor,
        dvz: torch.tensor,
        dp: torch.tensor,
    ):
        """This method calculates the pressure flux using the given input arrays.

        Args:
            dvx (torch.tensor): The array representing jump values across the faces of neighboring elements in velocity in the x-direction.
            dvy (torch.tensor): The array representing jump values across the faces of neighboring elements in velocity in the y-direction.
            dvz (torch.tensor): The array representing jump values across the faces of neighboring elements in velocity in the z-direction.
            dp (torch.tensor): The array representing jump values across the faces of neighboring elements in pressure.

        Returns:
            torch.tensor: The calculated pressure flux.

        """
        return (
            (self.csn1rho) * dvx
            + (self.csn2rho) * dvy
            + (self.csn3rho) * dvz
            - self.c0 / 2 * dp
        )

    def FluxVx(
        self,
        dvx: torch.tensor,
        dvy: torch.tensor,
        dvz: torch.tensor,
        dp: torch.tensor,
    ):
        """This method calculates the flux of velocity in x-direction using the given input arrays.

        Args:
            dvx (torch.tensor): The array representing jump values across the faces of neighboring elements in velocity in the x-direction.
            dvy (torch.tensor): The array representing jump values across the faces of neighboring elements in velocity in the y-direction.
            dvz (torch.tensor): The array representing jump values across the faces of neighboring elements in velocity in the z-direction.
            dp (torch.tensor): The array representing jump values across the faces of neighboring elements in pressure.

        Returns:
            torch.tensor: The calculated flux of velocity in x-direction.
        """
        return (
            (self.cn1s) * dvx
            + (self.cn1n2) * dvy
            + (self.cn1n3) * dvz
            + (self.n1rho) * dp
        )

    def FluxVy(
        self,
        dvx: torch.tensor,
        dvy: torch.tensor,
        dvz: torch.tensor,
        dp: torch.tensor,
    ):
        """This method calculates the flux of velocity in y-direction using the given input arrays.

        Args:
            dvx (torch.tensor): The array representing jump values across the faces of neighboring elements in velocity in the x-direction.
            dvy (torch.tensor): The array representing jump values across the faces of neighboring elements in velocity in the y-direction.
            dvz (torch.tensor): The array representing jump values across the faces of neighboring elements in velocity in the z-direction.
            dp (torch.tensor): The array representing jump values across the faces of neighboring elements in pressure.

        Returns:
            torch.tensor: The calculated flux of velocity in y-direction.
        """
        return (
            (self.cn1n2) * dvx
            + (self.cn2s) * dvy
            + (self.cn2n3) * dvz
            + (self.n2rho) * dp
        )

    def FluxVz(
        self,
        dvx: torch.tensor,
        dvy: torch.tensor,
        dvz: torch.tensor,
        dp: torch.tensor,
    ):
        """This method calculates the flux of velocity in z-direction using the given input arrays.

        Args:
            dvx (torch.tensor): The array representing jump values across the faces of neighboring elements in velocity in the x-direction.
            dvy (torch.tensor): The array representing jump values across the faces of neighboring elements in velocity in the y-direction.
            dvz (torch.tensor): The array representing jump values across the faces of neighboring elements in velocity in the z-direction.
            dp (torch.tensor): The array representing jump values across the faces of neighboring elements in pressure.

        Returns:
            torch.tensor: The calculated flux of velocity in z-direction.
        """
        return (
            (self.cn1n3) * dvx
            + (self.cn2n3) * dvy
            + (self.cn3s) * dvz
            + (self.n3rho) * dp
        )
