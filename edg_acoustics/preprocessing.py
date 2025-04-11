"""This module provides preprocessing functionalities for the edg_acoustics package.
"""

from __future__ import annotations
import abc
import torch
import numpy as np  # 保留numpy用于与numpy数组的兼容性


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
        n_xyz (torch.Tensor or numpy.ndarray): The array representing the normal vector of the face.

    Attributes:
        rho0 (float): see :attr:`edg_acoustics.AcousticsSimulation.rho0`.
        c0 (float): see :attr:`edg_acoustics.AcousticsSimulation.c0`.
        n_xyz (torch.Tensor): see :attr:`edg_acoustics.AcousticsSimulation.n_xyz`.
        cn1s (torch.Tensor): precalculated constant for the calculation of the flux of velocity in x-direction.
        cn2s (torch.Tensor): precalculated constant for the calculation of the flux of velocity in y-direction.
        cn3s (torch.Tensor): precalculated constant for the calculation of the flux of velocity in z-direction.
        cn1n2 (torch.Tensor): precalculated constant for the calculation of the flux of velocity.
        cn1n3 (torch.Tensor): precalculated constant for the calculation of the flux of velocity.
        cn2n3 (torch.Tensor): precalculated constant for the calculation of the flux of velocity.
        n1rho (torch.Tensor): precalculated constant for the calculation of the flux of velocity in x-direction.
        n2rho (torch.Tensor): precalculated constant for the calculation of the flux of velocity in y-direction.
        n3rho (torch.Tensor): precalculated constant for the calculation of the flux of velocity in z-direction.
        csn1rho (torch.Tensor): precalculated constant for the calculation of the pressure flux.
        csn2rho (torch.Tensor): precalculated constant for the calculation of the pressure flux.
        csn3rho (torch.Tensor): precalculated constant for the calculation of the pressure flux.
        device (str): The device on which the tensors are stored.
    """

    def __init__(self, rho0: float, c0: float, n_xyz):
        self.rho0 = rho0
        self.c0 = c0

        # 如果输入是numpy数组，转换为PyTorch张量
        if isinstance(n_xyz, np.ndarray):
            self.n_xyz = torch.from_numpy(n_xyz).float()
        else:
            self.n_xyz = n_xyz

        self.device = self.n_xyz.device if hasattr(
            self.n_xyz, 'device') else torch.device('cpu')

        # 将所有计算放在同一设备上
        self.cn1s = -c0 * self.n_xyz[0] ** 2 / 2
        self.cn2s = -c0 * self.n_xyz[1] ** 2 / 2
        self.cn3s = -c0 * self.n_xyz[2] ** 2 / 2
        self.cn1n2 = -c0 * self.n_xyz[0] * self.n_xyz[1] / 2
        self.cn1n3 = -c0 * self.n_xyz[0] * self.n_xyz[2] / 2
        self.cn2n3 = -c0 * self.n_xyz[1] * self.n_xyz[2] / 2
        self.n1rho = self.n_xyz[0] / 2 / rho0
        self.n2rho = self.n_xyz[1] / 2 / rho0
        self.n3rho = self.n_xyz[2] / 2 / rho0
        self.csn1rho = c0**2 * rho0 * self.n_xyz[0] / 2
        self.csn2rho = c0**2 * rho0 * self.n_xyz[1] / 2
        self.csn3rho = c0**2 * rho0 * self.n_xyz[2] / 2

    def FluxP(self, dvx, dvy, dvz, dp):
        """This method calculates the pressure flux using the given input arrays.

        Args:
            dvx (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the x-direction.
            dvy (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the y-direction.
            dvz (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the z-direction.
            dp (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in pressure.

        Returns:
            torch.Tensor: The calculated pressure flux.

        """
        # 确保输入是PyTorch张量
        if isinstance(dvx, np.ndarray):
            dvx = torch.from_numpy(dvx).float().to(self.device)
        if isinstance(dvy, np.ndarray):
            dvy = torch.from_numpy(dvy).float().to(self.device)
        if isinstance(dvz, np.ndarray):
            dvz = torch.from_numpy(dvz).float().to(self.device)
        if isinstance(dp, np.ndarray):
            dp = torch.from_numpy(dp).float().to(self.device)

        return self.csn1rho * dvx + self.csn2rho * dvy + self.csn3rho * dvz - self.c0 / 2 * dp

    def FluxVx(self, dvx, dvy, dvz, dp):
        """This method calculates the flux of velocity in x-direction using the given input arrays.

        Args:
            dvx (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the x-direction.
            dvy (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the y-direction.
            dvz (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the z-direction.
            dp (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in pressure.

        Returns:
            torch.Tensor: The calculated flux of velocity in x-direction.
        """
        # 确保输入是PyTorch张量
        if isinstance(dvx, np.ndarray):
            dvx = torch.from_numpy(dvx).float().to(self.device)
        if isinstance(dvy, np.ndarray):
            dvy = torch.from_numpy(dvy).float().to(self.device)
        if isinstance(dvz, np.ndarray):
            dvz = torch.from_numpy(dvz).float().to(self.device)
        if isinstance(dp, np.ndarray):
            dp = torch.from_numpy(dp).float().to(self.device)

        return self.cn1s * dvx + self.cn1n2 * dvy + self.cn1n3 * dvz + self.n1rho * dp

    def FluxVy(self, dvx, dvy, dvz, dp):
        """This method calculates the flux of velocity in y-direction using the given input arrays.

        Args:
            dvx (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the x-direction.
            dvy (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the y-direction.
            dvz (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the z-direction.
            dp (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in pressure.

        Returns:
            torch.Tensor: The calculated flux of velocity in y-direction.
        """
        # 确保输入是PyTorch张量
        if isinstance(dvx, np.ndarray):
            dvx = torch.from_numpy(dvx).float().to(self.device)
        if isinstance(dvy, np.ndarray):
            dvy = torch.from_numpy(dvy).float().to(self.device)
        if isinstance(dvz, np.ndarray):
            dvz = torch.from_numpy(dvz).float().to(self.device)
        if isinstance(dp, np.ndarray):
            dp = torch.from_numpy(dp).float().to(self.device)

        return self.cn1n2 * dvx + self.cn2s * dvy + self.cn2n3 * dvz + self.n2rho * dp

    def FluxVz(self, dvx, dvy, dvz, dp):
        """This method calculates the flux of velocity in z-direction using the given input arrays.

        Args:
            dvx (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the x-direction.
            dvy (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the y-direction.
            dvz (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in velocity in the z-direction.
            dp (torch.Tensor or numpy.ndarray): The array representing jump values across the faces of neighboring elements in pressure.

        Returns:
            torch.Tensor: The calculated flux of velocity in z-direction.
        """
        # 确保输入是PyTorch张量
        if isinstance(dvx, np.ndarray):
            dvx = torch.from_numpy(dvx).float().to(self.device)
        if isinstance(dvy, np.ndarray):
            dvy = torch.from_numpy(dvy).float().to(self.device)
        if isinstance(dvz, np.ndarray):
            dvz = torch.from_numpy(dvz).float().to(self.device)
        if isinstance(dp, np.ndarray):
            dp = torch.from_numpy(dp).float().to(self.device)

        return self.cn1n3 * dvx + self.cn2n3 * dvy + self.cn3s * dvz + self.n3rho * dp
