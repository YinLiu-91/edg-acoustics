""" This module provides initial condition functionalities for the edg_acoustics package.

The current version of edg_acoustics.initial_condition provides monopole source initial condition.
"""

from __future__ import annotations
import abc
import torch
import numpy as np  # 保留numpy，因为interp1d需要numpy数组
from scipy.interpolate import interp1d

__all__ = ["InitialCondition", "Monopole_IC"]


class InitialCondition(abc.ABC):
    """Setup initial condition of a DG acoustics simulation for a specific scenario.

    :class:`.InitialCondition` is used to setup initial condition.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def Pinit(self, xyz: torch.Tensor):
        """Setup initial condition for pressure."""

    @abc.abstractmethod
    def VXinit(self, xyz: torch.Tensor):
        """Setup initial condition for velocity in x-direction."""

    @abc.abstractmethod
    def VYinit(self, xyz: torch.Tensor):
        """Setup initial condition for velocity in y-direction."""

    @abc.abstractmethod
    def VZinit(self, xyz: torch.Tensor):
        """Setup initial condition for velocity in z-direction."""


class Monopole_IC(InitialCondition):
    """Setup a monopole source for a specific scenario.

    :class:`.Monopole_IC` is used to setup monopple source initial condition.

    Args:
        xyz (torch.Tensor): see :attr:`edg_acoustics.AcousticsSimulation.xyz`.
        source_xyz (torch.Tensor): an (3,) array containing the physical coordinates of the monopole source.
        halfwidth (float): half-bandwidth of the initial Gaussian pulse.

    Attributes:
        source_xyz (torch.Tensor): an (3,) array containing the physical coordinates of the monopole source.
        halfwidth (float): half-bandwidth of the initial Gaussian pulse.
        device (str): The device on which the tensors are stored.
    """

    def __init__(self, source_xyz, frequency: float):
        if isinstance(source_xyz, np.ndarray):
            self.source_xyz = torch.from_numpy(source_xyz).float()
        else:
            self.source_xyz = source_xyz
        self.halfwidth = Monopole_IC.solve_halfwidth(frequency)
        self.device = source_xyz.device if hasattr(
            source_xyz, 'device') else 'cpu'

    @staticmethod
    def solve_halfwidth(frequency: float):
        """Solve halfwidth of the initial Gaussian pulse, given a frequency, using linear interpolation. Avoids root choosing issue with analytical spectra of Gaussian pulse.

        Args:
            frequency (float): frequency of the monopole source.

        Returns:
            halfwidth (float): halfwidth of the initial Gaussian pulse.
        """
        # Given data points
        x_points = np.array([20, 300, 500, 800, 1000, 2000, 4000, 8000])
        y_points = np.array([0.6, 0.4, 0.33, 0.23, 0.17, 0.09, 0.05, 0.015])

        linear_interp = interp1d(x_points, y_points, kind="linear")

        return linear_interp(frequency)

    def Pinit(self, xyz):
        """Setup initial condition for pressure."""
        if isinstance(xyz, np.ndarray):
            xyz = torch.from_numpy(xyz).float()

        device = xyz.device if hasattr(xyz, 'device') else torch.device('cpu')
        source_xyz = self.source_xyz.to(device) if hasattr(
            self.source_xyz, 'to') else torch.tensor(self.source_xyz, device=device)

        pressure = torch.exp(
            -torch.log(torch.tensor(2.0, device=device))
            * (
                (xyz[0] - source_xyz[0]) ** 2
                + (xyz[1] - source_xyz[1]) ** 2
                + (xyz[2] - source_xyz[2]) ** 2
            )
            / self.halfwidth**2
        )
        return pressure

    def VXinit(self, xyz):
        """Setup initial condition for velocity in x-direction."""
        if isinstance(xyz, np.ndarray):
            xyz = torch.from_numpy(xyz).float()
        device = xyz.device if hasattr(xyz, 'device') else torch.device('cpu')
        return torch.zeros([xyz.shape[1], xyz.shape[2]], device=device)

    def VYinit(self, xyz):
        """Setup initial condition for velocity in y-direction."""
        if isinstance(xyz, np.ndarray):
            xyz = torch.from_numpy(xyz).float()
        device = xyz.device if hasattr(xyz, 'device') else torch.device('cpu')
        return torch.zeros([xyz.shape[1], xyz.shape[2]], device=device)

    def VZinit(self, xyz):
        """Setup initial condition for velocity in z-direction."""
        if isinstance(xyz, np.ndarray):
            xyz = torch.from_numpy(xyz).float()
        device = xyz.device if hasattr(xyz, 'device') else torch.device('cpu')
        return torch.zeros([xyz.shape[1], xyz.shape[2]], device=device)
