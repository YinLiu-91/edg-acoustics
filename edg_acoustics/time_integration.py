""" This module contains the abstract base class for time integrators and the implementation of the Taylor-series time integration scheme.

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
import numpy as np  # 保留numpy用于与numpy数组的兼容性

__all__ = ["TimeIntegrator", "TSI_TI", "CFL_Default", "RungeKutta"]

CFL_Default = 0.5
"""float: Default value of the CFL number for time integration schemes."""


class TimeIntegrator(abc.ABC):
    """Base class for time integrators.

    Args:
        L_operator (typing.Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list]]): the function in AcousticSimulation that enables the computation of Lq, given q = [P, Vx, Vy, Vz]
        dtscale (float): the time step scale based on the mesh size measure
        CFL (float): the CFL number

    Attributes:
        L_operator (typing.Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list]]): the function in AcousticSimulation that enables the computation of Lq, given q = [P, Vx, Vy, Vz]
        CFL (float): the CFL number
        dt (float): the time step size
    """

    def __init__(
        self,
        L_operator: typing.Callable[
            [numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list]
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
        L_operator (typing.Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list]]): the function in AcousticSimulation that enables the computation of Lq, given q = [P, Vx, Vy, Vz]
        dtscale (float): the time step scale based on the mesh size measure
        CFL (float): the CFL number
        Nt (int): the order of the time integration scheme.

    Attributes:
        L_operator (typing.Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list]]): the function in AcousticSimulation that enables the computation of Lq, given q = [P, Vx, Vy, Vz]
        CFL (float): the CFL number
        dt (float): the time step size
        Nt (int): the order of the time integration scheme.
    """

    def __init__(
        self,
        L_operator: typing.Callable[
            [numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list]
        ],
        dtscale: float,
        CFL: float = CFL_Default,
        **kwargs,
    ):
        super().__init__(L_operator, dtscale, CFL)
        self.Nt = kwargs["Nt"]
        print("TSI_TI initialized.")

    def step_dt(
        self,
        P: numpy.ndarray,
        Vx: numpy.ndarray,
        Vy: numpy.ndarray,
        Vz: numpy.ndarray,
        BC: edg_acoustics.AbsorbBC,
    ):
        """Takes the pressure, velocity at the time T and evolves/updates them to time T + dt.

        Args:
            P (numpy.ndarray): the pressure at the time T, will be updated to the pressure at the time T + dt
            Vx (numpy.ndarray): the x-component of the velocity at the time T, will be updated to the x-component of the velocity at the time T + dt
            Vy (numpy.ndarray): the y-component of the velocity at the time T, will be updated to the y-component of the velocity at the time T + dt
            Vz (numpy.ndarray): the z-component of the velocity at the time T, will be updated to the z-component of the velocity at the time T + dt
            BC (edg_acoustics.AbsorbBC): the boundary condition object
        """
        # Takes the pressure, velocity, and BCvar at the time T and evolves it to time T + dt
        # BC(edg_acoustics.BoundaryCondition) object neesds to be passed as a reference since we need to access the BCpara attribute
        #   P0 contains the (high-order) derivative values
        #   P := P(T) and P(T + dt)
        # the same for all the other variables
        ##########################
        P0 = P.copy()
        Vx0 = Vx.copy()
        Vy0 = Vy.copy()
        Vz0 = Vz.copy()

        for index, paras in enumerate(BC.BCpara):
            for polekey in paras:
                if polekey == "RP":
                    BC.BCvar[index]["phi"] = BC.BCvar[index]["PHI"].copy()
                elif polekey == "CP":
                    BC.BCvar[index]["kexi1"] = BC.BCvar[index]["KEXI1"].copy()
                    BC.BCvar[index]["kexi2"] = BC.BCvar[index]["KEXI2"].copy()

        ##########################
        for Tind in range(1, self.Nt + 1):
            # Compute L (L^{Tind-1} q)
            P0, Vx0, Vy0, Vz0, BC.BCvar = self.L_operator(
                P0, Vx0, Vy0, Vz0, BC.BCvar)

            # Add the Taylor term \frac{dt^{Tind}}{Tind!}L^{Tind}q
            Vx += self.dt**Tind / math.factorial(Tind) * Vx0
            Vy += self.dt**Tind / math.factorial(Tind) * Vy0
            Vz += self.dt**Tind / math.factorial(Tind) * Vz0
            P += self.dt**Tind / math.factorial(Tind) * P0

            for index, paras in enumerate(BC.BCpara):
                for polekey in paras:
                    if polekey == "RP":
                        BC.BCvar[index]["PHI"] += (
                            self.dt**Tind /
                            math.factorial(Tind) * BC.BCvar[index]["phi"]
                        )
                    elif polekey == "CP":
                        BC.BCvar[index]["KEXI1"] += (
                            self.dt**Tind /
                            math.factorial(Tind) * BC.BCvar[index]["kexi1"]
                        )
                        BC.BCvar[index]["KEXI2"] += (
                            self.dt**Tind /
                            math.factorial(Tind) * BC.BCvar[index]["kexi2"]
                        )


class RungeKutta(TimeIntegrator):
    """Setup time integration for a DG acoustics simulation for a specific scenario.

    :class:`.RungeKutta` is used to setup time integration scheme to solve the DG time-dependent problem.

    Args:
        dt (float): time step of simulation.
        Nt (int): explicit Runge-Kutta time integration scheme. <default>: :py:const:`edg_acoustics.time_integration.NT_DEFAULT`
        CFL (float): CFL number of the simulation. <default>: :py:const:`edg_acoustics.time_integration.CFL_DEFAULT`

    Attributes:
        dt (float): see `dt`.
        Nt (int): see `Nt`.
        CFL (float): see `CFL`.
        A (list[float]): parameter for Runge-Kutta scheme, specific to the time integration order.
        B (list[float]): parameter for Runge-Kutta scheme, specific to the time integration order.
        C (list[float]): parameter for Runge-Kutta scheme, specific to the time integration order.
        sim (edg_acoustics.AcousticsSimulation): the acoustics simulation object to be integrated in time.
    """

    def __init__(self, dt: float, Nt: int = 4, CFL: float = 0.75):
        self.dt = dt
        self.Nt = Nt
        self.CFL = CFL
        self.A = []
        self.B = []
        self.C = []

        if self.Nt == 1:
            # Forward Euler
            self.A = [0.0]
            self.B = [1.0]
            self.C = [0.0]

        elif self.Nt == 2:
            # Midpoint scheme
            self.A = [0.0, 0.5]
            self.B = [0.0, 1.0]
            self.C = [0.0, 0.5]

        elif self.Nt == 3:
            # Kutta's third-order scheme
            self.A = [0.0, 0.5, 1.0]
            self.B = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]
            self.C = [0.0, 0.5, 1.0]

        elif self.Nt == 4:
            # classic fourth-order Runge-Kutta
            self.A = [0.0, 0.5, 0.5, 1]
            self.B = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]
            self.C = [0.0, 0.5, 0.5, 1.0]

        elif self.Nt == 5:
            # Butcher's fifth-order scheme
            self.A = [0.0, 0.25, 0.25, 0.5, 0.75, 1.0]
            self.B = [7.0 / 90.0, 0.0, 16.0 / 45.0,
                      1.0 / 15.0, 16.0 / 45.0, 7.0 / 90.0]
            self.C = [0.0, 0.25, 0.25, 0.5, 0.75, 1.0]

        else:
            print("Runge-Kutta scheme for order " +
                  self.Nt + " is not implemented.")

    def step_dt(self, P, Vx, Vy, Vz, BC):
        """Step one step forward in time.

        Args:
            P (torch.Tensor or numpy.ndarray): pressure.
            Vx (torch.Tensor or numpy.ndarray): velocity in x-direction.
            Vy (torch.Tensor or numpy.ndarray): velocity in y-direction.
            Vz (torch.Tensor or numpy.ndarray): velocity in z-direction.
            BC (edg_acoustics.BoundaryCondition): boundary condition.
        """
        # 转换输入为PyTorch张量
        if isinstance(P, np.ndarray):
            P_tensor = torch.from_numpy(P).float()
        else:
            P_tensor = P

        if isinstance(Vx, np.ndarray):
            Vx_tensor = torch.from_numpy(Vx).float()
        else:
            Vx_tensor = Vx

        if isinstance(Vy, np.ndarray):
            Vy_tensor = torch.from_numpy(Vy).float()
        else:
            Vy_tensor = Vy

        if isinstance(Vz, np.ndarray):
            Vz_tensor = torch.from_numpy(Vz).float()
        else:
            Vz_tensor = Vz

        # 确定设备
        device = P_tensor.device if hasattr(
            P_tensor, 'device') else torch.device('cpu')

        # 保存原始形状
        P_shape = P_tensor.shape
        Vx_shape = Vx_tensor.shape
        Vy_shape = Vy_tensor.shape
        Vz_shape = Vz_tensor.shape

        # 复制原始数据
        Po = P_tensor.clone()
        Vxo = Vx_tensor.clone()
        Vyo = Vy_tensor.clone()
        Vzo = Vz_tensor.clone()

        # 创建临时存储
        kP = torch.zeros_like(P_tensor)
        kVx = torch.zeros_like(Vx_tensor)
        kVy = torch.zeros_like(Vy_tensor)
        kVz = torch.zeros_like(Vz_tensor)

        # 将BCvar复制一份作为临时变量
        from copy import deepcopy
        BCvar_tmp = deepcopy(BC.BCvar)

        # 开始Runge-Kutta迭代
        for i in range(self.Nt):
            # 如果这不是第一步，需要计算中间状态
            if i > 0:
                # 计算中间状态
                P_tensor = Po + self.dt * self.A[i] * kP
                Vx_tensor = Vxo + self.dt * self.A[i] * kVx
                Vy_tensor = Vyo + self.dt * self.A[i] * kVy
                Vz_tensor = Vzo + self.dt * self.A[i] * kVz

                # 更新边界条件变量
                for index, paras in enumerate(BC.BCpara):
                    for polekey in paras:
                        if polekey == "RP":
                            for j in range(paras["RP"].shape[1]):
                                BCvar_tmp[index]["PHI"][j] = BC.BCvar[index]["PHI"][j] + \
                                    self.dt * self.A[i] * \
                                    BC.BCvar[index]["phi"][j]
                        elif polekey == "CP":
                            for j in range(paras["CP"].shape[1]):
                                BCvar_tmp[index]["KEXI1"][j] = BC.BCvar[index]["KEXI1"][j] + \
                                    self.dt * self.A[i] * \
                                    BC.BCvar[index]["kexi1"][j]
                                BCvar_tmp[index]["KEXI2"][j] = BC.BCvar[index]["KEXI2"][j] + \
                                    self.dt * self.A[i] * \
                                    BC.BCvar[index]["kexi2"][j]

            # 保存BC.BCvar的phi, kexi1, kexi2
            for index, paras in enumerate(BC.BCpara):
                for polekey in paras:
                    if polekey == "RP":
                        for j in range(paras["RP"].shape[1]):
                            BCvar_tmp[index]["phi"][j] = BC.BCvar[index]["phi"][j]
                    elif polekey == "CP":
                        for j in range(paras["CP"].shape[1]):
                            BCvar_tmp[index]["kexi1"][j] = BC.BCvar[index]["kexi1"][j]
                            BCvar_tmp[index]["kexi2"][j] = BC.BCvar[index]["kexi2"][j]

            # 计算右侧项
            RHS_P, RHS_Vx, RHS_Vy, RHS_Vz, BCtmp = BC.sim.RHS_operator(
                P_tensor, Vx_tensor, Vy_tensor, Vz_tensor, BCvar_tmp)

            # 更新k值
            kP = kP + self.B[i] * RHS_P
            kVx = kVx + self.B[i] * RHS_Vx
            kVy = kVy + self.B[i] * RHS_Vy
            kVz = kVz + self.B[i] * RHS_Vz

            # 如果我们在最后一步之前，需要保存流体和边界更新变量，为下一个子步准备
            if i < self.Nt - 1:
                for index, paras in enumerate(BC.BCpara):
                    for polekey in paras:
                        if polekey == "RP":
                            for j in range(paras["RP"].shape[1]):
                                BC.BCvar[index]["phi"][j] = BCtmp[index]["phi"][j]
                        elif polekey == "CP":
                            for j in range(paras["CP"].shape[1]):
                                BC.BCvar[index]["kexi1"][j] = BCtmp[index]["kexi1"][j]
                                BC.BCvar[index]["kexi2"][j] = BCtmp[index]["kexi2"][j]

        # 更新最终值
        P_tensor = Po + self.dt * kP
        Vx_tensor = Vxo + self.dt * kVx
        Vy_tensor = Vyo + self.dt * kVy
        Vz_tensor = Vzo + self.dt * kVz

        # 确保最终值始终是最新的
        for index, paras in enumerate(BC.BCpara):
            for polekey in paras:
                if polekey == "RP":
                    for j in range(paras["RP"].shape[1]):
                        BC.BCvar[index]["phi"][j] = BCvar_tmp[index]["phi"][j]
                elif polekey == "CP":
                    for j in range(paras["CP"].shape[1]):
                        BC.BCvar[index]["kexi1"][j] = BCvar_tmp[index]["kexi1"][j]
                        BC.BCvar[index]["kexi2"][j] = BCvar_tmp[index]["kexi2"][j]

        # 如果输入是numpy数组，将结果转换回numpy
        if isinstance(P, np.ndarray):
            P[:] = P_tensor.cpu().numpy()
        else:
            P.copy_(P_tensor)

        if isinstance(Vx, np.ndarray):
            Vx[:] = Vx_tensor.cpu().numpy()
        else:
            Vx.copy_(Vx_tensor)

        if isinstance(Vy, np.ndarray):
            Vy[:] = Vy_tensor.cpu().numpy()
        else:
            Vy.copy_(Vy_tensor)

        if isinstance(Vz, np.ndarray):
            Vz[:] = Vz_tensor.cpu().numpy()
        else:
            Vz.copy_(Vz_tensor)
