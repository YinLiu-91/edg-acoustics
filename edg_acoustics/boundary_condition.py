""" This module provides boundary condition functionalities for the edg_acoustics package.

The edg_acoustics.boundary_condition provide more necessary functionalities 
(based upon :mod:`edg_acoustics.acoustics_simulation`) to setup boundary condition for a specific scenario.

Please note that most of used mesh functions and classes in edg_acoustics are present in the main :mod:`edg_acoustics` namespace 
rather than in :mod:`edg_acoustics.boundary_condition`. 

Todo:
    * For future development, the module can be extended to include other types of boundary conditions.
    * maybe add flux boundary conditions.
"""

from __future__ import annotations
import abc
import torch
import numpy as np  # 保留numpy用于与numpy数组的兼容性

__all__ = ["BoundaryCondition", "AbsorbBC", "FREQ_MAX"]

FREQ_MAX = 2e3  # maximum resolvable frequency
"""float: maximum resolvable frequency."""


class BoundaryCondition(abc.ABC):
    """Abstract base class for boundary conditions."""

    @abc.abstractmethod
    def __init__(self):
        pass

    # can be used for other transmission BC in the future
    @staticmethod
    def init_ADEvariables(BCpara: list[dict], BCnode: list[dict]):
        """Initiate ADE variables, normal velocity, characteristic waves (outgoing and incoming).

        Args:
            BCnode (list[dict]): see :attr:`edg_acoustics.AcousticsSimulation.BCnode`.
            BCpara (list[dict]): see :any:`edg_acoustics.AbsorbBC.BCpara`.

        Returns:
            BCvar (list [dict]): see :any:`edg_acoustics.AbsorbBC.BCvar`.
        """
        BCvar = []
        for index, paras in enumerate(BCpara):
            BCvar.append({"label": paras["label"]})
            # 确定设备
            device = 'cpu'
            for key, value in paras.items():
                if isinstance(value, torch.Tensor):
                    device = value.device
                    break

            # 创建变量
            for key in BCnode[index]["map"]:
                if isinstance(BCnode[index]["map"], np.ndarray):
                    map_shape = BCnode[index]["map"].shape
                    map_tensor = torch.from_numpy(
                        BCnode[index]["map"]).to(device)
                else:
                    map_shape = BCnode[index]["map"].shape
                    map_tensor = BCnode[index]["map"]

            BCvar[index].update(
                {key: torch.zeros(map_shape, device=device)
                 for key in ["vn", "ou", "in"]}
            )

            for polekey in paras:
                if polekey == "RP":
                    rp_shape = paras["RP"].shape if isinstance(
                        paras["RP"], torch.Tensor) else torch.from_numpy(paras["RP"]).shape
                    BCvar[index].update(
                        {
                            key: torch.zeros(
                                [rp_shape[1], map_shape[0]], device=device)
                            for key in ["phi", "PHI"]
                        }
                    )
                elif polekey == "CP":
                    cp_shape = paras["CP"].shape if isinstance(
                        paras["CP"], torch.Tensor) else torch.from_numpy(paras["CP"]).shape
                    BCvar[index].update(
                        {
                            key: torch.zeros(
                                [cp_shape[1], map_shape[0]], device=device)
                            for key in ["kexi1", "kexi2", "KEXI1", "KEXI2"]
                        }
                    )
        return BCvar

    @staticmethod
    def check_BCpara(BCnode: list[dict], BCpara: list[dict], freq_max: float):
        """Check if BCpara is compatible with AcousticsSimulation.BCnode and satisfies physical admissibility condition.

        Given an acoustics simulation data structure with a set of boundary conditions specified in acoustics_simulation.BCnode,
        check if the list of boundary conditions specification and parameters are compatible.
        By compatible we mean that all boundary conditions (keys) in BCpara exist in acoustics_simulation.BCnode, and vice-versa.
        Also, to satisfy the causality and reality conditions, multi-pole model parameters :math:`\\zeta_i` (stored in first row of
        numpy.array BCpara[BC_label] need to be positive.
        To satisfy the passivity condition, the magnitude of the reflection coefficient from the multi-pole model need to be smaller than 1,
        that is, :math:`|R(\\omega)|\\leq 1`, where

        .. math::
            R(\\omega)\\approx{R}_\\infty+\\sum_{k=1}^{S}\\frac{A_k}{\\zeta_k+\\mathrm{i}\\omega}+ \\sum_{l=1}^{T} \\frac{1}{2}\\Big( \\frac{B_l-\\mathrm{i}C_l}{\\alpha_l-\\mathrm{i}\\beta_l+\\mathrm{i}\\omega}+\\frac{B_l+\\mathrm{i}C_l}{\\alpha_l+\\mathrm{i}\\beta_l+\\mathrm{i}\\omega} \\Big)


        Args:
            BCnode (list[dict]): see :attr:`edg_acoustics.AcousticsSimulation.BCnode`.
            BCpara (list[dict]): see :attr:`edg_acoustics.AbsorbBC.BCpara`.

            freq_max (float): maximum resolvable frequency of the simulation. <default>: :attr:`edg_acoustics.boundary_condition.FREQ_MAX`

        Raises:
            AssertionError: If BCpara.[index]['label'] is not present in the acoustics_simulation.BCnode.[index]['label'], an error is raised.
                If a label is present in the acoustics_simulation.BCnode.[index]['label'] but not in BCpara, an error is raised.
                If the labels in BCpara and BCnode are not the same, an error is raised.
            AssertionError: If the reflection coefficient is not smaller than 1, an error is raised.
            AssertionError: If the number of BC types is not the same in the BC_labels and BC_para, an error is raised.
            AssertionError: If the causality and reality conditions are not met, an error is raised.
        """
        # 确定设备
        device = 'cpu'
        for para in BCpara:
            for key, value in para.items():
                if isinstance(value, torch.Tensor):
                    device = value.device
                    break

        omega = torch.arange(1.0, freq_max, device=device)

        assert len(BCpara) == len(BCnode), (
            "[edg_acoustics.BoundaryCondition] The number of BC types must be the same "
            "in the BC_labels and BC_para"
        )
        assert all(
            d1["label"] == d2["label"] for d1, d2 in zip(BCpara, BCnode)
        ), "[edg_acoustics.BoundaryCondition] The labels in BCpara and BCnode must be the same"

        for index, paras in enumerate(BCpara):
            assert paras["label"] == BCnode[index]["label"], (
                "[edg_acoustics.BoundaryCondition] "
                "All BC types must be present in the BCnode "
                "and all labels in the BCnode must have boundary parameters input."
            )
            assert (
                torch.abs(AbsorbBC.compute_Re(omega, paras)) <= 1.0
            ).all(), "[edg_acoustics.BoundaryCondition] All reflection coefficient must be smaller than 1"

            for polekey in paras:
                if polekey == "RP":
                    if isinstance(paras["RP"], np.ndarray):
                        zeta = torch.from_numpy(paras["RP"][1, :]).to(device)
                    else:
                        zeta = paras["RP"][1, :]
                    assert (zeta > 0).all(), (
                        "[edg_acoustics.BoundaryCondition] To satisfy causality and reality conditions, "
                        "all real poles must have positive damping ratio, physical boundary "
                        + str(paras["label"])
                        + " has failed the physical admissbility test"
                    )
                elif polekey == "CP":
                    if isinstance(paras["CP"], np.ndarray):
                        alpha = torch.from_numpy(paras["CP"][2, :]).to(device)
                    else:
                        alpha = paras["CP"][2, :]
                    assert ((alpha > 0)).all(), (
                        "[edg_acoustics.BoundaryCondition] To satisfy causality and reality conditions, "
                        "all complex poles must have positive damping ratio, physical boundary "
                        + str(paras["label"])
                        + " has failed the physical admissbility test"
                    )
            print(
                "boundary parameter with label: "
                + str(paras["label"])
                + " has passed the physical admissbility test"
            )

    @staticmethod
    def compute_Re(omega: torch.Tensor, paras: dict):
        """Computes the reflection coefficient given the passed parameter of the multi-pole model at the frequencies of omega.

        Args:
            omega (torch.Tensor): angular frequency.
            paras (dict): see :attr:`edg_acoustics.boundary_condition.AbsorbBC.BCpara`.

        Returns:
            Re (torch.Tensor): reflection coefficient at the frequencies of omega.
        """
        # 确定设备
        device = omega.device

        Re = torch.ones(omega.shape, device=device)

        for polekey in paras:
            if polekey == "RI":
                if isinstance(paras["RI"], np.ndarray):
                    Re = Re * torch.from_numpy(paras["RI"]).to(device)
                else:
                    Re = Re * paras["RI"]
            elif polekey == "RP":
                if isinstance(paras["RI"], np.ndarray):
                    Re = Re * torch.from_numpy(paras["RI"]).to(device)
                else:
                    Re = Re * paras["RI"]

                if isinstance(paras["RP"], np.ndarray):
                    A = torch.from_numpy(paras["RP"][0, :]).to(device)
                    zeta = torch.from_numpy(paras["RP"][1, :]).to(device)
                else:
                    A = paras["RP"][0, :]
                    zeta = paras["RP"][1, :]

                for j, a in enumerate(A):
                    Re = Re + a / (1j * omega + zeta[j])
            elif polekey == "CP":
                if isinstance(paras["RI"], np.ndarray):
                    Re = Re * torch.from_numpy(paras["RI"]).to(device)
                else:
                    Re = Re * paras["RI"]

                if isinstance(paras["CP"], np.ndarray):
                    B = torch.from_numpy(paras["CP"][0, :]).to(device)
                    C = torch.from_numpy(paras["CP"][1, :]).to(device)
                    alpha = torch.from_numpy(paras["CP"][2, :]).to(device)
                    beta = torch.from_numpy(paras["CP"][3, :]).to(device)
                else:
                    B = paras["CP"][0, :]
                    C = paras["CP"][1, :]
                    alpha = paras["CP"][2, :]
                    beta = paras["CP"][3, :]

                for j, _ in enumerate(B):
                    Re = Re + 0.5 * (
                        (B[j] + 1j * C[j]) /
                        (alpha[j] + 1j * beta[j] + 1j * omega)
                        + (B[j] - 1j * C[j]) /
                        (alpha[j] - 1j * beta[j] + 1j * omega)
                    )
        return Re


class AbsorbBC(BoundaryCondition):
    """Setup absorptive boundary condition of a DG acoustics simulation for a specific scenario.

    :class:`.AbsorbBC` is used to load the boundary condition parameters, and to initiate the ADE variables.

    Args:
        BCnode (list[dict]): see :attr:`edg_acoustics.AcousticsSimulation.BCnode`.
        BCpara (list[dict]): see :attr:`BCpara`.

        freq_max (float): maximum resolvable frequency of the simulation. <default>: :attr:`edg_acoustics.boundary_condition.FREQ_MAX`

    Attributes:
        BCpara (list [dict]): a list of boundary conditon parameters from the multi-pole model. Each element is a dictionary
            with keys (values) ['label'(int),'RI'(float),'RP'(torch.Tensor),'CP'(torch.Tensor)].
            'RI' refers to the limit value of the reflection coefficient as the frequency approaches infinity, i.e., :math:`R_\\infty`.
            'RP' refers to real pole pairs, i.e., :math:`A` (stored in 1st row), :math:`\\zeta` (stored in 2nd row).
            'CP' refers to complex pole pairs, i.e., :math:`B` (stored in 1st row), :math:`C` (stored in 2nd row), :math:`\\alpha` (stored in 3rd row), :math:`\\beta` (stored in 4th row).

            More details about the multi-pole model parameters and boundary condition can be found in reference https://doi.org/10.1121/10.0001128.

            BCpara[:]['label'] must contain the same integer elements as acoustics_simulation.BCnode[:]['label'],
            i.e., all boundary conditions in the simulation must have an associated boundary condition parameters.
        BCvar (list [dict]): a list of ADE variables. Each element corresponds to one type of BC, and is a dictionary
                with potential keys ['label', 'vn', 'ou', 'in', 'phi', 'PHI', 'kexi1', 'kexi2', 'KEXI1', 'KEXI2'].
    """

    def __init__(self, BCnode: list[dict], BCpara: list[dict], freq_max: float = FREQ_MAX):
        BoundaryCondition.check_BCpara(BCnode, BCpara, freq_max)
        self.BCpara = BCpara
        self.BCvar = BoundaryCondition.init_ADEvariables(self.BCpara, BCnode)

    # -----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
