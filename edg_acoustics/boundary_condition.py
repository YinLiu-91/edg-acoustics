"""This module provides boundary condition functionalities for the edg_acoustics package.

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
import numpy
import torch
import edg_acoustics.device_ini as device_ini

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
            BCvar[index].update(
                {
                    key: torch.zeros(
                        BCnode[index]["map"].shape,
                        device=device_ini.device,
                        dtype=device_ini.dtype,
                    )
                    for key in ["vn", "ou", "in"]
                }
            )
            for polekey in paras:
                if polekey == "RP":
                    BCvar[index].update(
                        {
                            key: torch.zeros(
                                [paras["RP"].shape[1], BCnode[index]["map"].shape[0]],
                                device=device_ini.device,
                                dtype=device_ini.dtype,
                            )
                            for key in ["phi", "PHI"]
                        }
                    )
                elif polekey == "CP":
                    BCvar[index].update(
                        {
                            key: torch.zeros(
                                [paras["CP"].shape[1], BCnode[index]["map"].shape[0]],
                                device=device_ini.device,
                                dtype=device_ini.dtype,
                            )
                            for key in ["kexi1", "kexi2", "KEXI1", "KEXI2"]
                        }
                    )
                elif polekey == "normal_velocity":
                    BCvar[index]["normal_velocity"] = torch.zeros(
                        BCnode[index]["map"].shape,
                        device=device_ini.device,
                        dtype=device_ini.dtype,
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
        omega = torch.arange(1.0, freq_max, dtype=device_ini.dtype)

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
                    zeta = paras["RP"][1, :]
                    assert (zeta > 0).all(), (
                        "[edg_acoustics.BoundaryCondition] To satisfy causality and reality conditions, "
                        "all real poles must have positive damping ratio, physical boundary "
                        + str(paras["label"])
                        + " has failed the physical admissbility test"
                    )
                elif polekey == "CP":
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
    def compute_Re(omega: torch.tensor, paras: dict):
        """Computes the reflection coefficient given the passed parameter of the multi-pole model at the frequencies of omega.

        Args:
            omega (torch.tensor): angular frequency.
            paras (dict): see :attr:`edg_acoustics.boundary_condition.AbsorbBC.BCpara`.

        Returns:
            Re (torch.tensor): reflection coefficient at the frequencies of omega.
        """
        Re = torch.ones(omega.shape, device=omega.device, dtype=omega.dtype)
        if "RI" in paras:
            Re = Re * paras["RI"]

        for polekey in paras:
            if polekey == "RP":
                A = paras["RP"][0, :]
                zeta = paras["RP"][1, :]
                for j, a in enumerate(A):
                    Re = Re + a / (1j * omega + zeta[j])
            elif polekey == "CP":
                B = paras["CP"][0, :]
                C = paras["CP"][1, :]
                alpha = paras["CP"][2, :]
                beta = paras["CP"][3, :]
                for j, _ in enumerate(B):
                    Re = Re + 0.5 * (
                        (B[j] + 1j * C[j]) / (alpha[j] + 1j * beta[j] + 1j * omega)
                        + (B[j] - 1j * C[j]) / (alpha[j] - 1j * beta[j] + 1j * omega)
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
            with keys (values) ['label'(int),'RI'(float),'RP'(torch.tensor),'CP'(torch.tensor)].
            'RI' refers to the limit value of the reflection coefficient as the frequency approaches infinity, i.e., :math:`R_\\infty`.
            'RP' refers to real pole pairs, i.e., :math:`A` (stored in 1st row), :math:`\\zeta` (stored in 2nd row).
            'CP' refers to complex pole pairs, i.e., :math:`B` (stored in 1st row), :math:`C` (stored in 2nd row), :math:`\\alpha` (stored in 3rd row), :math:`\\beta` (stored in 4th row).

            More details about the multi-pole model parameters and boundary condition can be found in reference https://doi.org/10.1121/10.0001128.

            BCpara[:]['label'] must contain the same integer elements as acoustics_simulation.BCnode[:]['label'],
            i.e., all boundary conditions in the simulation must have an associated boundary condition parameters.
        BCvar (list [dict]): a list of ADE variables. Each element corresponds to one type of BC, and is a dictionary
                with potential keys ['label', 'vn', 'ou', 'in', 'phi', 'PHI', 'kexi1', 'kexi2', 'KEXI1', 'KEXI2'].
    """

    def __init__(
        self, BCnode: list[dict], BCpara: list[dict], freq_max: float = FREQ_MAX
    ):
        BoundaryCondition.check_BCpara(BCnode, BCpara, freq_max)
        self.BCpara = BCpara
        self.BCvar = BoundaryCondition.init_ADEvariables(self.BCpara, BCnode)
        self.has_prescribed_normal_velocity = any(
            "normal_velocity" in paras for paras in self.BCpara
        )

    @staticmethod
    def _poly_derivative(coefficients: list[float]) -> list[float]:
        return [
            order * coefficients[order]
            for order in range(1, len(coefficients))
        ] or [0.0]

    @staticmethod
    def _poly_multiply_by_x(coefficients: list[float]) -> list[float]:
        return [0.0] + coefficients

    @staticmethod
    def _poly_add(*polynomials: list[float]) -> list[float]:
        size = max(len(poly) for poly in polynomials)
        out = [0.0] * size
        for poly in polynomials:
            for index, value in enumerate(poly):
                out[index] += value
        while len(out) > 1 and out[-1] == 0.0:
            out.pop()
        return out

    @staticmethod
    def _poly_scale(coefficients: list[float], scale: float) -> list[float]:
        return [scale * value for value in coefficients]

    @staticmethod
    def _poly_eval(coefficients: list[float], x: torch.tensor) -> torch.tensor:
        out = torch.zeros_like(x)
        for value in reversed(coefficients):
            out = out * x + value
        return out

    @staticmethod
    def gaussian_modulated_sine_normal_velocity(
        time: float | torch.tensor,
        derivative_order: int,
        *,
        amplitude: float = 1.0,
        frequency: float = 1000.0,
        delay: float = 0.0,
        sigma: float = 1.0,
        phase: float = 0.0,
        baseline: float = 0.0,
    ) -> float | torch.tensor:
        """Evaluate the n-th time derivative of a Gaussian-modulated sine."""

        omega = 2.0 * numpy.pi * frequency
        angular_sigma = omega * sigma
        P = [1.0]
        Q = [0.0]
        for _ in range(int(derivative_order)):
            P, Q = (
                AbsorbBC._poly_add(
                    AbsorbBC._poly_derivative(P),
                    AbsorbBC._poly_scale(AbsorbBC._poly_multiply_by_x(P), -1.0),
                    AbsorbBC._poly_scale(Q, -angular_sigma),
                ),
                AbsorbBC._poly_add(
                    AbsorbBC._poly_derivative(Q),
                    AbsorbBC._poly_scale(AbsorbBC._poly_multiply_by_x(Q), -1.0),
                    AbsorbBC._poly_scale(P, angular_sigma),
                ),
            )

        if torch.is_tensor(time):
            u = (time - delay) / sigma
            envelope = torch.exp(-0.5 * u * u)
            theta = omega * time + phase
            value = amplitude * envelope * (
                AbsorbBC._poly_eval(P, u) * torch.sin(theta)
                + AbsorbBC._poly_eval(Q, u) * torch.cos(theta)
            ) / (sigma ** int(derivative_order))
            if derivative_order == 0 and baseline:
                value = value + baseline
            return value

        u_float = (float(time) - delay) / sigma
        envelope_float = numpy.exp(-0.5 * u_float * u_float)
        theta_float = omega * float(time) + phase
        p_value = sum(value * (u_float**order) for order, value in enumerate(P))
        q_value = sum(value * (u_float**order) for order, value in enumerate(Q))
        value_float = amplitude * envelope_float * (
            p_value * numpy.sin(theta_float)
            + q_value * numpy.cos(theta_float)
        ) / (sigma ** int(derivative_order))
        if derivative_order == 0:
            value_float += baseline
        return float(value_float)

    @staticmethod
    def evaluate_normal_velocity(
        config: dict,
        time: float | torch.tensor,
        derivative_order: int = 0,
    ) -> float | torch.tensor:
        kind = config.get("kind", "gaussian_modulated_sine")
        if kind != "gaussian_modulated_sine":
            raise ValueError(f"Unsupported normal_velocity kind: {kind}")
        return AbsorbBC.gaussian_modulated_sine_normal_velocity(
            time,
            derivative_order,
            amplitude=float(config.get("amplitude", 1.0)),
            frequency=float(config.get("frequency", 1000.0)),
            delay=float(config.get("delay", 0.0)),
            sigma=float(config.get("sigma", 1.0)),
            phase=float(config.get("phase", 0.0)),
            baseline=float(config.get("baseline", 0.0)),
        )

    def prepare_prescribed_normal_velocity(
        self,
        time: float | torch.tensor,
        derivative_order: int = 0,
    ) -> None:
        """Fill prescribed normal-velocity values for the current Taylor stage."""

        for index, paras in enumerate(self.BCpara):
            config = paras.get("normal_velocity")
            if config is None:
                continue
            value = self.evaluate_normal_velocity(config, time, derivative_order)
            target = self.BCvar[index]["normal_velocity"]
            if torch.is_tensor(value):
                target.copy_(value.to(device=target.device, dtype=target.dtype).expand_as(target))
            else:
                target.fill_(float(value))

    # -----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
