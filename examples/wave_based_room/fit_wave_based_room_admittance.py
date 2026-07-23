#!/usr/bin/env python3
"""Convert COMSOL PFF admittance functions to EDG AbsorbBC material files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy
import scipy.io


CASE_DIR = Path(__file__).resolve().parent
RHO0 = 1.2
C0 = 343.0
Z0 = RHO0 * C0
FREQ_MAX_PASSIVITY = 2100.0


@dataclass(frozen=True)
class ComsolPff:
    name: str
    table: str
    y_inf: float
    real_residues: tuple[float, ...]
    real_poles: tuple[float, ...]
    complex_residues: tuple[complex, ...]
    complex_poles: tuple[complex, ...]


MATERIALS = (
    ComsolPff(
        "carpet",
        "wave_based_room_admittance_carpet.zh_CN.txt",
        0.0015195025193809625,
        (0.026677943572679146, 5.050826030455752e-4, 7.784291479796588e-6),
        (-1805.0046654615276, -598.5399836809326, -168.52886314704224),
        (13.53190522508639 + 18.4447826833512j,),
        (-2956.214378807806 + 7143.975905933162j,),
    ),
    ComsolPff(
        "ceiling",
        "wave_based_room_admittance_ceiling.zh_CN.txt",
        8.223331412938798e-4,
        (-0.04072175664357695,),
        (-549.4241379818307,),
        (12.347753112266295 + 7.269844789332405j,),
        (-1239.879314880262 + 3550.9914423437467j,),
    ),
    ComsolPff(
        "sofa",
        "wave_based_room_admittance_sofa.zh_CN.txt",
        9.203446659891514e-4,
        (),
        (),
        (
            6.423745528357205 + 10.759610790231022j,
            7.314563248938069 + 3.4438434969335088j,
        ),
        (
            -1646.0406584140023 + 6128.203722753253j,
            -1141.4749337559003 + 1816.6940256539144j,
        ),
    ),
    ComsolPff(
        "wall",
        "wave_based_room_admittance_wall.zh_CN.txt",
        6.67440521330136e-4,
        (-0.09937417551179016,),
        (-281.48818345618673,),
        (-1.7205375642129488 + 2.846863638431642j,),
        (-3222.506530349331 + 9794.831933382658j,),
    ),
)


def read_frequency_table(path: Path) -> numpy.ndarray:
    data = numpy.genfromtxt(path, comments="%", dtype=float)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Could not read frequency/admittance table: {path}")
    freq = data[:, 0]
    if not numpy.all(numpy.diff(freq) > 0.0):
        raise ValueError(f"Frequency grid is not strictly increasing: {path}")
    return freq


def pff_poles_residues(material: ComsolPff) -> tuple[list[complex], list[complex]]:
    poles = [complex(value) for value in material.real_poles]
    residues = [complex(value) for value in material.real_residues]
    for residue, pole in zip(material.complex_residues, material.complex_poles):
        poles.extend([pole, pole.conjugate()])
        residues.extend([0.5 * residue, 0.5 * residue.conjugate()])
    return poles, residues


def admittance_rational(material: ComsolPff) -> tuple[numpy.poly1d, numpy.poly1d]:
    poles, residues = pff_poles_residues(material)
    denominator = numpy.poly1d(numpy.poly(poles))
    numerator = numpy.poly1d(material.y_inf * denominator)
    for pole, residue in zip(poles, residues):
        quotient, remainder = numpy.polydiv(denominator, numpy.poly1d([1.0, -pole]))
        scale = max(1.0, float(numpy.max(numpy.abs(denominator.coeffs))))
        if numpy.max(numpy.abs(remainder.coeffs)) > 1.0e-12 * scale:
            raise RuntimeError("Internal polynomial division failed")
        numerator += numpy.poly1d(residue * quotient)
    return numerator, denominator


def reflection_partial_fraction(material: ComsolPff) -> dict[str, numpy.ndarray | float]:
    numerator_y, denominator_y = admittance_rational(material)
    numerator_r = denominator_y - Z0 * numerator_y
    denominator_r = denominator_y + Z0 * numerator_y
    ri = float(numpy.real(numerator_r.coeffs[0] / denominator_r.coeffs[0]))

    poles_z = numpy.roots(denominator_r)
    derivative = numpy.polyder(denominator_r)
    residues_z = numpy.asarray(
        [numerator_r(pole) / derivative(pole) for pole in poles_z],
        dtype=complex,
    )
    poles_s = 2.0 * numpy.pi * poles_z
    residues_s = 2.0 * numpy.pi * residues_z

    real_items: list[tuple[float, float]] = []
    complex_items: list[tuple[float, float, float, float]] = []
    for pole, residue in sorted(zip(poles_s, residues_s), key=lambda item: (item[0].real, item[0].imag)):
        if abs(pole.imag) < 1.0e-6:
            real_items.append((float(numpy.real(residue)), float(-numpy.real(pole))))
        elif pole.imag > 0.0:
            complex_items.append(
                (
                    float(2.0 * numpy.real(residue)),
                    float(-2.0 * numpy.imag(residue)),
                    float(-numpy.real(pole)),
                    float(numpy.imag(pole)),
                )
            )

    return {
        "RI": ri,
        "AS": numpy.asarray([item[0] for item in real_items], dtype=float),
        "lambdaS": numpy.asarray([item[1] for item in real_items], dtype=float),
        "BS": numpy.asarray([item[0] for item in complex_items], dtype=float),
        "CS": numpy.asarray([item[1] for item in complex_items], dtype=float),
        "alphaS": numpy.asarray([item[2] for item in complex_items], dtype=float),
        "betaS": numpy.asarray([item[3] for item in complex_items], dtype=float),
    }


def eval_comsol_admittance(freq: numpy.ndarray, material: ComsolPff) -> numpy.ndarray:
    s = 1j * freq
    value = material.y_inf * numpy.ones_like(s, dtype=complex)
    for residue, pole in zip(material.real_residues, material.real_poles):
        value += residue / (s - pole)
    for residue, pole in zip(material.complex_residues, material.complex_poles):
        value += 0.5 * (
            residue / (s - pole)
            + residue.conjugate() / (s - pole.conjugate())
        )
    return value


def eval_edg_reflection(omega: numpy.ndarray, coeffs: dict[str, numpy.ndarray | float]) -> numpy.ndarray:
    value = float(coeffs["RI"]) * numpy.ones_like(omega, dtype=complex)
    for residue, damping in zip(coeffs["AS"], coeffs["lambdaS"]):
        value += residue / (1j * omega + damping)
    for b, c, alpha, beta in zip(coeffs["BS"], coeffs["CS"], coeffs["alphaS"], coeffs["betaS"]):
        value += 0.5 * (
            (b + 1j * c) / (alpha + 1j * beta + 1j * omega)
            + (b - 1j * c) / (alpha - 1j * beta + 1j * omega)
        )
    return value


def apply_passive_scale(coeffs: dict[str, numpy.ndarray | float]) -> float:
    omega = numpy.linspace(1.0, 2.0 * numpy.pi * FREQ_MAX_PASSIVITY, 8000)
    current = numpy.max(numpy.abs(eval_edg_reflection(omega, coeffs)))
    if current <= 1.0:
        return 1.0
    lo = 0.0
    hi = 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        trial = dict(coeffs)
        trial["AS"] = coeffs["AS"] * mid
        trial["BS"] = coeffs["BS"] * mid
        trial["CS"] = coeffs["CS"] * mid
        if numpy.max(numpy.abs(eval_edg_reflection(omega, trial))) <= 1.0:
            lo = mid
        else:
            hi = mid
    coeffs["AS"] = coeffs["AS"] * lo
    coeffs["BS"] = coeffs["BS"] * lo
    coeffs["CS"] = coeffs["CS"] * lo
    return lo


def write_material(material: ComsolPff) -> None:
    freq = read_frequency_table(CASE_DIR / material.table)
    coeffs = reflection_partial_fraction(material)
    scale = apply_passive_scale(coeffs)
    admittance = eval_comsol_admittance(freq, material)
    true_value = (1.0 - Z0 * admittance) / (1.0 + Z0 * admittance)
    omega = 2.0 * numpy.pi * freq
    approx_value = eval_edg_reflection(omega, coeffs)
    rms_error = float(numpy.sqrt(numpy.mean(numpy.abs(approx_value - true_value) ** 2)))
    max_error = float(numpy.max(numpy.abs(approx_value - true_value)))
    omega_check = numpy.linspace(1.0, 2.0 * numpy.pi * FREQ_MAX_PASSIVITY, 8000)
    max_abs_r = float(numpy.max(numpy.abs(eval_edg_reflection(omega_check, coeffs))))
    if max_abs_r > 1.0 + 1.0e-8:
        raise RuntimeError(f"{material.name} is not passive after scaling: {max_abs_r}")

    output = CASE_DIR / f"{material.name}.mat"
    scipy.io.savemat(
        output,
        {
            **coeffs,
            "freq": freq,
            "trueValue": true_value,
            "ApproxValue": approx_value,
            "comsolPffAdmittance": admittance,
            "target_source": "COMSOL partial-fraction admittance",
            "pff_frequency_convention": "s=i*f",
            "edg_frequency_convention": "s=i*omega",
            "rho0": RHO0,
            "c0": C0,
            "Z0": Z0,
            "passive_scale": scale,
            "rms_error": rms_error,
            "max_error": max_error,
            "max_abs_R": max_abs_r,
            "freq_max_passivity": FREQ_MAX_PASSIVITY,
        },
    )
    print(
        f"{material.name}: wrote {output.name}, rms={rms_error:.6g}, "
        f"max={max_error:.6g}, max_abs_R={max_abs_r:.12g}, scale={scale:.12g}"
    )


def main() -> int:
    for material in MATERIALS:
        write_material(material)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
