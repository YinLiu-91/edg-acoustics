"""Regression checks for the constrained car-cabin seat material fit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy
import scipy.io


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "examples" / "car_cabin_acoustics_transient_63_cleared"
FIT_SCRIPT = CASE_DIR / "fit_seat_admittance.py"


def load_fit_module():
    spec = importlib.util.spec_from_file_location("fit_car_cabin_seat", FIT_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_constrained_seat_fit_is_accurate_stable_and_passive():
    fit = load_fit_module()
    data = scipy.io.loadmat(CASE_DIR / "seat.mat")
    frequency_hz, target = fit.load_target()
    bs = numpy.asarray(data["BS"]).reshape(-1)
    cs = numpy.asarray(data["CS"]).reshape(-1)
    alpha = numpy.asarray(data["alphaS"]).reshape(-1)
    beta = numpy.asarray(data["betaS"]).reshape(-1)
    approximation = fit.evaluate_reflection(
        2.0 * numpy.pi * frequency_hz,
        float(numpy.asarray(data["RI"]).reshape(-1)[0]),
        bs,
        cs,
        alpha,
        beta,
    )
    error = approximation - target
    validation_omega = numpy.linspace(
        0.0, 2.0 * numpy.pi * fit.FREQ_MAX_PASSIVITY_HZ, 20001
    )
    validation_reflection = fit.evaluate_reflection(
        validation_omega,
        float(numpy.asarray(data["RI"]).reshape(-1)[0]),
        bs,
        cs,
        alpha,
        beta,
    )

    assert "AS" not in data or numpy.asarray(data["AS"]).size == 0
    assert "lambdaS" not in data or numpy.asarray(data["lambdaS"]).size == 0
    assert len(bs) == len(fit.POLE_FREQUENCIES_HZ)
    assert len(cs) == len(bs)
    assert len(alpha) == len(bs)
    assert len(beta) == len(bs)
    numpy.testing.assert_allclose(
        numpy.asarray(data["pole_frequencies_hz"]).reshape(-1),
        fit.POLE_FREQUENCIES_HZ,
        rtol=0.0,
        atol=1.0e-12,
    )
    numpy.testing.assert_allclose(
        beta / (2.0 * numpy.pi),
        fit.POLE_FREQUENCIES_HZ,
        rtol=0.0,
        atol=1.0e-12,
    )
    numpy.testing.assert_allclose(
        approximation,
        numpy.asarray(data["ApproxValue"]).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    numpy.testing.assert_allclose(
        target,
        numpy.asarray(data["trueValue"]).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert numpy.all(alpha > 0.0)
    assert numpy.all(beta > 0.0)
    assert numpy.sqrt(numpy.mean(numpy.abs(error) ** 2)) <= fit.RMS_ERROR_LIMIT
    assert numpy.abs(error).max() <= fit.MAX_ERROR_LIMIT + 1.0e-8
    assert (
        numpy.sqrt(
            numpy.average(
                numpy.abs(error) ** 2, weights=fit.source_weights(frequency_hz)
            )
        )
        <= fit.SOURCE_WEIGHTED_RMS_LIMIT
    )
    assert numpy.abs(validation_reflection).max() <= 1.0 + 1.0e-8
