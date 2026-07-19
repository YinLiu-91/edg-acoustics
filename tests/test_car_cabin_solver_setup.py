from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy
import pytest
import scipy.io
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "examples" / "car_cabin_acoustics_transient_63_cleared"
MAIN_PATH = CASE_DIR / "main.py"
sys.path.insert(0, str(REPO_ROOT))
for module_name in list(sys.modules):
    if module_name == "edg_acoustics" or module_name.startswith("edg_acoustics."):
        sys.modules.pop(module_name, None)
import edg_acoustics


def load_case_main():
    spec = importlib.util.spec_from_file_location("car_cabin_main", MAIN_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gaussian_modulated_sine_normal_velocity_derivative():
    config = {
        "kind": "gaussian_modulated_sine",
        "amplitude": 1.0,
        "frequency": 1000.0,
        "delay": 0.002,
        "sigma": 0.0005,
        "phase": 0.0,
        "baseline": 0.0,
    }
    t = 0.0023
    omega = 2.0 * numpy.pi * config["frequency"]
    u = (t - config["delay"]) / config["sigma"]
    envelope = numpy.exp(-0.5 * u * u)
    expected_value = envelope * numpy.sin(omega * t)
    expected_first = envelope * (
        omega * numpy.cos(omega * t)
        - u / config["sigma"] * numpy.sin(omega * t)
    )

    value = edg_acoustics.AbsorbBC.evaluate_normal_velocity(config, t, 0)
    first = edg_acoustics.AbsorbBC.evaluate_normal_velocity(config, t, 1)

    assert value == pytest.approx(expected_value, rel=1.0e-12, abs=1.0e-12)
    assert first == pytest.approx(expected_first, rel=1.0e-12, abs=1.0e-12)


def test_absorbbc_prepares_prescribed_normal_velocity_tensor():
    bcnode = [{"label": 21, "map": torch.arange(4)}]
    params = [
        {
            "label": 21,
            "RI": 1.0,
            "normal_velocity": {
                "kind": "gaussian_modulated_sine",
                "amplitude": 1.0,
                "frequency": 1000.0,
                "delay": 0.002,
                "sigma": 0.0005,
            },
        }
    ]

    bc = edg_acoustics.AbsorbBC(bcnode, params)
    time = torch.tensor(0.002, dtype=bc.BCvar[0]["normal_velocity"].dtype)
    bc.prepare_prescribed_normal_velocity(time, 0)

    assert bc.has_prescribed_normal_velocity is True
    assert torch.allclose(
        bc.BCvar[0]["normal_velocity"],
        torch.zeros_like(bc.BCvar[0]["normal_velocity"]),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_car_cabin_boundary_parameters_match_comsol_case():
    module = load_case_main()
    bc = module.build_bc_parameters()
    by_label = {entry["label"]: entry for entry in bc}

    assert by_label[11]["RI"] == 1.0
    assert by_label[12]["RI"] == pytest.approx(numpy.sqrt(0.995))
    assert by_label[13]["RI"] == pytest.approx(numpy.sqrt(0.99))
    assert by_label[14]["RI"] == pytest.approx(numpy.sqrt(0.99))
    assert by_label[21]["normal_velocity"]["frequency"] == 1000.0
    assert by_label[21]["normal_velocity"]["delay"] == pytest.approx(0.002)
    assert by_label[21]["normal_velocity"]["sigma"] == pytest.approx(0.0005)


@pytest.mark.parametrize(
    ("name", "rms_limit"),
    (("seat", 0.08), ("carpet", 5.0e-4), ("roof", 5.0e-4)),
)
def test_car_cabin_material_fits_are_passive(name: str, rms_limit: float):
    mat_path = CASE_DIR / f"{name}.mat"
    if not mat_path.exists():
        pytest.skip(f"{mat_path} has not been generated")
    data = scipy.io.loadmat(mat_path)
    params = {"label": 0, "RI": float(numpy.asarray(data["RI"]).reshape(-1)[0])}
    if "AS" in data and "lambdaS" in data:
        params["RP"] = numpy.vstack(
            (
                numpy.asarray(data["AS"]).reshape(-1),
                numpy.asarray(data["lambdaS"]).reshape(-1),
            )
        )
    if {"BS", "CS", "alphaS", "betaS"} <= set(data):
        params["CP"] = numpy.vstack(
            (
                numpy.asarray(data["BS"]).reshape(-1),
                numpy.asarray(data["CS"]).reshape(-1),
                numpy.asarray(data["alphaS"]).reshape(-1),
                numpy.asarray(data["betaS"]).reshape(-1),
            )
        )

    omega = torch.linspace(1.0, 2.0 * numpy.pi * 2000.0, 5000)
    reflection = edg_acoustics.AbsorbBC.compute_Re(omega, params)

    assert float(numpy.asarray(data["rms_error"]).reshape(-1)[0]) <= rms_limit
    assert torch.max(torch.abs(reflection)).item() <= 1.0 + 1.0e-8
    if "RP" in params:
        assert numpy.all(params["RP"][1] > 0.0)
    if "CP" in params:
        assert numpy.all(params["CP"][2] > 0.0)
