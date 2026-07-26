from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import edg_acoustics
import numpy
import pytest
import scipy.io
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "examples" / "car_cabin_acoustics_transient_63_cleared"
MAIN_PATH = CASE_DIR / "main.py"
COMPARE_PATH = CASE_DIR / "compare_microphone_response.py"


def load_python_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_case_main():
    return load_python_module("car_cabin_main", MAIN_PATH)


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


@pytest.mark.parametrize("use_scaled_flux_kernels", [False, True])
def test_normal_velocity_boundary_flux_uses_local_fscale_when_scaled(
    use_scaled_flux_kernels: bool,
):
    sim = edg_acoustics.AcousticsSimulation.__new__(edg_acoustics.AcousticsSimulation)
    sim.rho0 = 1.2
    sim.c0 = 343.0
    sim._use_triton_boundary_ri = False
    sim._use_triton_boundary_ade = False
    sim._use_scaled_flux_kernels = use_scaled_flux_kernels

    dtype = torch.float64
    n_boundary = 2
    q_flat = torch.tensor([2.0, -1.0, 0.1, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=dtype)
    flux_flat = torch.zeros_like(q_flat)
    normal_velocity = torch.tensor([0.35, -0.05], dtype=dtype)
    fscale = torch.tensor([2.0, 5.0], dtype=dtype)
    node = {
        "vmap_q": torch.arange(4 * n_boundary),
        "flux_map_q": torch.arange(4 * n_boundary),
        "nx": torch.ones(n_boundary, dtype=dtype),
        "ny": torch.zeros(n_boundary, dtype=dtype),
        "nz": torch.zeros(n_boundary, dtype=dtype),
        "fscale": fscale,
    }
    bcvar = {
        "vn": torch.zeros(n_boundary, dtype=dtype),
        "ou": torch.zeros(n_boundary, dtype=dtype),
        "in": torch.zeros(n_boundary, dtype=dtype),
        "normal_velocity": normal_velocity.clone(),
    }
    bc_cache = {
        "RI": torch.ones(n_boundary, dtype=dtype),
        "simple_RI": True,
        "has_normal_velocity": True,
        "boundary_q": torch.empty((4, n_boundary), dtype=dtype),
        "boundary_temp": torch.empty(n_boundary, dtype=dtype),
        "boundary_flux": torch.empty((4, n_boundary), dtype=dtype),
        "incoming_outgoing": torch.empty(n_boundary, dtype=dtype),
    }

    sim._compute_boundary_flux(bc_cache, node, bcvar, q_flat, flux_flat)

    vn = q_flat[2:4]
    expected = torch.zeros((4, n_boundary), dtype=dtype)
    expected[0] = sim.rho0 * sim.c0**2 * (vn - normal_velocity)
    expected[1] = sim.c0 * (normal_velocity - vn)
    if use_scaled_flux_kernels:
        expected *= fscale.unsqueeze(0)

    assert torch.allclose(flux_flat, expected.reshape(-1), rtol=1.0e-12, atol=1.0e-12)
    assert torch.allclose(bcvar["normal_velocity"], normal_velocity)


def test_save_results_on_the_run_writes_completed_checkpoint(tmp_path: Path):
    sim = edg_acoustics.AcousticsSimulation.__new__(edg_acoustics.AcousticsSimulation)
    sim.IC = SimpleNamespace(
        metadata={"kind": "unit"},
        source_xyz=numpy.array([1.0, 2.0, 3.0]),
        halfwidth=0.25,
    )
    sim.BC = SimpleNamespace(BCpara=[{"label": 1, "RI": 1.0}])
    sim.prec = torch.arange(10, dtype=torch.float64).reshape(1, 10)
    sim.prec_times = numpy.arange(1, 11, dtype=float) * 0.1
    sim.rec = numpy.array([[0.0], [0.0], [0.0]])
    sim.time_integrator = SimpleNamespace(dt=0.1, Nt=4, CFL=0.5)
    sim.Ntimesteps = 10
    sim.Np = 4
    sim.N_tets = 2
    sim.rho0 = 1.2
    sim.c0 = 343.0
    sim.mesh = SimpleNamespace(filename="dummy.msh")
    sim.Nx = 3

    sim.save_results_on_the_run(output_dir=tmp_path, format="mat", step_index=4)

    data = scipy.io.loadmat(tmp_path / "results_on_the_run.mat")
    assert data["prec"].shape == (1, 4)
    assert data["prec_times"].reshape(-1).tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert int(data["current_step"].reshape(-1)[0]) == 4
    assert float(data["current_time"].reshape(-1)[0]) == pytest.approx(0.4)
    assert int(data["Ntimesteps"].reshape(-1)[0]) == 10
    assert float(data["total_time"].reshape(-1)[0]) == pytest.approx(1.0)


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


def test_car_cabin_receivers_match_comsol_microphone_response_points():
    module = load_case_main()

    assert module.COMSOL_MICROPHONE_POINT_IDS.tolist() == [197, 391, 402]
    numpy.testing.assert_allclose(
        module.RECEIVER,
        numpy.array(
            [
                [2.0, 2.5, 2.5],
                [-0.05, -0.55, 0.55],
                [1.2, 1.2, 1.2],
            ],
            dtype=float,
        ),
    )


def test_car_cabin_comsol_golden_export_targets_microphone_response_points():
    source = (CASE_DIR / "ExportComsolMicrophoneGolden.java").read_text()

    assert "POINT_IDS = new int[] {197, 391, 402}" in source
    assert 'set("data", "dset2")' in source
    assert 'setIndex("expr", "pate.p_t", 0)' in source
    assert "EXPECTED_NSAMPLES = 2401" in source


def test_car_cabin_microphone_compare_metrics(tmp_path: Path):
    compare = load_python_module("compare_microphone_response", COMPARE_PATH)
    comsol_path = tmp_path / "comsol.csv"
    edg_path = tmp_path / "edg.mat"

    comsol_path.write_text(
        "\n".join(
            [
                "# test golden",
                "time,p197,p391,p402",
                "0.0,0.0,1.0,2.0",
                "0.1,0.1,1.1,2.1",
                "0.2,0.2,1.2,2.2",
            ]
        )
        + "\n"
    )
    scipy.io.savemat(
        edg_path,
        {
            "prec": numpy.array([[0.05, 0.16], [1.05, 1.16], [2.05, 2.16]]),
            "prec_times": numpy.array([0.05, 0.15]),
            "rec": compare.COMSOL_RECEIVER,
            "receiver_point_ids": compare.COMSOL_POINT_IDS,
        },
    )

    t_comsol, p_comsol = compare.load_comsol_golden(comsol_path)
    t_edg, p_edg, receiver, point_ids = compare.load_edg_result(edg_path)
    compare.validate_edg_receiver(receiver, point_ids)
    p_ref = compare.interpolate_comsol_to_edg(t_comsol, p_comsol, t_edg)
    metrics = compare.compute_metrics(t_edg, p_edg, p_ref)

    numpy.testing.assert_allclose(
        p_ref,
        numpy.array([[0.05, 0.15], [1.05, 1.15], [2.05, 2.15]]),
    )
    assert metrics["global"]["max_abs"] == pytest.approx(0.01)
    assert metrics["per_receiver"][0]["point_id"] == 197


@pytest.mark.parametrize(
    ("name", "rms_limit"),
    (("seat", 1.0e-10), ("carpet", 1.0e-10), ("roof", 1.0e-10)),
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
