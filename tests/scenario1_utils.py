"""Shared helpers for scenario 1 regression and benchmark tests."""

from __future__ import annotations

import copy
import glob
from contextlib import contextmanager
from pathlib import Path

import numpy
import scipy.io
import torch

import edg_acoustics
import edg_acoustics.acoustics_simulation as acoustics_simulation
import edg_acoustics.device_ini as device_ini


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden_files" / "test_scenario1"
EXAMPLE_DIR = REPO_ROOT / "examples" / "scenario1"

RHO0 = 1.213
C0 = 343
BC_LABELS = {
    "hard wall": 11,
    "carpet": 13,
    "panel": 14,
}
MONOPOLE_XYZ = numpy.array([3.04, 2.59, 1.62])
FREQ_UPPER_LIMIT = 200
NX = 4
NT = 3
CFL = 0.5
REC = numpy.vstack((numpy.array([4.26]), numpy.array([1.76]), numpy.array([1.62])))
MESH_NAME = "scenario1_coarser.msh"


def load_bc_para(data_dir: Path = GOLDEN_DIR):
    """Load scenario 1 boundary parameters from material .mat files."""
    bc_para = []
    for material, label in BC_LABELS.items():
        if material == "hard wall":
            bc_para.append({"label": label, "RI": 1})
            continue

        mat_files = glob.glob(str(data_dir / f"{material}*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"No .mat file found for material '{material}'")

        mat_file = scipy.io.loadmat(mat_files[0])
        material_dict = {"label": label}
        material_dict["RI"] = mat_file["RI"][0] if "RI" in mat_file else 0

        if "AS" in mat_file and "lambdaS" in mat_file:
            material_dict["RP"] = numpy.array(
                [mat_file["AS"][0], mat_file["lambdaS"][0]]
            )
        if all(key in mat_file for key in ("BS", "CS", "alphaS", "betaS")):
            material_dict["CP"] = numpy.array(
                [
                    mat_file["BS"][0],
                    mat_file["CS"][0],
                    mat_file["alphaS"][0],
                    mat_file["betaS"][0],
                ]
            )

        bc_para.append(material_dict)

    return bc_para


def resolve_mesh_path(mesh_name: str | Path, data_dir: Path = GOLDEN_DIR):
    mesh_path = Path(mesh_name)
    if mesh_path.is_absolute():
        return mesh_path

    data_dir_mesh = data_dir / mesh_path
    if data_dir_mesh.exists():
        return data_dir_mesh

    return EXAMPLE_DIR / mesh_path


@contextmanager
def acoustic_device(device: str | torch.device):
    """Temporarily route newly created simulation objects to a device."""
    target = torch.device(device)
    previous_device_ini = device_ini.device
    previous_simulation_device = acoustics_simulation.device
    device_ini.device = target
    acoustics_simulation.device = target
    try:
        yield target
    finally:
        device_ini.device = previous_device_ini
        acoustics_simulation.device = previous_simulation_device


def build_scenario1_simulation(
    data_dir: Path = GOLDEN_DIR,
    mesh_name: str | Path = MESH_NAME,
    device: str | torch.device | None = None,
):
    """Build a fully initialized scenario 1 simulation without advancing time."""
    if device is not None:
        with acoustic_device(device):
            return build_scenario1_simulation(data_dir, mesh_name)

    mesh = edg_acoustics.Mesh(str(resolve_mesh_path(mesh_name, data_dir)), BC_LABELS)
    sim = edg_acoustics.AcousticsSimulation(RHO0, C0, NX, mesh, BC_LABELS)
    sim.init_BC(edg_acoustics.AbsorbBC(sim.BCnode, load_bc_para(data_dir)))
    sim.init_IC(edg_acoustics.Monopole_IC(MONOPOLE_XYZ, FREQ_UPPER_LIMIT))
    sim.init_Flux(edg_acoustics.UpwindFlux(RHO0, C0, sim.n_xyz))
    sim.init_rec(REC, "scipy")
    sim.init_TimeIntegrator(edg_acoustics.TSI_TI(sim.RHS_operator, sim.dtscale, CFL, Nt=NT))
    return sim


def clone_bcvar(bcvar: list[dict]):
    """Deep-copy BC state tensors for RHS comparison/golden generation."""
    cloned = []
    for item in bcvar:
        next_item = {}
        for key, value in item.items():
            next_item[key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        cloned.append(next_item)
    return cloned


def tensor_to_numpy(value: torch.Tensor):
    """Convert a tensor to a CPU NumPy array."""
    return value.detach().cpu().numpy()


def assert_tensor_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
):
    assert actual.dtype == torch.float64
    assert expected.dtype == torch.float64
    numpy.testing.assert_allclose(
        tensor_to_numpy(actual),
        tensor_to_numpy(expected),
        rtol=rtol,
        atol=atol,
        err_msg=name,
    )


def assert_bc_state_close(
    actual: list[dict],
    expected: list[dict],
    *,
    rtol: float,
    atol: float,
):
    assert len(actual) == len(expected)
    for index, (actual_state, expected_state) in enumerate(zip(actual, expected)):
        assert actual_state.keys() == expected_state.keys()
        for key, actual_value in actual_state.items():
            expected_value = expected_state[key]
            if torch.is_tensor(actual_value):
                assert_tensor_close(
                    f"bc{index}_{key}",
                    actual_value,
                    expected_value,
                    rtol=rtol,
                    atol=atol,
                )
            elif isinstance(actual_value, numpy.ndarray):
                numpy.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"bc{index}_{key}",
                )
            else:
                assert actual_value == expected_value


def assert_rhs_close(actual, expected, *, rtol: float, atol: float):
    for name, actual_value, expected_value in zip(
        ("rhs_p", "rhs_vx", "rhs_vy", "rhs_vz"),
        actual[:4],
        expected[:4],
    ):
        assert_tensor_close(
            name,
            actual_value,
            expected_value,
            rtol=rtol,
            atol=atol,
        )
    assert_bc_state_close(actual[4], expected[4], rtol=rtol, atol=atol)


def assert_simulation_state_close(actual, expected, *, rtol: float, atol: float):
    for name in ("P", "Vx", "Vy", "Vz", "prec"):
        assert_tensor_close(
            name,
            getattr(actual, name),
            getattr(expected, name),
            rtol=rtol,
            atol=atol,
        )
    assert_bc_state_close(actual.BC.BCvar, expected.BC.BCvar, rtol=rtol, atol=atol)
