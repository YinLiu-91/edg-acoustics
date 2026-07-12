"""Smoke tests for the minimal 2D acoustic DG support."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import meshio
import numpy
import pytest
import scipy.io
import torch

import edg_acoustics
from scenario1_utils import acoustic_device


REPO_ROOT = Path(__file__).resolve().parents[1]
SQUARE_DIR = REPO_ROOT / "examples" / "dgfem_acoustic_2d" / "square"
SQUARE_MESH = SQUARE_DIR / "square.msh"
SQUARE_MAIN = SQUARE_DIR / "main.py"


def load_example_module(path: Path):
    """Import a standalone example module from a file path."""
    spec = importlib.util.spec_from_file_location(
        f"example_{path.parent.name}_{path.stem}",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mesh2d_loads_square_connectivity():
    with acoustic_device("cpu"):
        mesh = edg_acoustics.Mesh2D(str(SQUARE_MESH), {"Absorbing": 11})

    assert mesh.dim == 2
    assert mesh.N_elements > 0
    assert mesh.N_triangles == mesh.N_elements
    assert mesh.N_BC_lines["Absorbing"] > 0
    assert mesh.BC_lines["Absorbing"].shape[1] == 2
    assert mesh.EToE.shape == (3, mesh.N_elements)
    assert mesh.EToF.shape == (3, mesh.N_elements)


def test_mesh2d_connectivity_hash_uses_vertex_id_range():
    with acoustic_device("cpu"):
        mesh = object.__new__(edg_acoustics.Mesh2D)
        EToV = torch.tensor([[0, 2], [4, 3], [5, 6]])
        EToE, EToF = mesh.compute_element_connectivity(EToV)

    expected_EToE = torch.arange(2).reshape(1, -1).repeat(3, 1)
    expected_EToF = torch.arange(3).repeat_interleave(2).reshape(3, 2)
    torch.testing.assert_close(EToE.cpu(), expected_EToE)
    torch.testing.assert_close(EToF.cpu(), expected_EToF)


def test_acoustics_simulation_2d_short_tsi_cpu():
    module = load_example_module(SQUARE_MAIN)
    with acoustic_device("cpu"):
        sim = module.build_simulation(Nx=1, Nt=3, cfl=0.5)
        sim.time_integration(n_time_steps=2, progress=False)

    assert sim.Q.dtype == torch.float64
    assert sim.Q.device.type == "cpu"
    assert torch.isfinite(sim.Q).all()
    assert numpy.isfinite(torch.linalg.vector_norm(sim.Q).item())
    assert torch.linalg.vector_norm(sim.Q).item() > 0
    torch.testing.assert_close(sim.Vz, torch.zeros_like(sim.Vz))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="2D CUDA parity requires CUDA")
def test_acoustics_simulation_2d_short_tsi_cuda_matches_cpu():
    module = load_example_module(SQUARE_MAIN)
    with acoustic_device("cpu"):
        cpu_sim = module.build_simulation(Nx=1, Nt=3, cfl=0.5)
        cpu_sim.time_integration(n_time_steps=2, progress=False)

    with acoustic_device("cuda"):
        cuda_sim = module.build_simulation(Nx=1, Nt=3, cfl=0.5)
        cuda_sim.time_integration(n_time_steps=2, progress=False)
        torch.cuda.synchronize()

    torch.testing.assert_close(cuda_sim.Q.cpu(), cpu_sim.Q, rtol=1.0e-9, atol=1.0e-9)


def test_square_example_writes_snapshot(tmp_path):
    module = load_example_module(SQUARE_MAIN)
    snapshot_path = tmp_path / "square_snapshot.mat"
    with acoustic_device("cpu"):
        output_path, sim = module.run_case(
            n_time_steps=1,
            snapshot_path=snapshot_path,
            progress=False,
        )

    assert output_path == snapshot_path
    assert snapshot_path.exists()
    assert sim.P.shape[1] == sim.mesh.N_elements

    data = scipy.io.loadmat(snapshot_path)
    for key in ("P", "Vx", "Vy", "Vz"):
        assert key in data
        assert data[key].size > 0


def test_save_mesh_results_on_the_run_2d(tmp_path: Path):
    module = load_example_module(SQUARE_MAIN)
    with acoustic_device("cpu"):
        sim = module.build_simulation(Nx=1, Nt=3, cfl=0.5)
        sim.save_mesh_results_on_the_run(
            output_dir=str(tmp_path),
            step_index=7,
            real_time=2.5e-4,
        )

    saved_files = list(tmp_path.glob("*.msh"))
    assert len(saved_files) == 1
    saved_file = saved_files[0]
    assert "step000007" in saved_file.name
    assert "t2.500000e-04" in saved_file.name

    mesh = meshio.read(saved_file)
    assert mesh.points.shape[0] == sim.mesh.N_vertices
    assert set(mesh.point_data) == {"P", "Vx", "Vy", "Vz"}
    for key in ("P", "Vx", "Vy", "Vz"):
        assert mesh.point_data[key].shape == (sim.mesh.N_vertices,)
        assert numpy.isfinite(mesh.point_data[key]).all()


def test_save_results_on_the_run_2d(tmp_path: Path):
    module = load_example_module(SQUARE_MAIN)
    with acoustic_device("cpu"):
        sim = module.build_simulation(Nx=1, Nt=3, cfl=0.5)
        sim.time_integration(n_time_steps=1, progress=False)
        sim.save_results_on_the_run(
            output_dir=tmp_path,
            format="mat",
            step_index=1,
        )

    saved_file = tmp_path / "results_on_the_run.mat"
    assert saved_file.exists()
    data = scipy.io.loadmat(saved_file)
    for key in ("P", "Vx", "Vy", "Vz", "xyz"):
        assert key in data
        assert data[key].size > 0
    assert int(data["current_step"].reshape(-1)[0]) == 1


def test_square_example_time_integration_saves_mesh_snapshots(tmp_path: Path):
    module = load_example_module(SQUARE_MAIN)
    snapshot_path = tmp_path / "square_snapshot.mat"
    mesh_dir = tmp_path / "mesh_frames"
    with acoustic_device("cpu"):
        _, _ = module.run_case(
            n_time_steps=2,
            snapshot_path=snapshot_path,
            save_mesh_step=1,
            save_mesh_dir=mesh_dir,
            progress=False,
        )

    saved_files = sorted(mesh_dir.glob("*.msh"))
    assert len(saved_files) == 2
    assert "step000001" in saved_files[0].name
    assert "step000002" in saved_files[1].name


def test_square_example_time_integration_saves_results_and_mesh_by_default_convention(tmp_path: Path):
    module = load_example_module(SQUARE_MAIN)
    original_case_dir = module.CASE_DIR
    original_snapshot = module.DEFAULT_SNAPSHOT
    original_results = module.DEFAULT_RESULTS_ON_THE_RUN
    original_results_msh_dir = module.DEFAULT_RESULTS_ON_THE_RUN_MSH_DIR
    try:
        module.CASE_DIR = tmp_path
        module.DEFAULT_SNAPSHOT = tmp_path / "square_snapshot.mat"
        module.DEFAULT_RESULTS_ON_THE_RUN = tmp_path / "results_on_the_run.mat"
        module.DEFAULT_RESULTS_ON_THE_RUN_MSH_DIR = tmp_path / "results_on_the_run_msh"
        with acoustic_device("cpu"):
            module.run_case(
                n_time_steps=1,
                snapshot_path=module.DEFAULT_SNAPSHOT,
                save_step=1,
                save_mesh_step=1,
                progress=False,
            )
    finally:
        module.CASE_DIR = original_case_dir
        module.DEFAULT_SNAPSHOT = original_snapshot
        module.DEFAULT_RESULTS_ON_THE_RUN = original_results
        module.DEFAULT_RESULTS_ON_THE_RUN_MSH_DIR = original_results_msh_dir

    assert (tmp_path / "results_on_the_run.mat").exists()
    mesh_files = sorted((tmp_path / "results_on_the_run_msh").glob("*.msh"))
    assert len(mesh_files) == 1
