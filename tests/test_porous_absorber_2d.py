"""Tests for the 2D porous absorber ER example."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest
import scipy.io
import torch

import edg_acoustics
from scenario1_utils import acoustic_device


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = (
    REPO_ROOT
    / "examples"
    / "dgfem_acoustic_2d"
    / "porous_absorber_time_domain"
)
EXAMPLE_MAIN = EXAMPLE_DIR / "main.py"
SQUARE_MESH = REPO_ROOT / "examples" / "dgfem_acoustic_2d" / "square" / "square.msh"
COMSOL_GOLDEN_PATHS = {
    0.05: EXAMPLE_DIR / "5cm_er_comsol_golden.txt",
    0.15: EXAMPLE_DIR / "15cm_er_comsol_golden.txt",
}


def load_example_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"example_{path.parent.name}_{path.stem}",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_synthetic_fit(path: Path):
    scipy.io.savemat(
        path,
        {
            "A_beta": numpy.zeros((0, 0), dtype=float),
            "B_beta": numpy.zeros((0, 1), dtype=float),
            "C_beta": numpy.zeros((1, 0), dtype=float),
            "D_beta": numpy.array([[1.0 / (1.8 * 180.0**2)]], dtype=float),
            "rmserr_beta": numpy.array([[0.0]], dtype=float),
            "A_rho": numpy.zeros((0, 0), dtype=float),
            "B_rho": numpy.zeros((0, 1), dtype=float),
            "C_rho": numpy.zeros((1, 0), dtype=float),
            "D_rho": numpy.array([[1.8]], dtype=float),
            "rmserr_rho": numpy.array([[0.0]], dtype=float),
        },
    )


def compute_comsol_error_metrics(sim_time, sim_pressure, golden_path: Path):
    golden = numpy.loadtxt(golden_path, comments="%", dtype=float)
    overlap = (golden[:, 0] >= sim_time[0]) & (golden[:, 0] <= sim_time[-1])
    golden_time = golden[overlap, 0]
    golden_pressure = golden[overlap, 1]
    aligned_pressure = numpy.interp(golden_time, sim_time, sim_pressure)
    error = aligned_pressure - golden_pressure
    return {
        "rmse": float(numpy.sqrt(numpy.mean(error**2))),
        "max_abs": float(numpy.max(numpy.abs(error))),
        "rel_l2": float(numpy.linalg.norm(error) / numpy.linalg.norm(golden_pressure)),
    }


def test_mesh2d_loads_domain_labels():
    with acoustic_device("cpu"):
        mesh = edg_acoustics.Mesh2D(
            str(SQUARE_MESH),
            {"Absorbing": 11},
            {"Domain": 1},
        )

    assert mesh.element_physical_labels.shape[0] == mesh.N_elements
    assert mesh.N_domain_elements["Domain"] == mesh.N_elements
    assert mesh.domain_elements["Domain"].shape[0] == mesh.N_elements


def test_radial_pressure_pulse_2d_is_finite():
    ic = edg_acoustics.RadialPressurePulse2D_IC(
        numpy.array([0.0, 0.0, 0.0], dtype=float),
        0.045,
    )
    xyz = torch.zeros((3, 4, 2), dtype=torch.float64)
    pressure = ic.Pinit(xyz)
    torch.testing.assert_close(pressure[0, 0], torch.tensor(1.0, dtype=torch.float64))
    assert torch.isfinite(pressure).all()


@pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh is required for the porous example")
def test_porous_absorber_example_short_run(tmp_path: Path):
    module = load_example_module(EXAMPLE_MAIN)
    fit_path = tmp_path / "synthetic_er_fit.mat"
    mesh_path = tmp_path / "porous_example.msh"
    output_root = tmp_path / "outputs"
    write_synthetic_fit(fit_path)
    module.ensure_mesh(0.05, mesh_path=mesh_path, force=True)

    with acoustic_device("cpu"):
        case_info, sim = module.run_case(
            thickness=0.05,
            output_root=output_root,
            fit_path=fit_path,
            mesh_path=mesh_path,
            n_time_steps=2,
            save_step=1,
            save_mesh_step=1,
            save_mesh_at_ms=0.0,
            progress=False,
        )

    assert torch.isfinite(sim.Q).all()
    assert hasattr(sim, "prec")
    assert sim.prec.shape == (1, 2)

    results_file = case_info["output_dir"] / "results_on_the_run.mat"
    snapshot_file = case_info["snapshot_path"]
    mesh_files = sorted(case_info["mesh_output_dir"].glob("*.msh"))
    assert results_file.exists()
    assert snapshot_file.exists()
    assert len(mesh_files) == 2

    data = scipy.io.loadmat(results_file)
    assert "prec" in data
    assert "time" in data
    assert data["prec"].shape[1] == 2
    assert data["time"].reshape(-1)[0] == pytest.approx(sim.time_integrator.dt)


def test_porous_absorber_packed_rhs_matches_reference_cpu():
    module = load_example_module(EXAMPLE_MAIN)
    mesh_path = module.default_mesh_path(0.05)
    fit_path = module.DEFAULT_FIT

    with acoustic_device("cpu"):
        reference = module.build_simulation(
            thickness=0.05,
            fit_path=fit_path,
            mesh_path=mesh_path,
            Nx=1,
            Nt=2,
            use_packed_rhs=False,
        )
        packed = module.build_simulation(
            thickness=0.05,
            fit_path=fit_path,
            mesh_path=mesh_path,
            Nx=1,
            Nt=2,
            use_packed_rhs=True,
        )
        reference.time_integration(n_time_steps=3, progress=False, use_cuda_graph=False)
        packed.time_integration(n_time_steps=3, progress=False, use_cuda_graph=False)

    for field_name in ("P", "Vx", "Vy", "Vz", "Q", "prec"):
        torch.testing.assert_close(
            getattr(packed, field_name),
            getattr(reference, field_name),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    for state_name in ("z_beta", "z_rho_x", "z_rho_y"):
        torch.testing.assert_close(
            getattr(packed, state_name),
            getattr(reference, state_name),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    assert packed.last_time_integration_used_packed_rhs is True


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="2D porous CUDA graph parity requires CUDA"
)
def test_porous_absorber_cuda_graph_matches_eager():
    module = load_example_module(EXAMPLE_MAIN)
    mesh_path = module.default_mesh_path(0.05)
    fit_path = module.DEFAULT_FIT

    with acoustic_device("cuda"):
        eager = module.build_simulation(
            thickness=0.05,
            fit_path=fit_path,
            mesh_path=mesh_path,
            Nx=1,
            Nt=2,
        )
        graphed = module.build_simulation(
            thickness=0.05,
            fit_path=fit_path,
            mesh_path=mesh_path,
            Nx=1,
            Nt=2,
        )

        eager.time_integration(n_time_steps=5, progress=False, use_cuda_graph=False)
        graphed.time_integration(
            n_time_steps=5,
            progress=False,
            use_cuda_graph=True,
            cuda_graph_chunk_steps=2,
        )
        torch.cuda.synchronize()

    for field_name in ("P", "Vx", "Vy", "Vz", "Q", "prec"):
        torch.testing.assert_close(
            getattr(graphed, field_name),
            getattr(eager, field_name),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    for state_name in ("z_beta", "z_rho_x", "z_rho_y"):
        torch.testing.assert_close(
            getattr(graphed, state_name),
            getattr(eager, state_name),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    assert graphed.last_time_integration_used_cuda_graph is True
    assert graphed.last_time_integration_cuda_graph_mode == "full"
    assert graphed.last_time_integration_cuda_graph_chunk_steps == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="2D porous Triton parity requires CUDA")
def test_porous_absorber_triton_matches_packed_cuda():
    module = load_example_module(EXAMPLE_MAIN)
    mesh_path = module.default_mesh_path(0.05)
    fit_path = module.DEFAULT_FIT

    with acoustic_device("cuda"):
        packed = module.build_simulation(
            thickness=0.05,
            fit_path=fit_path,
            mesh_path=mesh_path,
            Nx=1,
            Nt=2,
            use_packed_rhs=True,
            use_triton_kernels=False,
        )
        triton_sim = module.build_simulation(
            thickness=0.05,
            fit_path=fit_path,
            mesh_path=mesh_path,
            Nx=1,
            Nt=2,
            use_packed_rhs=True,
            use_triton_kernels=True,
            use_triton_deep_rhs=False,
        )
        packed.time_integration(n_time_steps=5, progress=False, use_cuda_graph=False)
        triton_sim.time_integration(n_time_steps=5, progress=False, use_cuda_graph=False)
        torch.cuda.synchronize()

    for field_name in ("P", "Vx", "Vy", "Vz", "Q", "prec"):
        torch.testing.assert_close(
            getattr(triton_sim, field_name),
            getattr(packed, field_name),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    for state_name in ("z_beta", "z_rho_x", "z_rho_y"):
        torch.testing.assert_close(
            getattr(triton_sim, state_name),
            getattr(packed, state_name),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    assert triton_sim._use_triton_interior_flux is True
    assert triton_sim._use_triton_boundary_ri is True
    assert triton_sim._use_triton_deep_rhs is False


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="2D Triton receiver parity requires CUDA"
)
def test_porous_absorber_triton_receiver_handles_strided_pressure():
    module = load_example_module(EXAMPLE_MAIN)

    with acoustic_device("cuda"):
        sim = module.build_simulation(
            thickness=0.05,
            fit_path=module.DEFAULT_FIT,
            mesh_path=module.default_mesh_path(0.05),
            Nx=4,
            Nt=4,
            use_packed_rhs=True,
            use_triton_kernels=True,
        )
        assert not sim.P.is_contiguous()
        sim.P.copy_(
            torch.arange(
                sim.P.numel(), device=sim.device, dtype=sim.P.dtype
            ).reshape_as(sim.P)
        )
        expected = torch.empty_like(sim._sample_output)
        actual = torch.empty_like(sim._sample_output)
        sim._use_triton_receiver_sampling = False
        sim._sample_receivers(expected)
        sim._use_triton_receiver_sampling = True
        sim._sample_receivers(actual)
        torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="2D porous deep fused parity requires CUDA")
def test_porous_absorber_deep_fused_matches_triton_cuda():
    module = load_example_module(EXAMPLE_MAIN)
    mesh_path = module.default_mesh_path(0.05)
    fit_path = module.DEFAULT_FIT

    with acoustic_device("cuda"):
        triton_sim = module.build_simulation(
            thickness=0.05,
            fit_path=fit_path,
            mesh_path=mesh_path,
            Nx=1,
            Nt=2,
            use_packed_rhs=True,
            use_triton_kernels=True,
            use_triton_deep_rhs=False,
        )
        deep_sim = module.build_simulation(
            thickness=0.05,
            fit_path=fit_path,
            mesh_path=mesh_path,
            Nx=1,
            Nt=2,
            use_packed_rhs=True,
            use_triton_kernels=True,
            use_triton_deep_rhs=True,
            use_triton_partitioned_er_rhs=False,
        )
        triton_sim.time_integration(n_time_steps=5, progress=False, use_cuda_graph=False)
        deep_sim.time_integration(n_time_steps=5, progress=False, use_cuda_graph=False)
        torch.cuda.synchronize()

    for field_name in ("P", "Vx", "Vy", "Vz", "Q", "prec"):
        torch.testing.assert_close(
            getattr(deep_sim, field_name),
            getattr(triton_sim, field_name),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    for state_name in ("z_beta", "z_rho_x", "z_rho_y"):
        torch.testing.assert_close(
            getattr(deep_sim, state_name),
            getattr(triton_sim, state_name),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    assert deep_sim._use_triton_deep_rhs is True
    assert deep_sim._use_triton_partitioned_er_rhs is False


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="2D porous partitioned ER parity requires CUDA",
)
def test_porous_absorber_partitioned_er_matches_deep_fused_cuda():
    module = load_example_module(EXAMPLE_MAIN)
    mesh_path = module.default_mesh_path(0.05)
    fit_path = module.DEFAULT_FIT

    with acoustic_device("cuda"):
        legacy = module.build_simulation(
            thickness=0.05,
            fit_path=fit_path,
            mesh_path=mesh_path,
            Nx=1,
            Nt=2,
            use_packed_rhs=True,
            use_triton_kernels=True,
            use_triton_deep_rhs=True,
            use_triton_partitioned_er_rhs=False,
        )
        partitioned = module.build_simulation(
            thickness=0.05,
            fit_path=fit_path,
            mesh_path=mesh_path,
            Nx=1,
            Nt=2,
            use_packed_rhs=True,
            use_triton_kernels=True,
            use_triton_deep_rhs=True,
            use_triton_partitioned_er_rhs=True,
        )
        legacy.time_integration(
            n_time_steps=5, progress=False, use_cuda_graph=False
        )
        partitioned.time_integration(
            n_time_steps=5, progress=False, use_cuda_graph=False
        )
        torch.cuda.synchronize()

    for field_name in ("P", "Vx", "Vy", "Vz", "Q", "prec"):
        torch.testing.assert_close(
            getattr(partitioned, field_name),
            getattr(legacy, field_name),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    for state_name in ("z_beta", "z_rho_x", "z_rho_y"):
        torch.testing.assert_close(
            getattr(partitioned, state_name),
            getattr(legacy, state_name),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    assert partitioned._use_triton_partitioned_er_rhs is True
    assert partitioned._combined_simple_ri_boundary is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="2D porous deep fused CUDA graph parity requires CUDA"
)
def test_porous_absorber_deep_fused_cuda_graph_matches_eager():
    module = load_example_module(EXAMPLE_MAIN)
    mesh_path = module.default_mesh_path(0.05)
    fit_path = module.DEFAULT_FIT

    with acoustic_device("cuda"):
        eager = module.build_simulation(
            thickness=0.05,
            fit_path=fit_path,
            mesh_path=mesh_path,
            Nx=1,
            Nt=2,
            use_packed_rhs=True,
            use_triton_kernels=True,
            use_triton_deep_rhs=True,
            use_triton_partitioned_er_rhs=True,
        )
        graphed = module.build_simulation(
            thickness=0.05,
            fit_path=fit_path,
            mesh_path=mesh_path,
            Nx=1,
            Nt=2,
            use_packed_rhs=True,
            use_triton_kernels=True,
            use_triton_deep_rhs=True,
            use_triton_partitioned_er_rhs=True,
        )

        eager.time_integration(n_time_steps=5, progress=False, use_cuda_graph=False)
        graphed.time_integration(
            n_time_steps=5,
            progress=False,
            use_cuda_graph=True,
            cuda_graph_chunk_steps=2,
        )
        torch.cuda.synchronize()

    for field_name in ("P", "Vx", "Vy", "Vz", "Q", "prec"):
        torch.testing.assert_close(
            getattr(graphed, field_name),
            getattr(eager, field_name),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    for state_name in ("z_beta", "z_rho_x", "z_rho_y"):
        torch.testing.assert_close(
            getattr(graphed, state_name),
            getattr(eager, state_name),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    assert graphed._use_triton_deep_rhs is True
    assert graphed._use_triton_partitioned_er_rhs is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The default optimized porous case requires CUDA",
)
@pytest.mark.parametrize("thickness", [0.05, 0.15])
def test_porous_absorber_default_optimized_run_matches_comsol(thickness):
    module = load_example_module(EXAMPLE_MAIN)

    with acoustic_device("cuda"):
        sim = module.build_simulation(
            thickness=thickness,
            fit_path=module.DEFAULT_FIT,
            mesh_path=module.default_mesh_path(thickness),
        )
        sim.time_integration(
            total_time=module.DEFAULT_TOTAL_TIME,
            progress=False,
            use_cuda_graph=module.DEFAULT_USE_CUDA_GRAPH,
            cuda_graph_chunk_steps=module.DEFAULT_CUDA_GRAPH_CHUNK_STEPS,
        )
        torch.cuda.synchronize()
        sim_time = (
            numpy.arange(1, sim.Ntimesteps + 1, dtype=float)
            * sim.time_integrator.dt
        )
        sim_pressure = sim.prec[0].detach().cpu().numpy()

    assert module.DEFAULT_USE_2D_PACKED_RHS is True
    assert sim.last_time_integration_used_packed_rhs is True
    assert sim.last_time_integration_used_cuda_graph is True
    assert sim._use_triton_interior_flux is True
    assert sim._use_triton_boundary_ri is True
    assert sim._use_triton_deep_rhs is True
    assert sim._use_triton_partitioned_er_rhs is True

    metrics = compute_comsol_error_metrics(
        sim_time,
        sim_pressure,
        COMSOL_GOLDEN_PATHS[thickness],
    )
    assert metrics["rmse"] <= 1.0e-3
    assert metrics["max_abs"] <= 3.6e-3
    assert metrics["rel_l2"] <= 8.0e-2


@pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh is required for mesh validation")
def test_porous_absorber_rejects_wrong_thickness_mesh(tmp_path: Path):
    module = load_example_module(EXAMPLE_MAIN)
    mesh_path = tmp_path / "porous_5cm.msh"
    module.ensure_mesh(0.05, mesh_path=mesh_path, force=True)

    with pytest.raises(ValueError, match="expected -0.15"):
        module.ensure_mesh(0.15, mesh_path=mesh_path, force=False)


def test_porous_absorber_rejects_single_mesh_for_both_thicknesses(monkeypatch):
    module = load_example_module(EXAMPLE_MAIN)
    args = SimpleNamespace(
        thickness="both",
        mesh_path=Path("one_mesh.msh"),
        output_root=Path("unused"),
        order=1,
        nt=1,
        cfl=0.1,
        total_time=0.0,
        n_time_steps=0,
        save_step=0,
        save_mesh_step=0,
        save_mesh_at_ms=0.0,
        progress=False,
        fit_path=Path("unused.mat"),
        force_fit=False,
        force_mesh=False,
        sponge_sigma_max=0.0,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="--mesh-path"):
        module.main()
