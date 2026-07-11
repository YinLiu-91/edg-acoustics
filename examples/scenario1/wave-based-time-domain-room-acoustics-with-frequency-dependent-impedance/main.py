"""Run the COMSOL wave-based room case with EDG Acoustics.

This script is derived from ``examples/scenario1/main.py`` but is local to the
COMSOL-exported room case. The COMSOL normal-velocity boundary source is
approximated with the existing EDG monopole initial condition, so no library
code is modified.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import modepy
import numpy
import scipy.io
import torch

os.environ.setdefault("EDG_ACOUSTICS_DEVICE", os.environ.get("WAVE_ROOM_DEVICE", "cpu"))


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import edg_acoustics

# Parameters from the COMSOL tutorial.
rho0 = 1.213
c0 = 343.0
f0 = 700.0
T0 = 1.0 / f0

BC_labels = {
    "hard_other": 11,
    "walls": 12,
    "carpet": 13,
    "ceiling": 14,
    "sofa": 15,
    "source": 16,
}

RIGID_MATERIALS = {"hard_other", "source"}
IMPEDANCE_MATERIALS = {"walls", "carpet", "ceiling", "sofa"}

mesh_name = os.environ.get("WAVE_ROOM_MESH", "room.msh")

# Center of COMSOL boundary 222 after STEP import/fragmentation.
monopole_xyz = numpy.array([2.32, 0.96, 1.1666666666666667])
freq_upper_limit = f0

# Approximation degrees.
Nx = 4
Nt = 3
CFL = 0.5

# COMSOL listening points LP1..LP4.
rec = numpy.array(
    [
        [1.2, 0.2, -0.8, -1.8],
        [0.75 * 1.75, 0.50 * 1.75, 0.25 * 1.75, 0.0],
        [1.0 + 1.0e-6, 1.0 + 1.0e-6, 1.0 + 1.0e-6, 1.0 + 1.0e-6],
    ],
    dtype=float,
)

impulse_length = float(os.environ.get("WAVE_ROOM_TOTAL_TIME", 30.0 * T0))
save_every_Nstep = int(os.environ.get("WAVE_ROOM_DELTA_STEP", "10"))
temporary_save_Nstep = int(os.environ.get("WAVE_ROOM_SAVE_STEP", "0"))
temporary_save_msh_Nstep = int(os.environ.get("WAVE_ROOM_SAVE_MSH_STEP", "0"))
temporary_save_msh_dir = SCRIPT_DIR / "results_on_the_run_msh"
result_filename = os.environ.get("WAVE_ROOM_RESULT_NAME", "wave_based_room_result")
skip_postprocess = os.environ.get("WAVE_ROOM_SKIP_POSTPROCESS", "0") != "0"


def _as_row_vector(mat_file: dict, key: str) -> numpy.ndarray:
    if key not in mat_file:
        return numpy.array([], dtype=float)
    values = numpy.asarray(mat_file[key], dtype=float).reshape(-1)
    return values[numpy.isfinite(values)]


def _load_fitted_material(material: str, label: int) -> dict:
    mat_files = sorted(glob.glob(str(SCRIPT_DIR / f"{material}*.mat")))
    if not mat_files:
        raise FileNotFoundError(
            f"No fitted .mat file found for material '{material}'. "
            "Run fit_wave_based_room_materials.m first."
        )

    mat_file = scipy.io.loadmat(mat_files[0])
    material_dict: dict[str, object] = {"label": label}

    ri = _as_row_vector(mat_file, "RI")
    material_dict["RI"] = float(ri[0]) if ri.size else 0.0

    real_residue = _as_row_vector(mat_file, "AS")
    real_pole = _as_row_vector(mat_file, "lambdaS")
    if real_residue.size and real_pole.size:
        n_real = min(real_residue.size, real_pole.size)
        material_dict["RP"] = numpy.vstack(
            (real_residue[:n_real], real_pole[:n_real])
        )

    complex_residue_re = _as_row_vector(mat_file, "BS")
    complex_residue_im = _as_row_vector(mat_file, "CS")
    complex_pole_re = _as_row_vector(mat_file, "alphaS")
    complex_pole_im = _as_row_vector(mat_file, "betaS")
    n_complex = min(
        complex_residue_re.size,
        complex_residue_im.size,
        complex_pole_re.size,
        complex_pole_im.size,
    )
    if n_complex:
        material_dict["CP"] = numpy.vstack(
            (
                complex_residue_re[:n_complex],
                complex_residue_im[:n_complex],
                complex_pole_re[:n_complex],
                complex_pole_im[:n_complex],
            )
        )

    if "RP" not in material_dict and "CP" not in material_dict:
        raise ValueError(f"Material '{material}' has no fitted pole data.")

    return material_dict


def _build_bc_parameters(bc_nodes: list[dict]) -> list[dict]:
    parameters_by_label = {}
    for material, label in BC_labels.items():
        if material in RIGID_MATERIALS:
            parameters_by_label[label] = {"label": label, "RI": 1.0}
        elif material in IMPEDANCE_MATERIALS:
            parameters_by_label[label] = _load_fitted_material(material, label)
        else:
            raise ValueError(f"Unhandled boundary material '{material}'.")

    return [parameters_by_label[int(node["label"])] for node in bc_nodes]


def _receiver_indices_are_valid(nodeindex: numpy.ndarray, n_tets: int) -> bool:
    indices = numpy.asarray(nodeindex)
    return bool(
        indices.ndim == 1
        and indices.size == rec.shape[1]
        and numpy.all(indices >= 0)
        and numpy.all(indices < n_tets)
    )


def _receivers_to_reference_coordinates(
    vertices: numpy.ndarray, e_to_v: numpy.ndarray, nodeindex: numpy.ndarray
) -> numpy.ndarray:
    rst = numpy.zeros_like(rec)
    for i, element_index in enumerate(nodeindex):
        tet_vertices = vertices[:, e_to_v[:, element_index]]
        jacobian = numpy.column_stack(
            (
                tet_vertices[:, 1] - tet_vertices[:, 0],
                tet_vertices[:, 2] - tet_vertices[:, 0],
                tet_vertices[:, 3] - tet_vertices[:, 0],
            )
        )
        barycentric_123 = numpy.linalg.solve(jacobian, rec[:, i] - tet_vertices[:, 0])
        rst[:, i] = 2.0 * barycentric_123 - 1.0
    return rst


def _init_receivers(sim: edg_acoustics.AcousticsSimulation) -> None:
    nodeindex = edg_acoustics.AcousticsSimulation.locate_simplex(
        sim.mesh.vertices,
        numpy.asarray(sim.mesh.EToV.cpu()),
        rec,
        "brute_force",
    )
    if not _receiver_indices_are_valid(nodeindex, sim.N_tets):
        raise RuntimeError(f"Failed to locate receiver points: {nodeindex}")

    rst_rec = _receivers_to_reference_coordinates(
        sim.mesh.vertices,
        numpy.asarray(sim.mesh.EToV.cpu()),
        nodeindex,
    )
    shape = modepy.Simplex(sim.dim)
    space = modepy.space_for_shape(shape, sim.Nx)
    simplex_basis = modepy.orthonormal_basis_for_space(space, shape).functions
    v_new = modepy.vandermonde(simplex_basis, rst_rec)
    sample_weight = v_new @ numpy.linalg.inv(sim.V.cpu().numpy())

    sim.rec = rec
    sim.sampleWeight = (
        torch.from_numpy(sample_weight)
        .to(device=sim.device, dtype=sim.P.dtype)
        .contiguous()
    )
    sim.nodeindex = nodeindex
    sim._nodeindex_tensor = torch.as_tensor(
        nodeindex, device=sim.device, dtype=torch.long
    )
    sim._sample_values = torch.empty(
        (sim.Np, sim.rec.shape[1]), device=sim.device, dtype=sim.P.dtype
    )
    sim._sample_output = torch.empty(
        (sim.rec.shape[1],), device=sim.device, dtype=sim.P.dtype
    )


def main() -> None:
    mesh_filename = SCRIPT_DIR / mesh_name
    mesh = edg_acoustics.Mesh(str(mesh_filename), BC_labels)

    ic = edg_acoustics.Monopole_IC(monopole_xyz, freq_upper_limit)
    sim = edg_acoustics.AcousticsSimulation(rho0, c0, Nx, mesh, BC_labels)

    bc_para = _build_bc_parameters(sim.BCnode)
    absorb_bc = edg_acoustics.AbsorbBC(sim.BCnode, bc_para, freq_max=1500)
    flux = edg_acoustics.UpwindFlux(rho0, c0, sim.n_xyz)

    sim.init_BC(absorb_bc)
    sim.init_IC(ic)
    sim.init_Flux(flux)
    _init_receivers(sim)

    if temporary_save_msh_Nstep > 0:
        temporary_save_msh_dir.mkdir(parents=True, exist_ok=True)

    time_integrator = edg_acoustics.TSI_TI(sim.RHS_operator, sim.dtscale, CFL, Nt=Nt)
    sim.init_TimeIntegrator(time_integrator)
    time_integration_kwargs = {
        "total_time": impulse_length,
        "format": "mat",
        "progress": os.environ.get("WAVE_ROOM_PROGRESS", "1") != "0",
        "use_cuda_graph": os.environ.get("WAVE_ROOM_CUDA_GRAPH", "1") != "0",
    }
    if save_every_Nstep > 0:
        time_integration_kwargs["delta_step"] = save_every_Nstep
    if temporary_save_Nstep > 0:
        time_integration_kwargs["save_step"] = temporary_save_Nstep
    if temporary_save_msh_Nstep > 0:
        time_integration_kwargs["save_mesh_step"] = temporary_save_msh_Nstep
        time_integration_kwargs["save_mesh_dir"] = str(temporary_save_msh_dir)

    sim.time_integration(**time_integration_kwargs)

    if not skip_postprocess:
        results = edg_acoustics.Monopole_postprocessor(sim, 1)
        results.apply_correction()
        results.write_results(str(SCRIPT_DIR / result_filename), "mat")
    print("Finished!")


if __name__ == "__main__":
    main()
