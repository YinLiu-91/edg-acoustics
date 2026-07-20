"""Run the COMSOL car-cabin transient case with EDG acoustics.

The COMSOL model uses a zero initial state and an active normal-velocity
boundary source on ``TweeterLSource``.  The EDG reproduction therefore should
not use the older scenario-style monopole pressure initial condition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy
import scipy.io

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import edg_acoustics


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_MESH = CASE_DIR / "car_cabin_comsol_virtual_hmax0p114_hmin0p02.msh"
FALLBACK_MESH = CASE_DIR / "car_cabin_acoustics_transient_63_cleared_curv_hxt_lc0p12_min0p06.msh"

RHO0 = 1.2
C0 = 343.0
F0 = 1000.0
FMAX = 1500.0
T0 = 0.001
TEND = 0.06
OUTPUT_DT = T0 / 40.0

NX = 4
NT = 4
CFL = 0.5

# COMSOL result group ``pg12`` ("Microphone Response") plots ``pate.p_t`` at
# microphone-array point entities 197, 391, and 402, in this order.
COMSOL_MICROPHONE_POINT_IDS = numpy.array([197, 391, 402], dtype=numpy.int32)
RECEIVER = numpy.array(
    [
        [2.0, 2.5, 2.5],
        [-0.05, -0.55, 0.55],
        [1.2, 1.2, 1.2],
    ],
    dtype=float,
)

BC_LABELS = {
    "default_hard_wall": 11,
    "windows": 12,
    "dashboard": 13,
    "doors": 14,
    "leather_seats": 15,
    "carpet_floor": 16,
    "roof_trim": 17,
    "tweeter_l_source": 21,
    "inactive_speakers_hard_wall": 22,
}

MATERIAL_MAT_FILES = {
    "leather_seats": CASE_DIR / "seat.mat",
    "carpet_floor": CASE_DIR / "carpet.mat",
    "roof_trim": CASE_DIR / "roof.mat",
}


def constant_absorption_reflection(alpha: float) -> float:
    return float(numpy.sqrt(1.0 - alpha))


def load_edg_material(label: int, mat_path: Path) -> dict:
    if not mat_path.exists():
        raise FileNotFoundError(
            f"Missing fitted material file: {mat_path}. "
            "Run fit_car_cabin_admittance.m first."
        )

    mat = scipy.io.loadmat(mat_path)
    material = {"label": label}
    material["RI"] = float(numpy.asarray(mat.get("RI", [[0.0]])).reshape(-1)[0])

    if "AS" in mat and "lambdaS" in mat:
        material["RP"] = numpy.vstack(
            (
                numpy.asarray(mat["AS"]).reshape(-1),
                numpy.asarray(mat["lambdaS"]).reshape(-1),
            )
        )
    if {"BS", "CS", "alphaS", "betaS"} <= set(mat):
        material["CP"] = numpy.vstack(
            (
                numpy.asarray(mat["BS"]).reshape(-1),
                numpy.asarray(mat["CS"]).reshape(-1),
                numpy.asarray(mat["alphaS"]).reshape(-1),
                numpy.asarray(mat["betaS"]).reshape(-1),
            )
        )
    return material


def build_bc_parameters() -> list[dict]:
    params: dict[int, dict] = {
        11: {"label": 11, "RI": 1.0},
        12: {"label": 12, "RI": constant_absorption_reflection(0.005)},
        13: {"label": 13, "RI": constant_absorption_reflection(0.01)},
        14: {"label": 14, "RI": constant_absorption_reflection(0.01)},
        21: {
            "label": 21,
            "RI": 1.0,
            "normal_velocity": {
                "kind": "gaussian_modulated_sine",
                "amplitude": 1.0,
                "frequency": F0,
                "delay": 2.0 * T0,
                "sigma": 0.5 * T0,
                "phase": 0.0,
                "baseline": 0.0,
            },
        },
        22: {"label": 22, "RI": 1.0},
    }
    for name, mat_path in MATERIAL_MAT_FILES.items():
        params[BC_LABELS[name]] = load_edg_material(BC_LABELS[name], mat_path)
    return [params[label] for label in BC_LABELS.values()]


def resolve_mesh(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_MESH.exists():
        return DEFAULT_MESH
    return FALLBACK_MESH


def build_simulation(mesh_path: Path, *, nx: int = NX, nt: int = NT, cfl: float = CFL):
    mesh = edg_acoustics.Mesh(str(mesh_path), BC_LABELS)
    sim = edg_acoustics.AcousticsSimulation(RHO0, C0, nx, mesh, BC_LABELS)
    sim.init_BC(edg_acoustics.AbsorbBC(sim.BCnode, build_bc_parameters()))
    sim.init_IC(edg_acoustics.Zero_IC())
    sim.init_Flux(edg_acoustics.UpwindFlux(RHO0, C0, sim.n_xyz))
    sim.init_rec(RECEIVER, "scipy")
    sim.init_TimeIntegrator(edg_acoustics.TSI_TI(sim.RHS_operator, sim.dtscale, CFL=cfl, Nt=nt))
    return sim


def write_result(sim, output_path: Path) -> None:
    scipy.io.savemat(
        output_path,
        {
            "BCpara": sim.BC.BCpara,
            "prec": sim.prec.detach().cpu().numpy(),
            "prec_times": numpy.asarray(sim.prec_times, dtype=float),
            "rec": sim.rec,
            "receiver_point_ids": COMSOL_MICROPHONE_POINT_IDS,
            "dt": sim.time_integrator.dt,
            "Ntimesteps": sim.Ntimesteps,
            "total_time": sim.Ntimesteps * sim.time_integrator.dt,
            "Np": sim.Np,
            "N_tets": sim.N_tets,
            "rho0": RHO0,
            "c0": C0,
            "mesh_filename": sim.mesh.filename,
            "source_kind": "normal_velocity",
            "source_label": 21,
            "Nx": sim.Nx,
            "Nt": sim.time_integrator.Nt,
            "CFL": sim.time_integrator.CFL,
            "cuda_graph_mode": getattr(sim, "last_time_integration_cuda_graph_mode", "disabled"),
        },
    )


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=None, help="Gmsh .msh file")
    parser.add_argument("--total-time", type=positive_float, default=TEND)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500000,
        help="Abort if the estimated step count is larger; use 0 to disable.",
    )
    parser.add_argument("--output", type=Path, default=CASE_DIR / "result.mat")
    parser.add_argument("--save-step", type=int, default=0)
    parser.add_argument("--save-mesh-step", type=int, default=0)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graph-chunk-steps", type=int, default=1)
    parser.add_argument(
        "--no-comsol-output-times",
        action="store_true",
        help="Keep every EDG step instead of interpolating to COMSOL range(0,T0/40,Tend).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mesh_path = resolve_mesh(args.mesh)
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    sim = build_simulation(mesh_path)
    estimated_steps = int(numpy.floor(args.total_time / sim.time_integrator.dt))
    if args.max_steps and estimated_steps > args.max_steps:
        raise RuntimeError(
            f"Requested run would take {estimated_steps} steps. "
            f"Increase --max-steps or use a mesh with larger minimum element size."
        )

    output_times = None
    if not args.no_comsol_output_times:
        output_times = numpy.arange(0.0, args.total_time + 0.5 * OUTPUT_DT, OUTPUT_DT)

    sim.time_integration(
        total_time=args.total_time,
        delta_step=max(1, estimated_steps // 20) if args.progress else 0,
        save_step=args.save_step,
        save_results_dir=str(CASE_DIR),
        save_mesh_step=args.save_mesh_step,
        save_mesh_dir=str(CASE_DIR / "results_on_the_run_msh"),
        format="mat",
        progress=args.progress,
        output_times=output_times,
        use_cuda_graph=args.use_cuda_graph,
        cuda_graph_chunk_steps=args.cuda_graph_chunk_steps,
    )
    write_result(sim, args.output)
    print(f"Finished. Result written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
