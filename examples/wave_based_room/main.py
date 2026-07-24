"""Run the COMSOL wave-based room case with EDG acoustics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy
import scipy.io
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import edg_acoustics


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_MESH = CASE_DIR / "wave_based_room_comsol_tet_hmax0p163_hmin0p04.msh"
DEFAULT_RECEIVER_JSON = CASE_DIR / "wave_based_room_receiver_points.json"

RHO0 = 1.2
C0 = 343.0
F0 = 700.0
T0 = 1.0 / F0
TEND = 30.0 * T0
OUTPUT_DT = T0

NX = 4
NT = 4
CFL = 0.5

COMSOL_RECEIVER_POINT_IDS = numpy.array([122, 121, 53, 35], dtype=numpy.int32)

BC_LABELS = {
    "default_hard_wall": 11,
    "carpet": 12,
    "ceiling": 13,
    "sofa": 14,
    "wall": 15,
    "normal_velocity_source": 21,
}

MATERIAL_MAT_FILES = {
    "carpet": CASE_DIR / "carpet.mat",
    "ceiling": CASE_DIR / "ceiling.mat",
    "sofa": CASE_DIR / "sofa.mat",
    "wall": CASE_DIR / "wall.mat",
}


def load_edg_material(label: int, mat_path: Path) -> dict:
    if not mat_path.exists():
        raise FileNotFoundError(
            f"Missing fitted material file: {mat_path}. "
            "Run fit_wave_based_room_admittance.py first."
        )
    mat = scipy.io.loadmat(mat_path)
    target_source = "".join(
        numpy.asarray(mat.get("target_source", []), dtype=str).reshape(-1).tolist()
    )
    if target_source != "COMSOL partial-fraction admittance":
        raise ValueError(f"{mat_path} was not generated from COMSOL PFF admittance.")
    max_abs_r = float(numpy.asarray(mat.get("max_abs_R", [[numpy.inf]])).reshape(-1)[0])
    if not numpy.isfinite(max_abs_r) or max_abs_r > 1.0 + 1.0e-8:
        raise ValueError(f"{mat_path} is not passive: max_abs_R={max_abs_r}")

    material = {"label": label}
    material["RI"] = float(numpy.asarray(mat.get("RI", [[0.0]])).reshape(-1)[0])
    if "AS" in mat and "lambdaS" in mat:
        as_values = numpy.asarray(mat["AS"]).reshape(-1)
        lambda_values = numpy.asarray(mat["lambdaS"]).reshape(-1)
        if as_values.size:
            material["RP"] = numpy.vstack((as_values, lambda_values))
    if {"BS", "CS", "alphaS", "betaS"} <= set(mat):
        bs_values = numpy.asarray(mat["BS"]).reshape(-1)
        cs_values = numpy.asarray(mat["CS"]).reshape(-1)
        alpha_values = numpy.asarray(mat["alphaS"]).reshape(-1)
        beta_values = numpy.asarray(mat["betaS"]).reshape(-1)
        if bs_values.size:
            material["CP"] = numpy.vstack((bs_values, cs_values, alpha_values, beta_values))
    return material


def build_bc_parameters() -> list[dict]:
    params: dict[int, dict] = {
        11: {"label": 11, "RI": 1.0},
        21: {
            "label": 21,
            "RI": 1.0,
            "normal_velocity": {
                "kind": "gaussian_modulated_sine",
                "amplitude": 1.0,
                "frequency": F0,
                "delay": 2.0 * T0,
                "sigma": 0.5 * T0,
                "phase": numpy.pi,
                "baseline": 0.0,
            },
        },
    }
    for name, mat_path in MATERIAL_MAT_FILES.items():
        params[BC_LABELS[name]] = load_edg_material(BC_LABELS[name], mat_path)
    return [params[label] for label in BC_LABELS.values()]


def load_receivers(path: Path) -> numpy.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("coordinate_unit") != "m":
        raise ValueError("receiver coordinate_unit must be 'm'")
    point_ids = numpy.asarray(data["point_ids"], dtype=numpy.int32)
    if not numpy.array_equal(point_ids, COMSOL_RECEIVER_POINT_IDS):
        raise ValueError(
            f"receiver point_ids {point_ids.tolist()} do not match {COMSOL_RECEIVER_POINT_IDS.tolist()}"
        )
    coords = numpy.asarray(data["coords"], dtype=float)
    if coords.shape != (3, 4):
        raise ValueError(f"receiver coords must have shape (3, 4), got {coords.shape}")
    if not numpy.all(numpy.isfinite(coords)):
        raise ValueError("receiver coords must all be finite")
    return coords


def preflight_inputs(mesh_path: Path, receiver_json: Path) -> None:
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    if mesh_path.suffix.lower() != ".msh":
        raise ValueError(f"Expected a Gmsh .msh mesh, got {mesh_path}")
    load_receivers(receiver_json)
    for name, mat_path in MATERIAL_MAT_FILES.items():
        load_edg_material(BC_LABELS[name], mat_path)
    report_path = CASE_DIR / "wave_based_room_mesh_conversion_report.json"
    if mesh_path.resolve() == DEFAULT_MESH.resolve() and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report.get("boundary_validation", {}).get("ok", False):
            raise ValueError("Default mesh boundary validation did not pass")
        if not report.get("volume_validation", {}).get("ok", False):
            raise ValueError("Default mesh volume validation did not pass")
        topology = report.get("diagnostics", {}).get("boundary_topology", {})
        if topology and not topology.get("all_shells_are_exterior", False):
            raise ValueError("Default mesh contains non-exterior shell triangles")


def build_simulation(
    mesh_path: Path,
    receiver_json: Path,
    *,
    nx: int = NX,
    nt: int = NT,
    cfl: float = CFL,
    receiver_locate_method: str = "brute_force",
):
    mesh = edg_acoustics.Mesh(str(mesh_path), BC_LABELS)
    sim = edg_acoustics.AcousticsSimulation(RHO0, C0, nx, mesh, BC_LABELS)
    sim.init_BC(edg_acoustics.AbsorbBC(sim.BCnode, build_bc_parameters()))
    sim.init_IC(edg_acoustics.Zero_IC())
    sim.init_Flux(edg_acoustics.UpwindFlux(RHO0, C0, sim.n_xyz))
    sim.init_rec(load_receivers(receiver_json), receiver_locate_method)
    sim.init_TimeIntegrator(edg_acoustics.TSI_TI(sim.RHS_operator, sim.dtscale, CFL=cfl, Nt=nt))
    return sim


def comsol_output_times(total_time: float) -> numpy.ndarray:
    interval_count = int(numpy.floor(total_time / OUTPUT_DT + 1.0e-10))
    times = numpy.arange(interval_count + 1, dtype=float) * OUTPUT_DT
    if times.size and numpy.isclose(times[-1], total_time, rtol=0.0, atol=1.0e-12):
        times[-1] = total_time
    return times


def write_result(sim, output_path: Path, requested_total_time: float | None, final_field_max_abs: float) -> None:
    integration_end_time = sim.Ntimesteps * sim.time_integrator.dt
    scipy.io.savemat(
        output_path,
        {
            "BCpara": sim.BC.BCpara,
            "prec": sim.prec.detach().cpu().numpy(),
            "prec_times": numpy.asarray(sim.prec_times, dtype=float),
            "rec": sim.rec,
            "receiver_point_ids": COMSOL_RECEIVER_POINT_IDS,
            "dt": sim.time_integrator.dt,
            "Ntimesteps": sim.Ntimesteps,
            "total_time": integration_end_time,
            "integration_end_time": integration_end_time,
            "requested_total_time": numpy.nan if requested_total_time is None else requested_total_time,
            "final_field_max_abs": final_field_max_abs,
            "Np": sim.Np,
            "N_tets": sim.N_tets,
            "rho0": RHO0,
            "c0": C0,
            "mesh_filename": sim.mesh.filename,
            "source_kind": "normal_velocity",
            "source_label": 21,
            "source_frequency": F0,
            "source_delay": 2.0 * T0,
            "source_sigma": 0.5 * T0,
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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--receiver-json", type=Path, default=DEFAULT_RECEIVER_JSON)
    parser.add_argument(
        "--receiver-locate-method",
        choices=("scipy", "brute_force"),
        # The exported COMSOL tetrahedra are not guaranteed to form a
        # Delaunay triangulation, so use direct barycentric containment.
        default="brute_force",
    )
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--total-time", type=positive_float, default=TEND)
    duration.add_argument("--n-time-steps", type=positive_int, default=None)
    parser.add_argument("--nx", type=positive_int, default=NX)
    parser.add_argument("--nt", type=positive_int, default=NT)
    parser.add_argument("--cfl", type=positive_float, default=CFL)
    parser.add_argument("--max-steps", type=nonnegative_int, default=0)
    parser.add_argument("--output", type=Path, default=CASE_DIR / "result.mat")
    parser.add_argument("--save-step", type=nonnegative_int, default=0)
    parser.add_argument("--save-mesh-step", type=nonnegative_int, default=0)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-step", type=positive_int, default=None)
    parser.add_argument("--use-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graph-chunk-steps", type=int, default=1)
    parser.add_argument("--no-comsol-output-times", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stability-check-step", type=positive_int, default=1000)
    parser.add_argument("--max-field-abs", type=positive_float, default=1.0e6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preflight_inputs(args.mesh, args.receiver_json)
    sim = build_simulation(
        args.mesh,
        args.receiver_json,
        nx=args.nx,
        nt=args.nt,
        cfl=args.cfl,
        receiver_locate_method=args.receiver_locate_method,
    )
    requested_total_time = None if args.n_time_steps is not None else args.total_time
    if args.n_time_steps is not None:
        estimated_steps = args.n_time_steps
    else:
        estimated_steps = int(numpy.ceil(args.total_time / sim.time_integrator.dt))
    if args.max_steps and estimated_steps > args.max_steps:
        raise RuntimeError(
            f"Requested run would take {estimated_steps} steps. "
            "Increase --max-steps or use a mesh/order with larger time step."
        )

    output_times = None
    if not args.no_comsol_output_times and requested_total_time is not None:
        output_times = comsol_output_times(requested_total_time)

    integration_end_time = estimated_steps * sim.time_integrator.dt
    print(
        f"Preflight passed: Nx={args.nx}, Nt={args.nt}, CFL={args.cfl}, "
        f"dt={sim.time_integrator.dt:.17g}, steps={estimated_steps}, "
        f"integration_end={integration_end_time:.17g} s"
    )
    if requested_total_time is not None:
        print(
            f"Requested end={requested_total_time:.17g} s; "
            f"COMSOL output samples={0 if output_times is None else output_times.size}"
        )
    if args.dry_run:
        return 0

    sim.time_integration(
        n_time_steps=estimated_steps,
        delta_step=(
            args.progress_step
            if args.progress_step is not None
            else max(1, estimated_steps // 20)
        )
        if args.progress
        else 0,
        save_step=args.save_step,
        save_results_dir=str(CASE_DIR),
        save_mesh_step=args.save_mesh_step,
        save_mesh_dir=str(CASE_DIR / "results_on_the_run_msh"),
        format="mat",
        progress=args.progress,
        output_times=output_times,
        use_cuda_graph=args.use_cuda_graph,
        cuda_graph_chunk_steps=args.cuda_graph_chunk_steps,
        stability_check_step=args.stability_check_step,
        max_field_abs=args.max_field_abs,
    )
    if not bool(torch.isfinite(sim.Q_flat).all().item()):
        raise FloatingPointError("EDG field contains NaN or Inf after time integration")
    final_field_max_abs = float(torch.max(torch.abs(sim.Q_flat)).item())
    print(f"Final field is finite; max_abs={final_field_max_abs:.17g}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_result(sim, args.output, requested_total_time, final_field_max_abs)
    print(f"Finished. Result written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
