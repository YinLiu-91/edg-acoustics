"""Run the COMSOL office-space initial-pulse case with EDG acoustics."""

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
import edg_acoustics.device_ini as device_ini


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_MESH = CASE_DIR / "office_space_comsol_mesh1.msh"
DEFAULT_RECEIVER_JSON = CASE_DIR / "office_space_receiver_points.json"

RHO0 = 1.2
C0 = 343.0
FC = 500.0
F0 = 750.0
T0 = 1.0 / F0
TEND = 0.4
OUTPUT_DT = T0 / 30.0

SOURCE_XYZ = numpy.array([4.0, 7.0, 1.5], dtype=float)
SOURCE_HALFWIDTH = math.sqrt(2.0) / (FC * 2.0 * math.pi / C0)
SOURCE_AMPLITUDE = 1.0

NX = 4
NT = 4
CFL = 0.1
MAX_VALIDATED_CFL = 0.1

COMSOL_RECEIVER_POINT_IDS = numpy.array([230, 233, 467], dtype=numpy.int32)

BC_LABELS = {
    "default_hard_wall": 11,
    "closed_windows": 12,
    "doors": 13,
    "brick_wall": 14,
    "carpet": 15,
    "ceiling": 16,
    "gypsum": 17,
    "open_window_absorbing_layer": 18,
}

MATERIAL_MAT_FILES = {
    "carpet": CASE_DIR / "carpet.mat",
    "ceiling": CASE_DIR / "ceiling.mat",
    "gypsum": CASE_DIR / "gypsum.mat",
}


class OfficeGaussianPressureIC(edg_acoustics.InitialCondition):
    """COMSOL p0(x,y,z) initial pressure with zero initial velocity."""

    def __init__(self, source_xyz: numpy.ndarray, halfwidth: float, amplitude: float):
        self.source_xyz = torch.as_tensor(source_xyz, dtype=device_ini.dtype, device=device_ini.device)
        self.halfwidth = float(halfwidth)
        self.amplitude = float(amplitude)
        self.metadata = {
            "kind": "office_space_gaussian_pressure",
            "source_xyz": source_xyz.tolist(),
            "halfwidth": self.halfwidth,
            "amplitude": self.amplitude,
            "expression": "S0*exp(-log(2)*((x-xs)^2+(y-ys)^2+(z-zs)^2)/B^2)",
        }

    def Pinit(self, xyz: torch.Tensor):
        radius_squared = (
            (xyz[0] - self.source_xyz[0]) ** 2
            + (xyz[1] - self.source_xyz[1]) ** 2
            + (xyz[2] - self.source_xyz[2]) ** 2
        )
        return (
            self.amplitude
            * torch.exp(-math.log(2.0) * radius_squared / (self.halfwidth**2))
        ).to(device=xyz.device, dtype=device_ini.dtype)

    def VXinit(self, xyz: torch.Tensor):
        return torch.zeros([xyz.shape[1], xyz.shape[2]], device=xyz.device, dtype=device_ini.dtype)

    def VYinit(self, xyz: torch.Tensor):
        return torch.zeros([xyz.shape[1], xyz.shape[2]], device=xyz.device, dtype=device_ini.dtype)

    def VZinit(self, xyz: torch.Tensor):
        return torch.zeros([xyz.shape[1], xyz.shape[2]], device=xyz.device, dtype=device_ini.dtype)


def constant_absorption_reflection(alpha: float) -> float:
    return float(numpy.sqrt(1.0 - alpha))


def load_edg_material(label: int, mat_path: Path) -> dict:
    if not mat_path.exists():
        raise FileNotFoundError(
            f"Missing fitted material file: {mat_path}. Run fit_office_space_admittance.m first."
        )
    mat = scipy.io.loadmat(mat_path)
    target_source = "".join(
        numpy.asarray(mat.get("target_source", []), dtype=str).reshape(-1).tolist()
    )
    if target_source != "COMSOL partial-fraction admittance":
        raise ValueError(
            f"{mat_path} was not fitted from the COMSOL partial-fraction admittance. "
            "Re-run fit_office_space_admittance.m."
        )
    rms_error = float(numpy.asarray(mat.get("rms_error", [[numpy.inf]])).reshape(-1)[0])
    max_abs_r = float(numpy.asarray(mat.get("max_abs_R", [[numpy.inf]])).reshape(-1)[0])
    if not numpy.isfinite(rms_error) or not numpy.isfinite(max_abs_r):
        raise ValueError(f"{mat_path} contains non-finite fit diagnostics")
    if max_abs_r > 1.0 + 1.0e-8:
        raise ValueError(f"{mat_path} is not passive: max_abs_R={max_abs_r}")
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
        13: {"label": 13, "RI": constant_absorption_reflection(0.04)},
        14: {"label": 14, "RI": constant_absorption_reflection(0.01)},
        18: {"label": 18, "RI": 0.0},
    }
    for name, mat_path in MATERIAL_MAT_FILES.items():
        params[BC_LABELS[name]] = load_edg_material(BC_LABELS[name], mat_path)
    return [params[label] for label in BC_LABELS.values()]


def load_receivers(path: Path) -> numpy.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing receiver coordinate file: {path}. "
            "Export the COMSOL point coordinates before running EDG, then store "
            "{\"point_ids\":[230,233,467],\"coords\":[[x1,x2,x3],[y1,y2,y3],[z1,z2,z3]]}."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("coordinate_unit") != "m":
        raise ValueError("receiver coordinate_unit must be 'm'")
    if data.get("source") != "COMSOL comp1/geom1.getVertexCoord()":
        raise ValueError("receiver coordinates must come from COMSOL comp1/geom1.getVertexCoord()")
    point_ids = numpy.asarray(data["point_ids"], dtype=numpy.int32)
    if not numpy.array_equal(point_ids, COMSOL_RECEIVER_POINT_IDS):
        raise ValueError(f"receiver point_ids {point_ids.tolist()} do not match [230, 233, 467]")
    coords = numpy.asarray(data["coords"], dtype=float)
    if coords.shape != (3, 3):
        raise ValueError(f"receiver coords must have shape (3, 3), got {coords.shape}")
    if not numpy.all(numpy.isfinite(coords)):
        raise ValueError("receiver coords must all be finite")
    if numpy.unique(coords, axis=1).shape[1] != COMSOL_RECEIVER_POINT_IDS.size:
        raise ValueError("receiver coordinates must identify three distinct points")
    return coords


def preflight_inputs(mesh_path: Path, receiver_json: Path) -> None:
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    if mesh_path.suffix.lower() != ".msh":
        raise ValueError(f"Expected a Gmsh .msh mesh, got {mesh_path}")
    load_receivers(receiver_json)
    for name, mat_path in MATERIAL_MAT_FILES.items():
        load_edg_material(BC_LABELS[name], mat_path)

    if mesh_path.resolve() == DEFAULT_MESH.resolve():
        report_path = CASE_DIR / "office_space_mesh_conversion_report.json"
        if not report_path.exists():
            raise FileNotFoundError(
                f"Missing mesh conversion report: {report_path}. Re-run the NASTRAN converter."
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        topology = report.get("diagnostics", {}).get("boundary_topology", {})
        if not topology.get("all_shells_are_exterior", False):
            raise ValueError("Default mesh contains non-exterior shell triangles")
        if not report.get("boundary_validation", {}).get("ok", False):
            raise ValueError("Default mesh boundary validation did not pass")
        if not report.get("volume_validation", {}).get("ok", False):
            raise ValueError("Default mesh volume validation did not pass")


def build_simulation(
    mesh_path: Path,
    receiver_json: Path,
    *,
    nx: int = NX,
    nt: int = NT,
    cfl: float = CFL,
):
    mesh = edg_acoustics.Mesh(str(mesh_path), BC_LABELS)
    sim = edg_acoustics.AcousticsSimulation(RHO0, C0, nx, mesh, BC_LABELS)
    sim.init_BC(edg_acoustics.AbsorbBC(sim.BCnode, build_bc_parameters()))
    sim.init_IC(OfficeGaussianPressureIC(SOURCE_XYZ, SOURCE_HALFWIDTH, SOURCE_AMPLITUDE))
    sim.init_Flux(edg_acoustics.UpwindFlux(RHO0, C0, sim.n_xyz))
    # These COMSOL point entities lie exactly on mesh vertices/interfaces.
    # Barycentric containment with a tolerance is deterministic for that case.
    sim.init_rec(load_receivers(receiver_json), "brute_force")
    sim.init_TimeIntegrator(edg_acoustics.TSI_TI(sim.RHS_operator, sim.dtscale, CFL=cfl, Nt=nt))
    return sim


def write_result(
    sim,
    output_path: Path,
    requested_total_time: float | None,
    final_field_max_abs: float,
) -> None:
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
            "requested_total_time": (
                numpy.nan if requested_total_time is None else requested_total_time
            ),
            "final_field_max_abs": final_field_max_abs,
            "Np": sim.Np,
            "N_tets": sim.N_tets,
            "rho0": RHO0,
            "c0": C0,
            "mesh_filename": sim.mesh.filename,
            "source_kind": "initial_pressure_gaussian",
            "source_xyz": SOURCE_XYZ,
            "source_halfwidth": SOURCE_HALFWIDTH,
            "source_amplitude": SOURCE_AMPLITUDE,
            "Nx": sim.Nx,
            "Nt": sim.time_integrator.Nt,
            "CFL": sim.time_integrator.CFL,
            "absorbing_layer_treatment": "outer matched boundary RI=0; no 3D COMSOL AL volume scaling",
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


def comsol_output_times(total_time: float) -> numpy.ndarray:
    interval_count = int(numpy.floor(total_time / OUTPUT_DT + 1.0e-10))
    times = numpy.arange(interval_count + 1, dtype=float) * OUTPUT_DT
    if numpy.isclose(times[-1], total_time, rtol=0.0, atol=1.0e-12):
        times[-1] = total_time
    return times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--receiver-json", type=Path, default=DEFAULT_RECEIVER_JSON)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--total-time", type=positive_float, default=TEND)
    duration.add_argument(
        "--n-time-steps",
        type=positive_int,
        default=None,
        help="Run an exact number of EDG steps; intended for short validation runs.",
    )
    parser.add_argument("--nx", type=positive_int, default=NX)
    parser.add_argument("--nt", type=positive_int, default=NT)
    parser.add_argument("--cfl", type=positive_float, default=CFL)
    parser.add_argument(
        "--allow-unsafe-cfl",
        action="store_true",
        help="Allow CFL values above the case-validated limit of 0.1.",
    )
    parser.add_argument(
        "--max-steps",
        type=nonnegative_int,
        default=0,
        help="Abort if the estimated step count is larger; use 0 to disable.",
    )
    parser.add_argument("--output", type=Path, default=CASE_DIR / "result.mat")
    parser.add_argument("--save-step", type=nonnegative_int, default=0)
    parser.add_argument("--save-mesh-step", type=nonnegative_int, default=0)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--progress-step",
        type=positive_int,
        default=None,
        help="Print progress after every N completed steps; default prints about 20 times.",
    )
    parser.add_argument("--use-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graph-chunk-steps", type=int, default=1)
    parser.add_argument(
        "--no-comsol-output-times",
        action="store_true",
        help="Keep every EDG step instead of interpolating to COMSOL range(0,T0/30,T_ir).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize the complete EDG case and report its time-step plan without integrating.",
    )
    parser.add_argument(
        "--stability-check-step",
        type=positive_int,
        default=1000,
        help="Check the full EDG field for divergence every N steps.",
    )
    parser.add_argument(
        "--max-field-abs",
        type=positive_float,
        default=1.0e6,
        help="Abort when the full-field absolute maximum exceeds this value.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cfl > MAX_VALIDATED_CFL and not args.allow_unsafe_cfl:
        raise ValueError(
            f"CFL={args.cfl} exceeds the validated office-case limit "
            f"{MAX_VALIDATED_CFL}. Use --cfl 0.1, or add --allow-unsafe-cfl "
            "only for explicit stability experiments."
        )
    preflight_inputs(args.mesh, args.receiver_json)

    sim = build_simulation(
        args.mesh,
        args.receiver_json,
        nx=args.nx,
        nt=args.nt,
        cfl=args.cfl,
    )
    requested_total_time = None if args.n_time_steps is not None else args.total_time
    if args.n_time_steps is not None:
        estimated_steps = args.n_time_steps
    else:
        estimated_steps = int(numpy.ceil(args.total_time / sim.time_integrator.dt))
    if args.max_steps and estimated_steps > args.max_steps:
        raise RuntimeError(
            f"Requested run would take {estimated_steps} steps. "
            "Increase --max-steps or use a mesh with larger minimum element size."
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
