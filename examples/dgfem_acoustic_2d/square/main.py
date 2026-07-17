"""Minimal 2D acoustic square example using the TSI time integrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import edg_acoustics


CASE_DIR = Path(__file__).resolve().parent
RHO0 = 1.213
C0 = 343.0
# BC_LABELS = {"Absorbing": 11}
BC_LABELS = {"Wall": 11}
BC_PARA = [{"label": 11, "RI": 1.0}]
MONOPOLE_XYZ = numpy.array([0.0, -0.0, 0.0], dtype=numpy.float64)
FREQ_UPPER_LIMIT = 200.0
NX = 4
NT = 4
CFL = 0.5
TOTAL_TIME = 0.03
DEFAULT_SNAPSHOT = CASE_DIR / "square_snapshot.mat"
DEFAULT_MESH = CASE_DIR / "square.msh"
DEFAULT_RESULTS_ON_THE_RUN = CASE_DIR / "results_on_the_run.mat"
DEFAULT_RESULTS_ON_THE_RUN_MSH_DIR = CASE_DIR / "results_on_the_run_msh"
DEFAULT_USE_2D_PACKED_RHS = False
DEFAULT_USE_2D_TRITON_KERNELS = False
DEFAULT_USE_2D_DEEP_FUSED_RHS = True


def build_simulation(
    *,
    Nx: int = NX,
    Nt: int = NT,
    cfl: float = CFL,
    mesh_path: Path | None = None,
    use_packed_rhs: bool = DEFAULT_USE_2D_PACKED_RHS,
    use_triton_kernels: bool = DEFAULT_USE_2D_TRITON_KERNELS,
    use_triton_deep_rhs: bool = DEFAULT_USE_2D_DEEP_FUSED_RHS,
):
    """Build a fully initialized 2D acoustic simulation."""
    mesh = edg_acoustics.Mesh2D(
        str(DEFAULT_MESH if mesh_path is None else Path(mesh_path)),
        BC_LABELS,
    )
    sim = edg_acoustics.AcousticsSimulation2D(RHO0, C0, Nx, mesh, BC_LABELS)
    sim.configure_fast_paths(
        use_packed_rhs=use_packed_rhs,
        use_triton_kernels=use_triton_kernels,
        use_triton_deep_rhs=use_triton_deep_rhs,
    )
    sim.init_BC(edg_acoustics.AbsorbBC(sim.BCnode, BC_PARA))
    sim.init_IC(edg_acoustics.Monopole_IC(MONOPOLE_XYZ, FREQ_UPPER_LIMIT))
    sim.init_Flux(edg_acoustics.UpwindFlux(RHO0, C0, sim.n_xyz))
    sim.init_TimeIntegrator(edg_acoustics.TSI_TI(sim.RHS_operator, sim.dtscale, cfl, Nt=Nt))
    return sim


def run_case(
    *,
    Nx: int = NX,
    Nt: int = NT,
    cfl: float = CFL,
    n_time_steps: int | None = None,
    total_time: float = TOTAL_TIME,
    snapshot_path: Path | None = None,
    mesh_path: Path | None = None,
    save_step: int = 0,
    save_results_dir: Path | None = None,
    save_format: str = "mat",
    save_mesh_step: int = 0,
    save_mesh_dir: Path | None = None,
    save_mesh_format: str = "gmsh22",
    progress: bool = True,
    use_packed_rhs: bool = DEFAULT_USE_2D_PACKED_RHS,
    use_triton_kernels: bool = DEFAULT_USE_2D_TRITON_KERNELS,
    use_triton_deep_rhs: bool = DEFAULT_USE_2D_DEEP_FUSED_RHS,
):
    """Run the square case and write the final state snapshot."""
    sim = build_simulation(
        Nx=Nx,
        Nt=Nt,
        cfl=cfl,
        mesh_path=mesh_path,
        use_packed_rhs=use_packed_rhs,
        use_triton_kernels=use_triton_kernels,
        use_triton_deep_rhs=use_triton_deep_rhs,
    )
    if save_results_dir is None and save_step > 0:
        save_results_dir = CASE_DIR
    if save_mesh_dir is None and save_mesh_step > 0:
        save_mesh_dir = DEFAULT_RESULTS_ON_THE_RUN_MSH_DIR
    sim.time_integration(
        n_time_steps=n_time_steps,
        total_time=total_time if n_time_steps is None else None,
        save_step=save_step,
        save_results_dir=None if save_results_dir is None else str(save_results_dir),
        format=save_format,
        save_mesh_step=save_mesh_step,
        save_mesh_dir=None if save_mesh_dir is None else str(save_mesh_dir),
        save_mesh_format=save_mesh_format,
        progress=progress,
    )
    snapshot = DEFAULT_SNAPSHOT if snapshot_path is None else Path(snapshot_path)
    sim.write_snapshot(snapshot)
    return snapshot, sim


def parse_args():
    """Parse command line arguments for the minimal 2D example."""
    parser = argparse.ArgumentParser(description="Run the minimal 2D acoustic square case.")
    parser.add_argument("--order", type=int, default=NX, help="DG polynomial order.")
    parser.add_argument("--nt", type=int, default=NT, help="TSI Taylor order.")
    parser.add_argument("--cfl", type=float, default=CFL, help="CFL number.")
    parser.add_argument(
        "--n-time-steps",
        type=int,
        default=None,
        help="Run a fixed number of time steps.",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=TOTAL_TIME,
        help="Total simulated time when --n-time-steps is not set.",
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Path of the final .mat or .npz snapshot.",
    )
    parser.add_argument(
        "--mesh-path",
        type=Path,
        default=DEFAULT_MESH,
        help="Path to the Gmsh triangle mesh.",
    )
    parser.add_argument(
        "--save-step",
        type=int,
        default=0,
        help="Write temporary results_on_the_run every N time steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--save-results-dir",
        type=Path,
        default=None,
        help="Output directory for temporary results_on_the_run files.",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        default="mat",
        help="Temporary results_on_the_run format: mat or npy.",
    )
    parser.add_argument(
        "--save-mesh-step",
        type=int,
        default=0,
        help="Export one Gmsh field snapshot every N time steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--save-mesh-dir",
        type=Path,
        default=None,
        help="Output directory for temporary Gmsh field snapshots.",
    )
    parser.add_argument(
        "--save-mesh-format",
        type=str,
        default="gmsh22",
        help="MeshIO format name used for temporary field snapshots.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print periodic step progress.",
    )
    parser.add_argument(
        "--use-2d-packed-rhs",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_2D_PACKED_RHS,
        help="Enable the 2D packed RHS/time-integration fast path.",
    )
    parser.add_argument(
        "--use-2d-triton-kernels",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_2D_TRITON_KERNELS,
        help="Enable optional 2D Triton flux/boundary kernels.",
    )
    parser.add_argument(
        "--use-2d-deep-fused-rhs",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_2D_DEEP_FUSED_RHS,
        help="Enable the deeper 2D Triton timestep fusion path when packed RHS is enabled.",
    )
    return parser.parse_args()


def main():
    """Entry point for the minimal 2D acoustic example."""
    args = parse_args()
    snapshot_path, _ = run_case(
        Nx=args.order,
        Nt=args.nt,
        cfl=args.cfl,
        n_time_steps=args.n_time_steps,
        total_time=args.total_time,
        snapshot_path=args.snapshot_path,
        mesh_path=args.mesh_path,
        save_step=args.save_step,
        save_results_dir=args.save_results_dir,
        save_format=args.save_format,
        save_mesh_step=args.save_mesh_step,
        save_mesh_dir=args.save_mesh_dir,
        save_mesh_format=args.save_mesh_format,
        progress=args.progress,
        use_packed_rhs=args.use_2d_packed_rhs,
        use_triton_kernels=args.use_2d_triton_kernels,
        use_triton_deep_rhs=args.use_2d_deep_fused_rhs,
    )
    print(f"Wrote 2D acoustic snapshot to {snapshot_path}")


if __name__ == "__main__":
    main()
