"""Extended-reaction porous absorber reproduction based on the 2D DG solver."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import meshio
import numpy

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import edg_acoustics


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_GEO = CASE_DIR / "porous_absorber_time_domain.geo"
DEFAULT_FIT = CASE_DIR / "er_material_fit.mat"
DEFAULT_OUTPUT_ROOT = CASE_DIR / "outputs"
DEFAULT_TOTAL_TIME = 0.01
DEFAULT_SAVE_MESH_AT_MS = 5.5
DEFAULT_USE_CUDA_GRAPH = True
DEFAULT_CUDA_GRAPH_CHUNK_STEPS = 1
DEFAULT_USE_2D_PACKED_RHS = True
DEFAULT_USE_2D_TRITON_KERNELS = True
DEFAULT_USE_2D_DEEP_FUSED_RHS = True
DEFAULT_USE_2D_PARTITIONED_ER_RHS = True
DEFAULT_ER_RHS_BACKEND = "auto"

RHO0 = 1.213
C0 = 343.0
L0 = 1.5
PML_THICKNESS = L0 / 5.0
SPONGE_THICKNESS = PML_THICKNESS
PHYSICAL_XMIN = -1.5 * L0
PHYSICAL_XMAX = 1.5 * L0
PHYSICAL_YMAX = L0
SOURCE_XYZ = numpy.array([-1.0, 0.5, 0.0], dtype=numpy.float64)
RECEIVER_XYZ = numpy.array([[1.0], [0.5], [0.0]], dtype=numpy.float64)
PULSE_B = 0.045
DEFAULT_ORDER = 4
DEFAULT_NT = 4
DEFAULT_CFL = 0.25
DEFAULT_ABSORBING_LAYER = "pml"
DEFAULT_PML_AMP_SIGMA = 1000.0
DEFAULT_PML_PROFILE = "quadratic"
DEFAULT_SPONGE_SIGMA_MAX = 2500.0
DOMAIN_LABELS = {"Air": 1, "Porous": 2, "PML": 3}
BC_LABELS = {"Outer": 11, "Rigid": 12}
BC_PARA = [{"label": 11, "RI": 0.0}, {"label": 12, "RI": 1.0}]
EXPECTED_TRIANGLE_LABELS = set(DOMAIN_LABELS.values())
EXPECTED_LINE_LABELS = set(BC_LABELS.values())


def thickness_tag(thickness: float):
    return f"{int(round(thickness * 100.0))}cm"


def default_mesh_path(thickness: float):
    return CASE_DIR / f"porous_absorber_time_domain_{thickness_tag(thickness)}.msh"


def default_snapshot_path(output_dir: Path):
    return output_dir / "snapshot.mat"


def ensure_material_fit(
    *,
    fit_path: Path = DEFAULT_FIT,
    force: bool = False,
    octave_bin: str = "octave",
):
    if fit_path.exists() and not force:
        return fit_path

    function_call = (
        f"addpath('{CASE_DIR.as_posix()}'); "
        f"fit_er_material("
        f"'{(CASE_DIR / 'porous_absorber_time_domain_compressibility.zh_CN.txt').as_posix()}',"
        f"'{(CASE_DIR / 'porous_absorber_time_domain_density.zh_CN.txt').as_posix()}',"
        f"'{fit_path.as_posix()}');"
    )
    subprocess.run(
        [octave_bin, "--quiet", "--eval", function_call],
        cwd=CASE_DIR,
        check=True,
    )
    return fit_path


def ensure_mesh(
    thickness: float,
    *,
    geo_path: Path = DEFAULT_GEO,
    mesh_path: Path | None = None,
    force: bool = False,
    gmsh_bin: str = "gmsh",
):
    resolved_mesh_path = default_mesh_path(thickness) if mesh_path is None else Path(mesh_path)
    if resolved_mesh_path.exists() and not force:
        validate_mesh(resolved_mesh_path, thickness)
        return resolved_mesh_path

    resolved_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            gmsh_bin,
            "-2",
            geo_path.name,
            "-setnumber",
            "w0",
            str(float(thickness)),
            "-format",
            "msh2",
            "-o",
            str(resolved_mesh_path),
        ],
        cwd=geo_path.parent,
        check=True,
    )
    validate_mesh(resolved_mesh_path, thickness)
    return resolved_mesh_path


def validate_mesh(mesh_path: Path, thickness: float):
    mesh = meshio.read(mesh_path)
    if "triangle" not in mesh.cells_dict:
        raise ValueError(f"{mesh_path} does not contain triangle cells.")
    if "line" not in mesh.cells_dict:
        raise ValueError(f"{mesh_path} does not contain line cells.")

    physical_data = mesh.cell_data_dict.get("gmsh:physical", {})
    triangle_labels = set(map(int, numpy.unique(physical_data["triangle"])))
    line_labels = set(map(int, numpy.unique(physical_data["line"])))
    if triangle_labels != EXPECTED_TRIANGLE_LABELS:
        raise ValueError(
            f"{mesh_path} has triangle labels {sorted(triangle_labels)}, "
            f"expected {sorted(EXPECTED_TRIANGLE_LABELS)}."
        )
    if line_labels != EXPECTED_LINE_LABELS:
        raise ValueError(
            f"{mesh_path} has line labels {sorted(line_labels)}, "
            f"expected {sorted(EXPECTED_LINE_LABELS)}."
        )

    points = numpy.asarray(mesh.points)
    y_min = float(points[:, 1].min())
    expected_y_min = -float(thickness)
    tolerance = 1.0e-8
    if abs(y_min - expected_y_min) > tolerance:
        raise ValueError(
            f"{mesh_path} has ymin={y_min:g}, expected {expected_y_min:g} "
            f"for thickness={thickness:g}."
        )


def build_simulation(
    *,
    thickness: float,
    fit_path: Path,
    mesh_path: Path,
    Nx: int = DEFAULT_ORDER,
    Nt: int = DEFAULT_NT,
    cfl: float = DEFAULT_CFL,
    absorbing_layer: str = DEFAULT_ABSORBING_LAYER,
    pml_amp_sigma: float = DEFAULT_PML_AMP_SIGMA,
    pml_profile: str = DEFAULT_PML_PROFILE,
    sponge_sigma_max: float = DEFAULT_SPONGE_SIGMA_MAX,
    use_packed_rhs: bool = DEFAULT_USE_2D_PACKED_RHS,
    use_triton_kernels: bool = DEFAULT_USE_2D_TRITON_KERNELS,
    use_triton_deep_rhs: bool = DEFAULT_USE_2D_DEEP_FUSED_RHS,
    use_triton_partitioned_er_rhs: bool = DEFAULT_USE_2D_PARTITIONED_ER_RHS,
    er_rhs_backend: str = DEFAULT_ER_RHS_BACKEND,
):
    mesh = edg_acoustics.Mesh2D(str(mesh_path), BC_LABELS, DOMAIN_LABELS)
    material_fit = edg_acoustics.ExtendedReactionMaterialFit.from_mat(fit_path)
    absorbing_layer = str(absorbing_layer).strip().lower()
    pml_region = None
    pml_damping = None
    if absorbing_layer == "pml":
        pml_region = edg_acoustics.PMLRegion(
            (PHYSICAL_XMAX, PHYSICAL_YMAX),
            mode="ground",
            region_name="PML",
        )
        pml_damping = edg_acoustics.PMLDamping(
            amp_sigma=pml_amp_sigma,
            profile=pml_profile,
        )
    sim = edg_acoustics.ExtendedReactionSimulation2D(
        RHO0,
        C0,
        Nx,
        mesh,
        BC_LABELS,
        DOMAIN_LABELS,
        material_fit,
        physical_bbox=(PHYSICAL_XMIN, PHYSICAL_XMAX, -float(thickness), PHYSICAL_YMAX),
        sponge_thickness=SPONGE_THICKNESS,
        sponge_sigma_max=sponge_sigma_max,
        absorbing_layer=absorbing_layer,
        pml_region=pml_region,
        pml_damping=pml_damping,
    )
    sim.configure_fast_paths(
        use_packed_rhs=use_packed_rhs,
        use_triton_kernels=use_triton_kernels,
        use_triton_deep_rhs=use_triton_deep_rhs,
        use_triton_partitioned_er_rhs=use_triton_partitioned_er_rhs,
        er_rhs_backend=er_rhs_backend,
    )
    sim.init_BC(edg_acoustics.AbsorbBC(sim.BCnode, BC_PARA))
    sim.init_IC(edg_acoustics.RadialPressurePulse2D_IC(SOURCE_XYZ, PULSE_B))
    sim.init_rec(RECEIVER_XYZ)
    sim.init_TimeIntegrator(edg_acoustics.TSI_TI(sim.RHS_operator, sim.dtscale, cfl, Nt=Nt))
    sim.extra_results_metadata.update(
        {
            "thickness_m": float(thickness),
            "mesh_filename": str(mesh_path),
            "fit_filename": str(fit_path),
            "source_xyz": SOURCE_XYZ,
            "receiver_xyz": RECEIVER_XYZ,
            "absorbing_layer": absorbing_layer,
            "pml_amp_sigma": float(pml_amp_sigma),
            "pml_profile": str(pml_profile),
            "sponge_sigma_max": float(sponge_sigma_max),
        }
    )
    return sim


def run_case(
    *,
    thickness: float,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    Nx: int = DEFAULT_ORDER,
    Nt: int = DEFAULT_NT,
    cfl: float = DEFAULT_CFL,
    total_time: float = DEFAULT_TOTAL_TIME,
    n_time_steps: int | None = None,
    save_step: int = 0,
    save_mesh_step: int = 0,
    save_mesh_at_ms: float = DEFAULT_SAVE_MESH_AT_MS,
    progress: bool = True,
    fit_path: Path = DEFAULT_FIT,
    force_fit: bool = False,
    mesh_path: Path | None = None,
    force_mesh: bool = False,
    absorbing_layer: str = DEFAULT_ABSORBING_LAYER,
    pml_amp_sigma: float = DEFAULT_PML_AMP_SIGMA,
    pml_profile: str = DEFAULT_PML_PROFILE,
    sponge_sigma_max: float = DEFAULT_SPONGE_SIGMA_MAX,
    use_cuda_graph: bool = DEFAULT_USE_CUDA_GRAPH,
    cuda_graph_chunk_steps: int = DEFAULT_CUDA_GRAPH_CHUNK_STEPS,
    use_packed_rhs: bool = DEFAULT_USE_2D_PACKED_RHS,
    use_triton_kernels: bool = DEFAULT_USE_2D_TRITON_KERNELS,
    use_triton_deep_rhs: bool = DEFAULT_USE_2D_DEEP_FUSED_RHS,
    use_triton_partitioned_er_rhs: bool = DEFAULT_USE_2D_PARTITIONED_ER_RHS,
    er_rhs_backend: str = DEFAULT_ER_RHS_BACKEND,
):
    fit_path = ensure_material_fit(fit_path=fit_path, force=force_fit)
    mesh_path = ensure_mesh(thickness, mesh_path=mesh_path, force=force_mesh)
    sim = build_simulation(
        thickness=thickness,
        fit_path=fit_path,
        mesh_path=mesh_path,
        Nx=Nx,
        Nt=Nt,
        cfl=cfl,
        absorbing_layer=absorbing_layer,
        pml_amp_sigma=pml_amp_sigma,
        pml_profile=pml_profile,
        sponge_sigma_max=sponge_sigma_max,
        use_packed_rhs=use_packed_rhs,
        use_triton_kernels=use_triton_kernels,
        use_triton_deep_rhs=use_triton_deep_rhs,
        use_triton_partitioned_er_rhs=use_triton_partitioned_er_rhs,
        er_rhs_backend=er_rhs_backend,
    )

    output_dir = Path(output_root) / thickness_tag(thickness)
    mesh_output_dir = output_dir / "results_on_the_run_msh"
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_output_dir.mkdir(parents=True, exist_ok=True)

    exact_mesh_steps: list[int] = []
    if save_mesh_at_ms > 0.0:
        mesh_step = int(round((save_mesh_at_ms * 1.0e-3) / sim.time_integrator.dt))
        if mesh_step > 0:
            exact_mesh_steps.append(mesh_step)

    sim.extra_results_metadata.update(
        {
            "use_cuda_graph_requested": bool(use_cuda_graph),
            "cuda_graph_chunk_steps_requested": int(cuda_graph_chunk_steps),
            "use_packed_rhs_requested": bool(use_packed_rhs),
            "use_triton_kernels_requested": bool(use_triton_kernels),
            "use_triton_deep_rhs_requested": bool(use_triton_deep_rhs),
            "use_triton_partitioned_er_rhs_requested": bool(
                use_triton_partitioned_er_rhs
            ),
            "er_rhs_backend_requested": str(er_rhs_backend),
            "absorbing_layer_requested": str(absorbing_layer),
            "pml_amp_sigma_requested": float(pml_amp_sigma),
            "pml_profile_requested": str(pml_profile),
            "sponge_sigma_max_requested": float(sponge_sigma_max),
        }
    )
    sim.time_integration(
        n_time_steps=n_time_steps,
        total_time=total_time if n_time_steps is None else None,
        save_step=save_step,
        save_results_dir=str(output_dir),
        save_mesh_step=save_mesh_step,
        save_mesh_steps=exact_mesh_steps,
        save_mesh_dir=str(mesh_output_dir),
        progress=progress,
        use_cuda_graph=use_cuda_graph,
        cuda_graph_chunk_steps=cuda_graph_chunk_steps,
    )
    sim.extra_results_metadata.update(
        {
            "use_cuda_graph": bool(
                getattr(sim, "last_time_integration_used_cuda_graph", False)
            ),
            "cuda_graph_mode": getattr(
                sim, "last_time_integration_cuda_graph_mode", "disabled"
            ),
            "cuda_graph_chunk_steps": int(
                getattr(sim, "last_time_integration_cuda_graph_chunk_steps", 0)
            ),
            "use_packed_rhs": bool(
                getattr(sim, "last_time_integration_used_packed_rhs", False)
            ),
            "use_triton_interior_flux": bool(
                getattr(sim, "_use_triton_interior_flux", False)
            ),
            "use_triton_boundary_ri": bool(
                getattr(sim, "_use_triton_boundary_ri", False)
            ),
            "use_triton_deep_rhs": bool(
                getattr(sim, "_use_triton_deep_rhs", False)
            ),
            "use_triton_partitioned_er_rhs": bool(
                getattr(sim, "_use_triton_partitioned_er_rhs", False)
            ),
            "er_rhs_backend": str(
                getattr(sim, "_er_rhs_backend_active", DEFAULT_ER_RHS_BACKEND)
            ),
            "er_rhs_backend_fallback_reason": str(
                getattr(sim, "_er_rhs_backend_fallback_reason", "")
            ),
        }
    )
    sim.save_results_on_the_run(output_dir=output_dir, format="mat", step_index=sim.Ntimesteps)
    snapshot_path = default_snapshot_path(output_dir)
    sim.write_snapshot(snapshot_path)
    return {
        "thickness": thickness,
        "fit_path": fit_path,
        "mesh_path": mesh_path,
        "output_dir": output_dir,
        "mesh_output_dir": mesh_output_dir,
        "snapshot_path": snapshot_path,
    }, sim


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 2D ER porous absorber case.")
    parser.add_argument(
        "--thickness",
        type=str,
        default="both",
        help="Thickness in meters (0.05 or 0.15) or 'both'.",
    )
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER, help="DG polynomial order.")
    parser.add_argument("--nt", type=int, default=DEFAULT_NT, help="TSI Taylor order.")
    parser.add_argument("--cfl", type=float, default=DEFAULT_CFL, help="CFL number.")
    parser.add_argument(
        "--total-time",
        type=float,
        default=DEFAULT_TOTAL_TIME,
        help="Total simulated time in seconds.",
    )
    parser.add_argument(
        "--n-time-steps",
        type=int,
        default=None,
        help="Run a fixed number of time steps instead of using --total-time.",
    )
    parser.add_argument(
        "--save-step",
        type=int,
        default=0,
        help="Write results_on_the_run.mat every N time steps.",
    )
    parser.add_argument(
        "--save-mesh-step",
        type=int,
        default=0,
        help="Write one .msh field snapshot every N time steps.",
    )
    parser.add_argument(
        "--save-mesh-at-ms",
        type=float,
        default=DEFAULT_SAVE_MESH_AT_MS,
        help="Write an exact .msh field snapshot at this time in milliseconds.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for case outputs.",
    )
    parser.add_argument(
        "--fit-path",
        type=Path,
        default=DEFAULT_FIT,
        help="Vector-fitted ER material file.",
    )
    parser.add_argument(
        "--force-fit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate the ER fit file with Octave.",
    )
    parser.add_argument(
        "--mesh-path",
        type=Path,
        default=None,
        help="Use an existing mesh file instead of generating one from the .geo file.",
    )
    parser.add_argument(
        "--force-mesh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate the mesh even if it already exists.",
    )
    parser.add_argument(
        "--absorbing-layer",
        choices=("pml", "sponge"),
        default=DEFAULT_ABSORBING_LAYER,
        help="Exterior absorbing layer formulation.",
    )
    parser.add_argument(
        "--pml-amp-sigma",
        type=float,
        default=DEFAULT_PML_AMP_SIGMA,
        help="Reference PML damping amplitude.",
    )
    parser.add_argument(
        "--pml-profile",
        choices=("constant", "linear", "quadratic", "cubic", "sine-linear"),
        default=DEFAULT_PML_PROFILE,
        help="PML damping profile.",
    )
    parser.add_argument(
        "--sponge-sigma-max",
        type=float,
        default=DEFAULT_SPONGE_SIGMA_MAX,
        help="Maximum sponge damping coefficient when --absorbing-layer sponge is used.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print time-step progress.",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_CUDA_GRAPH,
        help="Enable CUDA graph replay for time stepping when running on CUDA.",
    )
    parser.add_argument(
        "--cuda-graph-chunk-steps",
        type=int,
        default=DEFAULT_CUDA_GRAPH_CHUNK_STEPS,
        help="Number of time steps captured in one CUDA graph replay.",
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
    parser.add_argument(
        "--use-2d-partitioned-er-rhs",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_2D_PARTITIONED_ER_RHS,
        help="Use compact porous ADE state and one merged simple-RI boundary kernel.",
    )
    parser.add_argument(
        "--er-rhs-backend",
        choices=("auto", "legacy", "gemm"),
        default=DEFAULT_ER_RHS_BACKEND,
        help="ER RHS backend selection for the packed/deep partitioned path.",
    )
    return parser.parse_args()


def resolve_thicknesses(thickness_argument: str):
    if thickness_argument == "both":
        return [0.05, 0.15]
    return [float(thickness_argument)]


def main():
    args = parse_args()
    thicknesses = resolve_thicknesses(args.thickness)
    if args.mesh_path is not None and len(thicknesses) != 1:
        raise ValueError("--mesh-path can only be used with a single --thickness value.")
    for thickness in thicknesses:
        case_info, _ = run_case(
            thickness=thickness,
            output_root=args.output_root,
            Nx=args.order,
            Nt=args.nt,
            cfl=args.cfl,
            total_time=args.total_time,
            n_time_steps=args.n_time_steps,
            save_step=args.save_step,
            save_mesh_step=args.save_mesh_step,
            save_mesh_at_ms=args.save_mesh_at_ms,
            progress=args.progress,
            fit_path=args.fit_path,
            force_fit=args.force_fit,
            mesh_path=args.mesh_path,
            force_mesh=args.force_mesh,
            absorbing_layer=args.absorbing_layer,
            pml_amp_sigma=args.pml_amp_sigma,
            pml_profile=args.pml_profile,
            sponge_sigma_max=args.sponge_sigma_max,
            use_cuda_graph=args.use_cuda_graph,
            cuda_graph_chunk_steps=args.cuda_graph_chunk_steps,
            use_packed_rhs=args.use_2d_packed_rhs,
            use_triton_kernels=args.use_2d_triton_kernels,
            use_triton_deep_rhs=args.use_2d_deep_fused_rhs,
            use_triton_partitioned_er_rhs=args.use_2d_partitioned_er_rhs,
            er_rhs_backend=args.er_rhs_backend,
        )
        print(f"Wrote ER porous absorber outputs to {case_info['output_dir']}")


if __name__ == "__main__":
    main()
