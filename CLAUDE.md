# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install: `python3 -m pip install .`
- Install with dev deps: `python3 -m pip install -e ".[dev]"`
- Run all tests: `pytest`
- Run scenario1 regression test: `pytest tests/test_scenario1.py::test_scenario1_simulation`
- Run golden tests (RHS, time-integration, CUDA Graph): `pytest tests/test_scenario1_golden.py tests/test_scenario1.py::test_scenario1_simulation -q`
- Run CUDA optimization variant tests (requires CUDA): `pytest tests/test_cuda_optimization_variants.py`
- Run coverage: `coverage run`
- Build docs: `cd docs && make html`
- Install docs deps: `python3 -m pip install -r docs/requirements.txt`
- Benchmark (coarse mesh, CPU 8 threads): `python benchmarks/scenario1_benchmark.py --device cpu --cpu-threads 8 --steps 1000`
- Benchmark (coarse mesh, CUDA + Graph): `python benchmarks/scenario1_benchmark.py --device cuda --steps 1000 --cuda-graph`
- Profile: `python benchmarks/scenario1_benchmark.py --device cuda --steps 300 --profile`
- Compile-check all Python: `python -m py_compile edg_acoustics/*.py benchmarks/scenario1_benchmark.py tests/scenario1_utils.py tests/test_scenario1_golden.py`

## Architecture

`edg_acoustics` is a 3D time-domain nodal discontinuous Galerkin (DG) room acoustics solver. It solves the linear acoustic equations on tetrahedral meshes using high-order ADER time integration. The package exposes a public API via `edg_acoustics/__init__.py` with top-level classes: `Mesh`, `AcousticsSimulation`, `AbsorbBC`, `Monopole_IC`, `UpwindFlux`, `TSI_TI`, `Monopole_postprocessor`.

### Simulation pipeline (assembled explicitly by scenario scripts)

1. **`Mesh`** — loads `.msh` (Gmsh legacy 2.2 format) or generates `.msh` from `.geo` via the Gmsh Python API. Validates boundary labels, reads tetrahedra/boundary triangles via `meshio`, builds element connectivity (`EToE`, `EToF`).
2. **`AcousticsSimulation`** — builds the DG local system from the mesh: Fekete collocation nodes, Vandermonde/derivative matrices, face masks (Fmask), lift matrix, geometric factors, normals (n_xyz), interior maps (vmapM/vmapP), boundary maps (BCnode), and the time-step scale (dtscale). Exposes the interior RHS computation (`RHS_operator`/`RHS_operator_packed`) and the time-integration loop (`time_integration`).
3. **Physics components** (`Monopole_IC`, `UpwindFlux`, `AbsorbBC`) are created by scenario code and wired into `AcousticsSimulation` via `init_IC`, `init_Flux`, `init_BC`, and `init_rec`.
4. **`TSI_TI`** (Taylor-series time integration) advances pressure and velocity via `step_dt` (per-field) or `step_dt_packed` (packed layout). Iteratively calls `RHS_operator`/`RHS_operator_packed` with boundary ADE state updates.
5. **`time_integration()`** records receiver pressure, optionally writes intermediate `results_on_the_run.mat`/`.npz`, then `Monopole_postprocessor` resamples/corrects the monopole source spectrum and writes final `.mat`/`.npz` results.

### Core data types and conventions

- **Device selection**: `edg_acoustics.device_ini.device` resolves from `EDG_ACOUSTICS_DEVICE` env var (`auto`/`cpu`/`cuda`). Defaults to CUDA when available. All tensors use `edg_acoustics.device_ini.dtype` = `torch.float64`.
- **Array shapes**:
  - Receiver coordinates: `[3, N_rec]`
  - Field variables (P, Vx, Vy, Vz) per-element: `[Np, N_tets]`
  - Packed state `Q_flat`: `[Np, 4 * N_tets]` with view `Q: [Np, 4, N_tets]`
  - Physical coordinates: `[3, Np, N_tets]`
  - Face quantities: `[4*Nfp, N_tets]` or packed `[4*Nfp, 4, N_tets]`
  - Boundary/face maps: flattened node indices
- **Boundary labels** are strict: `Mesh.BC_labels` values must match all physical surface labels in the mesh, and `AbsorbBC.BCpara` entries must correspond to `sim.BCnode` labels in order.
- **Boundary parameter dictionaries** use `RI` for frequency-independent reflection, `RP` as `2 x N` real-pole array, `CP` as `4 x N` complex-pole array. `AbsorbBC` validates label coverage, passivity (`|R(ω)| ≤ 1`), and positive damping terms (causality/reality).
- **Mesh format**: Gmsh legacy `.msh` 2.2. For `.geo` inputs, `Mesh.create_mesh_from_geo_file()` generates a matching `.msh` and bisection-iterates mesh size to target 8–10 points per wavelength (PPW).

### Material loading convention

Scenario 1 loads material `.mat` files by matching each non-hard-wall `BC_labels` key against the prefix of files in the scenario directory, then maps MATLAB variables `RI`, `AS`/`lambdaS`, and `BS`/`CS`/`alphaS`/`betaS` into `BC_para`.

### GPU optimization architecture

The hot path (`RHS_operator_packed` → `step_dt_packed`) is heavily optimized for CUDA with several layers:

- **Triton kernels** for interior flux (`compact_interior_flux_kernel`), volume RHS (`volume_surface_rhs_kernel`/`volume_rhs_aos_kernel`), and boundary flux (`boundary_ri_flux_kernel`, `boundary_rp_flux_kernel`, `boundary_rp_cp_flux_kernel`) — all enabled by default on CUDA.
- **AoS state layout** (`EDG_ACOUSTICS_AOS_STATE_LAYOUT`) — packs `[p, vx, vy, vz]` contiguously per node-tet pair for better memory coalescing. Auto-enabled for `N_tets >= 10_000` on CUDA.
- **Compact flux coefficients** (`EDG_ACOUSTICS_COMPACT_FLUX_COEFFICIENTS=1`) — derives flux from normals directly instead of loading precomputed coefficient tensors.
- **CUDA Graph** — `time_integration(use_cuda_graph=True)` captures one packed Taylor step into a `torch.cuda.CUDAGraph` and replays at every step, removing CPU kernel-launch overhead.
- **Fused state accumulation** (`EDG_ACOUSTICS_FUSED_STATE_ACCUMULATION`) — fuses RHS write with Taylor accumulation; auto-enabled for `N_tets >= 10_000`.
- **Merged derivatives** (`EDG_ACOUSTICS_MERGED_DERIVATIVES`) — combines Dr/Ds/Dt into one tall cuBLAS DGEMM; auto-enabled for `N_tets >= 10_000`.
- **TileLang lift** (`EDG_ACOUSTICS_TILELANG_LIFT=1`, default on CUDA) — replaces `torch.mm(lift, flux_by_face)` with a TileLang FP64 GEMM kernel (config `bm48_bn64_bk16_s0_t256_fullcol`, ~2.26x speedup on this shape). Defined in `edg_acoustics/tilelang_lift_kernel.py`. Auto-enabled for `N_tets >= 10_000` on CUDA; set to `0` to force fallback to `torch.mm`. Automatically disabled when CUDA Graph is active (non-torch kernels aren't graph-capturable).

Key env var overrides: `EDG_ACOUSTICS_DEVICE`, `EDG_ACOUSTICS_TRITON_VOLUME_RHS`, `EDG_ACOUSTICS_TRITON_INTERIOR_FLUX`, `EDG_ACOUSTICS_TRITON_BOUNDARY_RI`, `EDG_ACOUSTICS_TRITON_BOUNDARY_ADE`, `EDG_ACOUSTICS_COMPACT_FLUX_COEFFICIENTS`, `EDG_ACOUSTICS_FUSED_STATE_ACCUMULATION`, `EDG_ACOUSTICS_MERGED_DERIVATIVES`, `EDG_ACOUSTICS_AOS_STATE_LAYOUT`, `EDG_ACOUSTICS_SCALED_FLUX_KERNELS`, `EDG_ACOUSTICS_PAIRED_INTERIOR_FLUX`, `EDG_ACOUSTICS_TRITON_DERIVATIVE_VOLUME`, `EDG_ACOUSTICS_TRITON_LIFT_SURFACE`, `EDG_ACOUSTICS_TILELANG_LIFT`, `EDG_ACOUSTICS_INTERIOR_FACE_ORDER`.

### Tests

Tests are regression-style scenario tests. `tests/test_scenario1.py` constructs a shortened scenario with a coarser mesh (`tests/golden_files/test_scenario1/scenario1_coarser.msh`), runs the full pipeline, and compares complex transfer function arrays to hard-coded golden outputs. `tests/test_scenario1_golden.py` provides RHS-level and time-integration fp64 golden tests. CUDA variant correctness is verified by `tests/test_cuda_optimization_variants.py` and `tests/test_scenario1_profile_mesh_optimizations.py`, which compare each optimization path against the PyTorch fallback.

### Repository conventions

- Keep public classes/functions synchronized between module-level `__all__` and `edg_acoustics/__init__.py`.
- Use `edg_acoustics.device_ini.device` and `edg_acoustics.device_ini.dtype` for all tensors — never hard-code CPU tensors or float32 in simulation paths.
- Documentation uses Google-style docstrings (configured in `pyproject.toml`); Sphinx Napoleon + AutoAPI generate reference pages.
- Tests use `pytest`; goldens are hard-coded arrays in test files.
- `CHANGELOG.md` uses Keep a Changelog format with Semantic Versioning.
- The `main` branch is the development target; contributions follow the guidelines in `CONTRIBUTING.md`.
