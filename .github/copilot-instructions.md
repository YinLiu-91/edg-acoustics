# Copilot instructions for edg-acoustics

## Commands

- Install the package: `python3 -m pip install .`
- Install development dependencies: `python3 -m pip install -e ".[dev]"`
- Run the test suite: `pytest`
- Run the scenario 1 regression test: `pytest tests/test_scenario1.py::test_scenario1_simulation`
- Run coverage using the repository coverage config: `coverage run`
- Build the Sphinx documentation: `cd docs && make html`
- Install pinned documentation dependencies: `python3 -m pip install -r docs/requirements.txt`

## Architecture

`edg_acoustics` is a Python room-acoustics simulation package for 3D time-domain nodal discontinuous Galerkin (DG) modeling of the linear acoustic equations. The public API is re-exported from `edg_acoustics/__init__.py`, and examples/tests generally use `import edg_acoustics` followed by top-level classes such as `Mesh`, `AcousticsSimulation`, `AbsorbBC`, `Monopole_IC`, `UpwindFlux`, `TSI_TI`, and `Monopole_postprocessor`.

The simulation pipeline is assembled explicitly by scenario scripts:

1. `Mesh` loads a `.msh` file or generates a `.msh` from `.geo` via the Gmsh Python API, validates boundary labels, reads tetrahedra/boundary triangles with `meshio`, and builds element connectivity.
2. `AcousticsSimulation` builds the DG local system from the mesh: collocation nodes, Vandermonde and derivative matrices, face masks, lift matrix, geometric factors, normals, interior maps, boundary maps, and the time-step scale.
3. Scenario code creates the physics components (`Monopole_IC`, `UpwindFlux`, `AbsorbBC`) and wires them into `AcousticsSimulation` with `init_IC`, `init_Flux`, `init_BC`, and `init_rec`.
4. `TSI_TI` advances pressure and velocity fields by repeatedly calling `AcousticsSimulation.RHS_operator`, including ADE state updates for absorbing boundary conditions.
5. `time_integration()` records pressure at receivers, optionally writes temporary `results_on_the_run.mat`/`.npz` in the current working directory, and `Monopole_postprocessor` resamples/corrects the monopole source spectrum and writes final `.mat`/`.npz` results.

Documentation is built with Sphinx from `docs/`, using AutoAPI over `edg_acoustics` and MyST Markdown support. Example scenarios live under `examples/`; `examples/material_fit` contains MATLAB vector-fitting scripts for producing boundary material coefficients consumed by Python scenarios.

## Repository conventions

- Keep public classes/functions synchronized between module-level `__all__` declarations and `edg_acoustics/__init__.py` when adding API intended for examples or users.
- Use `edg_acoustics.device_ini.device` and `edg_acoustics.device_ini.dtype` for tensors. The code defaults to CUDA when available and `torch.float64`; avoid hard-coding CPU tensors or mismatched float dtypes in simulation paths.
- Preserve the package's array shape conventions: receiver coordinates are `[3, N_rec]`, field variables are generally `[Np, N_tets]`, physical coordinates are `[3, Np, N_tets]`, face quantities are `[4*Nfp, N_tets]`, and boundary maps are flattened node indices.
- Boundary labels are strict. `Mesh` requires `BC_labels` values to match all physical surface labels in the mesh, and `AbsorbBC` expects `BC_para` entries to correspond to `sim.BCnode` labels in order. Build `BC_para` from the ordered `BC_labels` items unless there is a deliberate reason not to.
- Boundary parameter dictionaries use `RI` for frequency-independent reflection, `RP` as a `2 x N` real-pole array, and `CP` as a `4 x N` complex-pole array. `AbsorbBC` validates label coverage, passivity (`|R(omega)| <= 1`), and positive damping terms.
- Gmsh meshes used directly by the package are expected in legacy `.msh` 2.2 format. For `.geo` inputs, `Mesh.create_mesh_from_geo_file()` generates a matching `.msh` and iterates mesh size to target 8-10 points per wavelength.
- Scenario 1 loads material `.mat` files by matching each non-hard-wall `BC_labels` key against the prefix of files in the scenario directory, then maps MATLAB variables `RI`, `AS`/`lambdaS`, and `BS`/`CS`/`alphaS`/`betaS` into `BC_para`.
- Tests are regression-style scenario tests. `tests/test_scenario1.py` constructs a shortened scenario with `tests/golden_files/test_scenario1/scenario1_coarser.msh`, runs the full simulation pipeline, applies postprocessing, and compares arrays to hard-coded golden outputs.
- Follow the existing documentation style for API docs: Google-style docstrings are configured in `pyproject.toml`, and Sphinx Napoleon/AutoAPI generate the reference pages.
