# Office Space Admittance Carpet COMSOL Reproduction

This directory reproduces `office_space_acoustics_64_cleared.mph` through the COMSOL mesh-export path. The model has virtual geometry and a COMSOL Absorbing Layer, so the supported baseline is:

- export COMSOL `mesh1` to NASTRAN with geometric refs;
- convert NASTRAN refs to Gmsh 2.2 physical labels, scaling COMSOL cm coordinates by `0.01` to meters;
- evaluate the active COMSOL `pff1/pff2/pff3` admittances and fit their reflection coefficients for EDG;
- run EDG with the COMSOL initial pressure pulse and compare against `pg5` point responses.

## Recovered COMSOL Setup

Main scalar parameters:

| name | value |
|---|---:|
| `fc` | `500 Hz` |
| `f0` | `750 Hz` |
| `c0` | `343 m/s` |
| `rho0` | `1.2 kg/m^3` |
| `xs,ys,zs` | `4.0, 7.0, 1.5 m` |
| `B` | `0.15440424822094373 m` |
| `S0` | `1 Pa` |
| `T0` | `0.0013333333333333333 s` |
| `T_ir` | `0.4 s` |

Initial condition:

```text
p0(x,y,z) = S0*exp(-log(2)*((x-xs)^2+(y-ys)^2+(z-zs)^2)/B^2)
u0 = 0
```

COMSOL `std2` stores `range(0,T0/30,T_ir)`, giving 9001 samples at `4.4444444444444447e-5 s`.

## Boundary Labels

| EDG label | name | COMSOL feature | entities | EDG treatment |
|---:|---|---|---|---|
| 11 | `DefaultHardWall` | `shb1` default | exported boundary refs minus active groups | `RI=1` |
| 12 | `ClosedWindows` | `imp1` / `sel3` | `13,18,42,46,54,56,58,62,70,72,74,78,86` | `sqrt(1-0.005)` |
| 13 | `Doors` | `imp2` / `sel4` | `381,389` | `sqrt(1-0.04)` |
| 14 | `BrickWall` | `imp4` / `sel5` | `383` | `sqrt(1-0.01)` |
| 15 | `Carpet` | `imp3` / `sel6` | `35` | fitted `carpet.mat` |
| 16 | `Ceiling` | `imp5` / `sel7` | `41` | fitted `ceiling.mat` |
| 17 | `Gypsum` | `imp6` / `sel8` | `33,34,88,374,375,382` | fitted `gypsum.mat` |
| 18 | `OpenWindowAbsorbingLayer` | `imp7` explicit | `5,6,7,8` | matched `RI=0` baseline |

Known gap: COMSOL applies a spherical Absorbing Layer coordinate transform to domains `48,1,2,3,4`. Exported `mesh1` contains tetra references `1..9,46,47`; virtual-geometry domain `48` has no `CTETRA` and is recorded as the sole reviewed missing volume reference. Current 3D EDG has no equivalent volume coordinate transform, so the exported layer tetrahedra are ordinary air and only the outer boundaries use `RI=0`. Late-tail agreement must be interpreted with that limitation.

## Commands

Generate the raw recovered model report:

```bash
rtk python examples/office_space_admittance_carpet/office_space_boundary_groups.py \
  --json-out examples/office_space_admittance_carpet/office_space_recovery_report.json
```

Export COMSOL `mesh1`:

```bash
rtk /usr/local/comsol64/multiphysics/bin/comsol compile \
  examples/office_space_admittance_carpet/ExportOfficeSpaceMesh.java

rtk /usr/local/comsol64/multiphysics/bin/comsol batch \
  -inputfile /media/liu/research/linux/edg-muxi/edg-acoustics/examples/office_space_admittance_carpet/ExportOfficeSpaceMesh.class \
  /media/liu/research/linux/edg-muxi/edg-acoustics/examples/office_space_admittance_carpet/office_space_acoustics_64_cleared.mph \
  /media/liu/research/linux/edg-muxi/edg-acoustics/examples/office_space_admittance_carpet/office_space_comsol_mesh1.nas \
  -batchlog /media/liu/research/linux/edg-muxi/edg-acoustics/examples/office_space_admittance_carpet/office_space_mesh_export.log \
  -batchlogout -nosave
```

Convert to Gmsh 2.2 and write diagnostics:

```bash
rtk python examples/office_space_admittance_carpet/convert_comsol_nastran_to_gmsh.py \
  --nastran examples/office_space_admittance_carpet/office_space_comsol_mesh1.nas \
  --output examples/office_space_admittance_carpet/office_space_comsol_mesh1.msh \
  --json-out examples/office_space_admittance_carpet/office_space_mesh_conversion_report.json

rtk gmsh -check examples/office_space_admittance_carpet/office_space_comsol_mesh1.msh
```

Fit admittance tables:

```bash
rtk octave -qf examples/office_space_admittance_carpet/fit_office_space_admittance.m
```

Actual mesh diagnostics from the generated mesh:

| item | value |
|---|---:|
| points | `59684` |
| tetrahedra | `300750` |
| topological boundary triangles | `37232` |
| discarded internal shell triangles | `2824` |
| bbox | `8.6 x 9.2 x 3.0 m` |
| min edge length | `0.014965965555218976 m` |
| min insphere diameter | `0.008938802997256802 m` |
| exterior boundary refs covered | `343 / 343` |

EDG Nx=4/Nt=4/CFL=0.1 budget:

| item | value |
|---|---:|
| `dt` | `2.895627793085788e-7 s` |
| steps to `20*T0` | `92093` |
| steps to `0.4 s` | `1381394` |

`Nx=4` is the COMSOL-order reproduction target. `Nx=2` is provided for low-memory initialization and smoke validation; it is not the comparison discretization. Longer tests showed that `CFL=0.5` diverges for both `Nt=3` and `Nt=4` on this mesh. The default and validated upper limit are therefore `CFL=0.1`. The driver checks the full field every 1000 steps and aborts on NaN, Inf, or `max_abs > 1e6`.

Material fit results:

| material | poles | RMS | max `|R|` |
|---|---:|---:|---:|
| carpet | 8 | `3.0505880776567473e-16` | `0.9978943360315591` |
| ceiling | 16 | `7.081680338605388e-15` | `0.9921410369207863` |
| gypsum | 16 | `1.000698735081774e-15` | `0.999199928820656` |

The saved `.mat` files contain both the raw table admittance and the COMSOL PFF admittance. Their reflection RMS differences are `0.00366996`, `0.00117583`, and `0.00287333`, respectively; therefore fitting the raw tables is not equivalent to reproducing the active COMSOL boundary features.

Export the receiver coordinates from COMSOL geometry without solving `std2`:

```bash
rtk /usr/local/comsol64/multiphysics/bin/comsol compile \
  examples/office_space_admittance_carpet/ExportOfficeSpaceReceiverPoints.java

rtk /usr/local/comsol64/multiphysics/bin/comsol batch \
  -inputfile /media/liu/research/linux/edg-muxi/edg-acoustics/examples/office_space_admittance_carpet/ExportOfficeSpaceReceiverPoints.class \
  /media/liu/research/linux/edg-muxi/edg-acoustics/examples/office_space_admittance_carpet/office_space_acoustics_64_cleared.mph \
  -batchlog /tmp/office_space_receiver_progress.log \
  -batchlogout -nosave \
  > examples/office_space_admittance_carpet/office_space_receiver_export.log 2>&1

rtk python examples/office_space_admittance_carpet/extract_receiver_points.py \
  --log examples/office_space_admittance_carpet/office_space_receiver_export.log \
  --output examples/office_space_admittance_carpet/office_space_receiver_points.json
```

The recovered point entities are:

| point entity | coordinate (m) |
|---:|---|
| `230` | `(1.5, 1.7, 1.0)` |
| `233` | `(1.5, 7.3, 1.0)` |
| `467` | `(4.0, 6.0, 1.0)` |

Export COMSOL golden pressure traces. The supplied `cleared` MPH currently has empty `sol2`; the command without `runstd2` is a guard and will fail fast instead of silently exporting stale data. Add `runstd2` only when you are ready to spend the full COMSOL solve time. Samples are written to the batch log so the procedure works with COMSOL's default restricted Java file access.

```bash
rtk /usr/local/comsol64/multiphysics/bin/comsol compile \
  examples/office_space_admittance_carpet/ExportOfficeSpaceGolden.java

rtk /usr/local/comsol64/multiphysics/bin/comsol batch \
  -inputfile /media/liu/research/linux/edg-muxi/edg-acoustics/examples/office_space_admittance_carpet/ExportOfficeSpaceGolden.class \
  /media/liu/research/linux/edg-muxi/edg-acoustics/examples/office_space_admittance_carpet/office_space_acoustics_64_cleared.mph \
  runstd2 \
  -batchlog /tmp/office_space_golden_progress.log \
  -batchlogout -nosave \
  > examples/office_space_admittance_carpet/office_space_golden_export.log 2>&1

rtk python examples/office_space_admittance_carpet/extract_comsol_golden.py \
  --log examples/office_space_admittance_carpet/office_space_golden_export.log \
  --receiver-json examples/office_space_admittance_carpet/office_space_receiver_points.json \
  --output examples/office_space_admittance_carpet/office_space_receiver_golden.csv
```

Run the low-memory initialization and two-step smoke validation used for this reproduction:

```bash
rtk python examples/office_space_admittance_carpet/main.py \
  --nx 2 --nt 4 --cfl 0.1 --n-time-steps 2 \
  --no-progress --no-use-cuda-graph \
  --output examples/office_space_admittance_carpet/result_nx2_smoke.mat
```

This run must report a finite final field. Short runs only validate setup and execution; use at least several thousand steps when assessing time-integration stability.

Run the full case:

```bash
rtk python examples/office_space_admittance_carpet/main.py \
  --nx 4 --nt 4 --cfl 0.1 --total-time 0.4 \
  --output examples/office_space_admittance_carpet/result.mat
```

The driver uses `ceil(T/dt)` integration steps so the numerical trajectory covers `0.4 s`, then interpolates receivers only inside the simulated interval onto the exact 9001-point COMSOL output grid. `--n-time-steps` stores raw EDG step samples and is intended for smoke runs.

Use `--save-step 10000 --save-mesh-step 10000` to retain checkpoints every 10000 steps. MAT checkpoints are named `results_step010000_t*.mat`, `results_step020000_t*.mat`, and so on; `results_on_the_run.mat` is also updated as an alias for the latest checkpoint. Mesh snapshots are written under `results_on_the_run_msh/` with the step and physical time in each filename.

Use `--progress-step 20` to print receiver values and progress after every 20 completed steps. Very small progress intervals add terminal I/O overhead during a full run.

Compare against COMSOL:

```bash
rtk python examples/office_space_admittance_carpet/compare_receiver_response.py \
  --comsol examples/office_space_admittance_carpet/office_space_receiver_golden.csv \
  --edg examples/office_space_admittance_carpet/result.mat \
  --receiver-json examples/office_space_admittance_carpet/office_space_receiver_points.json \
  --metrics-out examples/office_space_admittance_carpet/office_space_comparison_metrics.json \
  --plot examples/office_space_admittance_carpet/office_space_comparison.png
```

## Completion Criteria

The reproduction should not be considered complete until:

- NASTRAN conversion reports only topological exterior shells, no duplicate boundary refs, and only the reviewed missing virtual domain `48`;
- `gmsh -check` passes and all `37232` triangles have tetra-face multiplicity one;
- all three material fits are stable and passive over `0..2000 Hz`;
- COMSOL golden, receiver JSON, and EDG result contain identical point IDs and coordinates;
- EDG short and full runs produce finite, nonzero receiver traces after wave arrival;
- comparison reports full-interval plus windowed errors;
- any discrepancy from the missing 3D COMSOL Absorbing Layer is documented.

Run the office-specific regression tests with:

```bash
rtk pytest -q tests/test_office_space_repro.py
```

## Current Verification Status

Completed in this workspace:

- `gmsh -check` passed for `59684` nodes and `337982` total elements;
- conversion retained `37232` topological exterior triangles and discarded `2824` internal shells;
- all eight boundary labels passed EDG physical-admissibility checks;
- COMSOL geometry export and log extraction reproduced point entities `230,233,467` in meters;
- all three COMSOL-PFF material fits passed the `0..2000 Hz` passivity check;
- `11` office-specific tests and `12` related car-cabin solver tests passed;
- the `Nx=2`, `Nt=4`, two-step EDG smoke run completed with a finite full field.

Pending on the full-run machine:

- run COMSOL `std2` because the supplied cleared MPH has an empty `sol2`;
- extract `office_space_receiver_golden.csv` from the COMSOL batch stdout;
- run the comparison target at `Nx=4`, `Nt=4` through `0.4 s`;
- generate the comparison metrics and plot. These results remain subject to the documented missing 3D absorbing-layer volume transform.
