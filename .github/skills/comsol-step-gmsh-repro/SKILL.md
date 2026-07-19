---
name: comsol-step-gmsh-repro
description: Recover and reproduce COMSOL acoustics cases from provided .mph/.step files through verified boundary semantics, COMSOL or Gmsh mesh export, frequency-dependent material fitting, EDG solver setup, and receiver-result comparison. Use when Codex needs to inspect COMSOL XML/JSON/resources, rebuild named selections or active physics groups, generate or repair .geo/.msh files, export a virtual-geometry mesh through COMSOL, convert admittance/impedance data to EDG RI/RP/CP .mat files, create a case main.py, validate mesh/time-step quality, or document an end-to-end COMSOL-to-EDG reproduction.
---

# Reproduce COMSOL acoustics cases

## Core workflow

1. Inventory the supplied `.mph`, `.step`, tables, existing meshes, scripts, and COMSOL reference outputs. Record the model version and local COMSOL version.
2. Treat `.mph` as a ZIP archive. Recover selections, active physics features, evaluated parameters, interpolation resources, sources, receivers, study time range, and output times before constructing solver inputs.
3. Resolve boundary semantics from active physics features. Let active impedance, normal-velocity, source, or special features override display/helper selections. Never interpret negative COMSOL sentinels as entities.
4. Create an explicit entity-to-physical-label mapping. Require groups to be disjoint, cover every acoustic boundary, and preserve a separate acoustic volume label.
5. Choose the mesh path from evidence:
   - Use STEP/OCC to Gmsh only when COMSOL entity IDs map to imported surfaces and mesh quality gives a practical explicit time step.
   - Use the COMSOL virtual/defeatured mesh export path when STEP topology contains tiny features, entity IDs do not survive import, or the original model already has a validated cleanup mesh.
6. Validate `.msh` topology, physical labels, bbox, cell quality, minimum tetrahedron insphere diameter, estimated EDG time step, and total step count. Unknown or duplicate entity references must fail.
7. Convert COMSOL boundary equations to the quantity expected by EDG. For admittance/impedance tables, fit the reflection coefficient, verify stable poles and passivity, and write the exact `RI/RP/CP` `.mat` schema used by `AbsorbBC`.
8. Build the EDG entrypoint from recovered physics: choose the correct initial condition, boundary-driven source, receiver coordinates, output times, polynomial/time order, CFL, end time, and result metadata.
9. Run structural tests and a short smoke test, then run the full physical case. Compare EDG and COMSOL at identical receiver locations and times; report full-interval and windowed errors.

## Non-negotiable rules

- Do not infer physics from geometry names alone when active COMSOL features are available.
- Do not map STEP/Gmsh surfaces by numeric coincidence without checking import scale, labels, counts, and coverage.
- Do not silently assign unknown NASTRAN or Gmsh entity references to hard wall.
- Do not call a mesh suitable only because `gmsh -check` can read it; include the explicit time-step budget.
- Do not copy COMSOL rational-approximation coefficients into EDG before confirming the represented transfer quantity and sign convention.
- Do not treat a pre-arrival smoke run or an all-zero receiver trace as physical validation.
- Preserve unrelated user files and generated reference artifacts unless the task explicitly authorizes replacing them.

## Reference routing

- Read [mph-physics-recovery.md](references/mph-physics-recovery.md) when extracting selections, physics, parameters, sources, receivers, or resources from `.mph`.
- Read [mesh-export-validation.md](references/mesh-export-validation.md) when choosing STEP/Gmsh versus COMSOL mesh export, assigning physical groups, or diagnosing mesh quality.
- Read [material-solver-validation.md](references/material-solver-validation.md) when fitting materials, generating an EDG entrypoint, or comparing transient results.
- Read [workflow-checklist.md](references/workflow-checklist.md) for every full case reproduction and before claiming completion.

## Reusable templates

Copy and adapt files from `assets/templates/` instead of rewriting common scaffolding:

- `ExportComsolMesh.java.template`: run a selected COMSOL mesh and export NASTRAN with geometry references.
- `recover_mph_case.py.template`: produce a raw, reviewable MPH semantics report.
- `convert_nastran_to_gmsh.py.template`: apply an explicit JSON boundary mapping and write Gmsh 2.2 physical tags.
- `fit_admittance.m.template`: fit normalized reflection data to EDG `RI/RP/CP` coefficients.
- `edg_main.py.template`: configure a zero-initial-state, boundary-driven EDG transient case.

Keep case-specific values in the copied files or case README. Do not store large `.mph`, `.step`, `.nas`, `.msh`, result, or diagnostic image files in the skill.
