---
name: comsol-step-gmsh-repro
description: Recover COMSOL case geometry and boundary semantics from provided .mph and .step files, then generate and validate Gmsh .geo/.msh files for reproducing COMSOL cases. Use when Codex needs to inspect or fix STEP-to-GEO-to-MSH workflows, extract COMSOL named selections/material/source boundary groups from .mph XML/JSON, assign Gmsh Physical Surface labels, validate mesh physical groups/quality, or document how a COMSOL case maps into EDG/acoustics inputs.
---

# COMSOL STEP to Gmsh reproduction

## Core workflow

1. Ground the case from files, not assumptions:
   - Identify `.mph`, `.step`, existing `.geo`, existing `.msh`, docs, and solver entrypoints.
   - Treat `.mph` as a ZIP archive first; do not rely on a local COMSOL version opening it.
   - Check COMSOL creation/last-computation version and local COMSOL version if available.

2. Recover COMSOL boundary semantics before editing `.geo`:
   - Extract `dmodel.xml`, `smodel.json`, and `modelinfo.xml` from `.mph`.
   - Parse `SelectionFeature` nodes for named boundary groups.
   - Parse `PhysicsFeature` nodes for active impedance/source/hard-wall features.
   - Resolve `selection/named` references like `/selection/sel12` to the corresponding selection.
   - Parse `selection/explicit` entities for physics features that do not use named selections.
   - Ignore negative COMSOL sentinels in entity lists, e.g. `-454,-1`.

3. Prefer active physics features over display selections:
   - If a COMSOL display selection conflicts with a physics feature, map the boundary according to the active physics feature.
   - Define default hard wall as imported STEP boundary surfaces minus active impedance/source/special groups.
   - Keep inactive speakers, diagnostic covers, or unused selections separate if they help prevent accidental source/hard-wall mixing.

4. Validate STEP surface tag compatibility before trusting COMSOL entity IDs:
   - Import STEP with Gmsh/OCC using the same scale and import-label options as the target `.geo`.
   - Check surface tag count/range and whether all COMSOL boundary entity IDs exist in Gmsh.
   - If COMSOL IDs do not map to Gmsh surface tags, do not guess; require COMSOL export with named selections or perform manual/geometric classification with explicit validation.

5. Generate `.geo` physical groups only after mapping is validated:
   - Preserve unit/scale logic and document why it is needed.
   - Assign stable physical labels for solver use.
   - Use set-difference for large default hard-wall groups instead of hand-listing hundreds of surfaces.
   - Write `Physical Volume` for the acoustic domain.

6. Validate `.msh` beyond “file exists”:
   - Run `gmsh -check`.
   - Use `meshio` to inspect cells, `gmsh:physical`, `gmsh:geometrical`, bbox, and quality metrics.
   - Verify every boundary triangle has exactly one physical tag and all expected physical labels are non-empty.
   - Treat Gmsh generation warnings about invalid surface elements or ill-shaped tetrahedra as real mesh-quality risks even if `gmsh -check` can read the file.

7. Document the mapping and remaining solver gap:
   - Explain source files, recovered groups, labels, counts, commands, and mesh warnings.
   - Distinguish “physical labels recovered” from “COMSOL boundary equations fully implemented”.
   - For impedance/admittance rational approximations, confirm the target solver expects the same quantity before reusing coefficients.

## Reference

Read [workflow-checklist.md](references/workflow-checklist.md) when executing a full reproduction, writing docs, or debugging a failed COMSOL STEP/Gmsh mesh.
