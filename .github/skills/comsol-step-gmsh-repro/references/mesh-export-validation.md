# Generate and validate a solver mesh

## Contents

1. Choose the mesh path
2. STEP/OCC to Gmsh
3. COMSOL mesh export
4. NASTRAN to Gmsh conversion
5. Quality and time-step acceptance

## 1. Choose the mesh path

Prefer STEP/OCC/Gmsh when surface identity survives import, the CAD does not contain harmful slivers, and the resulting explicit time step is practical. Prefer COMSOL mesh export when the `.mph` already contains defeaturing or virtual operations, STEP import changes entity identity, or tiny STEP features dominate the time step.

Do not remove or merge STEP faces solely to eliminate warnings unless the boundary mapping is rebuilt and visually/topologically revalidated. Geometry cleanup can change physical semantics even when the exterior shape looks unchanged.

## 2. STEP/OCC to Gmsh

Import using exactly the scale and label settings intended for the `.geo`:

```python
import gmsh

gmsh.initialize([])
try:
    gmsh.option.setNumber("Geometry.OCCScaling", 0.001)
    gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
    gmsh.model.add("check")
    gmsh.model.occ.importShapes("case.step")
    gmsh.model.occ.synchronize()
    surfaces = sorted(tag for dim, tag in gmsh.model.getEntities(2))
    volumes = sorted(tag for dim, tag in gmsh.model.getEntities(3))
finally:
    gmsh.finalize()
```

Require plausible bbox/volume count and coverage of every COMSOL entity used by an active boundary. Numeric overlap alone is insufficient when counts, units, or topology differ.

Generate default hard wall by set difference:

```geo
boundary_surfaces[] = Boundary{ Volume{volumes[]}; };
non_default_surfaces[] = {};
non_default_surfaces[] += impedance_surfaces[];
non_default_surfaces[] += source_surfaces[];
default_hard_wall_surfaces[] = boundary_surfaces[];
default_hard_wall_surfaces[] -= non_default_surfaces[];

Physical Surface("DefaultHardWall", 11) = {default_hard_wall_surfaces[]};
Physical Surface("Source", 21) = {source_surfaces[]};
Physical Volume("AcousticAir", 1) = {volumes[]};
```

## 3. COMSOL mesh export

Use `assets/templates/ExportComsolMesh.java.template`. Supply the actual component and mesh tags rather than assuming `comp1/mesh1`. The selected mesh should include the intended virtual geometry/defeaturing operations.

```bash
/path/to/comsol compile ExportCaseMesh.java
/path/to/comsol batch \
  -inputfile ExportCaseMesh.class \
  case.mph case.nas comp1 mesh2 \
  -batchlog case_mesh_export.log \
  -batchlogout -nosave
```

Export linear NASTRAN solid and shell elements with geometry information. The shell elements carry boundary entity/property references; the solid elements form the acoustic domain.

Check the batch log for a successful mesh build and export. Do not reuse a stale `.nas` after changing selections, virtual operations, or mesh settings.

## 4. NASTRAN to Gmsh conversion

Create a reviewed mapping JSON for `assets/templates/convert_nastran_to_gmsh.py.template`:

```json
{
  "volume_label": 1,
  "physical_groups": [
    {"name": "DefaultHardWall", "label": 11, "entities": [1, 2, 3]},
    {"name": "Source", "label": 21, "entities": [10, 11]}
  ]
}
```

Require:

- each entity appears in exactly one physical surface group;
- every exported boundary reference is present in the mapping;
- tetrahedra receive the acoustic volume label;
- `gmsh:geometrical` preserves the COMSOL entity reference;
- `gmsh:physical` contains the stable EDG label.

Write Gmsh 2.2 when the target EDG mesh reader requires it. Never replace unknown boundary references with the default wall label.

## 5. Quality and time-step acceptance

Run:

```bash
gmsh -check case.msh -
python mesh_diagnostics.py case.msh
```

Record:

- point, triangle, and tetrahedron counts;
- bbox and physical-group counts;
- minimum and median triangle area;
- minimum and median tetra volume and edge length;
- tetrahedron insphere diameter `d_in = 6 V / S`, where `S` is total face area;
- invalid/ill-shaped element warnings;
- actual EDG `dt` for the intended `Nx`, time order, CFL, and sound speed;
- `floor(T_end/dt)` or the solver's exact step count.

`gmsh -check` verifies readability and topology; it does not prove that the mesh is efficient or stable for an explicit high-order solver. Reject a mesh with non-positive cells, missing labels, unknown references, implausible scale, or a time-step count outside the case budget. Document marginal quality instead of hiding it.
