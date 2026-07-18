# COMSOL `.mph + .step` to Gmsh `.geo/.msh` checklist

## MPH inspection

Use `.mph` as a ZIP archive:

```bash
file case.mph
unzip -l case.mph | sed -n '1,80p'
unzip -p case.mph dmodel.xml > /tmp/case_dmodel.xml
unzip -p case.mph smodel.json > /tmp/case_smodel.json
unzip -p case.mph modelinfo.xml > /tmp/case_modelinfo.xml
```

Common useful members:

- `dmodel.xml`: selections, physics features, geometry feature history.
- `smodel.json`: evaluated scalar parameters such as `c0`, `rho0`, impedance values, time ranges.
- `modelinfo.xml`: title, physics interface, COMSOL version, geometry dimension.
- `resources/*`: imported impedance/admittance tables or function data.

## Boundary group extraction

Parse `SelectionFeature`:

- use `tag` as the stable COMSOL reference (`sel2`, `sel12`, `dif1`, ...);
- use `name` as human-facing group name;
- read `outputSelection/explicit[@dim="2"]/@entities` for boundary surface IDs;
- ignore negative sentinel IDs such as `-454,-1`.

Parse `PhysicsFeature`:

- `op="Impedance"`: material/impedance boundary;
- `op="NormalVelocity"`: active velocity source;
- `op="SoundHard"`: hard wall/default wall;
- resolve `selection/named` text like `/selection/sel3`;
- read `selection/explicit/@entities` when a physics feature has its own explicit group;
- collect `param` values such as `ImpedanceModel`, `Zn`, `Y_inf`, `R`, `xi`, `Q`, `zeta`, `ApproximantFunctionReference`, `nvel`.

Important rule: active physics features override display/helper selections. If a helper hard-wall selection includes a surface also selected by an active impedance feature, assign the surface to the active impedance group.

## STEP/Gmsh mapping validation

Import the STEP with the intended scale:

```python
import gmsh
gmsh.initialize([])
gmsh.option.setNumber("Geometry.OCCScaling", 0.001)
gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
gmsh.model.add("check")
gmsh.model.occ.importShapes("case.step")
gmsh.model.occ.synchronize()
surfaces = sorted(tag for dim, tag in gmsh.model.getEntities(2))
volumes = sorted(tag for dim, tag in gmsh.model.getEntities(3))
gmsh.finalize()
```

Check:

- number of volumes is expected;
- bbox has physically plausible size;
- all COMSOL surface IDs referenced by boundary groups are present in Gmsh surfaces;
- extra Gmsh surfaces are deliberately assigned, usually to default hard wall.

If IDs do not align, do not silently map by number. Prefer COMSOL export with named selections, or create a manual geometric classifier and validate with colored previews.

## GEO generation pattern

Use the target repo’s required command prefix if any.

```geo
SetFactory("OpenCASCADE");

If (!Exists(scale))
  scale = 0.001;
EndIf

If (!Exists(lc))
  lc = 0.20;
EndIf

Geometry.OCCScaling = scale;
Geometry.OCCImportLabels = 1;

Merge "case.step";

volumes[] = Volume{:};
boundary_surfaces[] = Boundary{ Volume{volumes[]}; };

impedance_surfaces[] = { ... };
source_surfaces[] = { ... };

non_default_surfaces[] = {};
non_default_surfaces[] += impedance_surfaces[];
non_default_surfaces[] += source_surfaces[];

default_hard_wall_surfaces[] = boundary_surfaces[];
default_hard_wall_surfaces[] -= non_default_surfaces[];

Physical Surface("DefaultHardWall", 11) = {default_hard_wall_surfaces[]};
Physical Surface("MaterialOrSource", 12) = {impedance_surfaces[]};
Physical Volume("AcousticAir", 1) = {volumes[]};

Mesh.MshFileVersion = 2.2;
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1;
Mesh.Optimize = 1;
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
```

## Mesh validation commands

Generate:

```bash
gmsh -3 case.geo -setnumber lc 0.20 -format msh2 -o case_lc0p20.msh
gmsh -check case_lc0p20.msh -
```

Inspect with `meshio`:

```python
import meshio, numpy as np
mesh = meshio.read("case_lc0p20.msh")
print(mesh.points.min(axis=0), mesh.points.max(axis=0))
print({b.type: len(b.data) for b in mesh.cells})
print(mesh.cell_data_dict["gmsh:physical"])
print(mesh.cell_data_dict["gmsh:geometrical"])
```

Minimum acceptance:

- expected physical surface labels exist and are non-empty;
- tetra volume physical tag exists and is correct;
- boundary triangle physical counts sum to total boundary triangles;
- no missing or duplicated assignment when comparing recovered group surfaces to Gmsh surfaces;
- bbox dimensions match the physical problem;
- min area/volume/edge-length are plausible for the intended timestep/order.

Warnings to document explicitly:

- invalid surface elements;
- ill-shaped tetrahedra after optimization;
- extremely small triangle areas/tet volumes;
- unexpected extra surfaces after STEP import;
- mismatch between STEP unit declaration and required OCC scaling.
