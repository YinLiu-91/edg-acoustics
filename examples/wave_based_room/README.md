# Wave-Based Room COMSOL Reproduction

This case reproduces `wave_based_room.mph` with zero initial state and the
active COMSOL normal-velocity boundary source `nvel1`.

## Recovered COMSOL Semantics

- COMSOL model: 6.4.0.250, local export tested with COMSOL 6.4.
- Physics: Pressure Acoustics, Time Explicit.
- Air: `rho0=1.2 kg/m^3`, `c0=343 m/s`.
- Source: boundary entity `222`, `vn(t)`, Gaussian-modulated sine,
  `f0=700 Hz`, `T0=1/f0`, `delay=2*T0`, `sigma=T0/2`, amplitude `1 m/s`.
- Study output range: `range(0,T0,30*T0)`, `Tend=0.04285714285714286 s`.
- Listening points, in probe order:
  `122=(1.2,1.3125,1)`, `121=(0.2,0.875,1)`,
  `53=(-0.8,0.4375,1)`, `35=(-1.8,0,1)`.

Physical labels:

| label | name | COMSOL feature | entities | EDG treatment |
|---:|---|---|---|---|
| 11 | DefaultHardWall | `shb1` default | exported exterior minus active groups | `RI=1` |
| 12 | Carpet | `imp1/sel9` | `3,75` | fitted reflection |
| 13 | Ceiling | `imp2/sel8` | `7,77` | fitted reflection |
| 14 | Sofa | `imp3/sel1` | `10-20,25-31,51,52,61,68-71` | fitted reflection |
| 15 | Wall | `imp4/sel10` | `1,2,4,5,8,9,74,78,262` | fitted reflection |
| 21 | NormalVelocitySource | `nvel1` | `222` | prescribed normal velocity |

## Build Inputs

Generate EDG material files:

```bash
PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
python examples/wave_based_room/fit_wave_based_room_admittance.py
```

Export a pure-tet COMSOL mesh and convert it:

```bash
cd /media/liu/research/linux/edg-muxi/edg-acoustics/examples/wave_based_room
comsol compile ExportWaveBasedRoomMesh.java
comsol batch -inputfile ExportWaveBasedRoomMesh.class \
  wave_based_room.mph wave_based_room_comsol_tet_hmax0p163_hmin0p04.nas \
  -nosave > wave_based_room_mesh_export_stdout.log 2>&1

PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
python convert_comsol_nastran_to_gmsh.py \
  --nastran wave_based_room_comsol_tet_hmax0p163_hmin0p04.nas \
  --output wave_based_room_comsol_tet_hmax0p163_hmin0p04.msh \
  --json-out wave_based_room_mesh_conversion_report.json
```

Export COMSOL listening-point golden:

```bash
cd /media/liu/research/linux/edg-muxi/edg-acoustics/examples/wave_based_room
comsol compile ExportWaveBasedRoomReceiverPoints.java
comsol compile ExportWaveBasedRoomGolden.java
comsol batch -inputfile ExportWaveBasedRoomGolden.class \
  wave_based_room.mph \
  -nosave > wave_based_room_golden_export_stdout.log 2>&1

PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
python extract_comsol_golden.py \
  --log wave_based_room_golden_export_stdout.log \
  --receiver-json wave_based_room_receiver_points.json \
  --csv wave_based_room_comsol_golden.csv \
  --mat wave_based_room_comsol_golden.mat
```

The CSV/MAT golden contains both `pressure_pa` and `pressure_normalized`.
The normalized COMSOL probe expression is
`pate.p_t/(1[m/s]*pate.Z)`, converted to Pa with `rho0*c0`.

## Run EDG

Short local smoke run:

```bash
cd /media/liu/research/linux/edg-muxi/edg-acoustics/examples/wave_based_room
PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
python main.py --nx 3 --nt 3 --cfl 0.5 --n-time-steps 3 \
  --no-use-cuda-graph --output nx3nt3_smoke.mat
```

Full run with progress every 20 steps and every 10000-step MAT/MSH checkpoints:

```bash
cd /media/liu/research/linux/edg-muxi/edg-acoustics/examples/wave_based_room
PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
python main.py --nx 3 --nt 3 --cfl 0.5 \
  --total-time 0.04285714285714286 \
  --progress-step 20 \
  --save-step 10000 \
  --save-mesh-step 10000 \
  --no-use-cuda-graph \
  --output nx3nt3.mat
```

Compare against COMSOL:

```bash
PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
python compare_receiver_response.py \
  --comsol wave_based_room_comsol_golden.csv \
  --edg nx3nt3.mat \
  --receiver-json wave_based_room_receiver_points.json \
  --plot wave_based_room_comparison.png \
  --metrics-out wave_based_room_comparison_metrics.json
```

## Known Gaps

The mesh is regenerated as pure tetrahedra because EDG 3D currently consumes
tetrahedral meshes. This intentionally does not reuse COMSOL's original mixed
hex/prism/pyramid/tet mesh, so final discrepancies include mesh-dispersion and
mesh-topology differences in addition to material/source differences.

On COMSOL 6.4.0.293 the Java size subproperties used in older examples are
not accepted for this temporary mesh sequence, so the generated mesh uses
`autoMeshSize(5)`. The conversion report records the actual topology and
quality: `52900` tetrahedra, `12238` exterior triangles, bbox
`[-2.5,-2,0]` to `[2.5,2,2.6]`, and min insphere diameter about
`4.93e-3 m`.
