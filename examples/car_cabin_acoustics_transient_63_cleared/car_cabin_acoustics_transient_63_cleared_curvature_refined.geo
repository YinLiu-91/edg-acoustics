// Curvature-refined mesh entry for the COMSOL car cabin STEP.
//
// This file reuses the recovered physical boundary groups from the base .geo,
// then enables Gmsh curvature-based sizing so regions with stronger geometric
// variation get smaller elements than the smooth large surfaces.
//
// Recommended command:
//   gmsh -3 car_cabin_acoustics_transient_63_cleared_curvature_refined.geo \
//     -setnumber algo3d 10 \
//     -setnumber lc_min 0.06 -setnumber lc_max 0.12 -setnumber curvature_refine 32 \
//     -format msh2 -o car_cabin_acoustics_transient_63_cleared_curv_hxt_lc0p12_min0p06.msh
//
// The HXT 3D mesher (algo3d=10) is used here because the Delaunay path
// (algo3d=1, lc_min=0.04, lc_max=0.12) still reports ill-shaped tetrahedra on
// tiny STEP features.  This file intentionally keeps the imported CAD topology
// and recovered Physical Surface groups unchanged.

If (!Exists(lc_max))
  lc_max = 0.12;
EndIf

If (!Exists(lc_min))
  lc_min = 0.06;
EndIf

If (!Exists(curvature_refine))
  curvature_refine = 32;
EndIf

If (!Exists(algo3d))
  algo3d = 10;
EndIf

If (!Exists(optimize_netgen))
  optimize_netgen = 0;
EndIf

If (!Exists(optimize_threshold))
  optimize_threshold = 0.3;
EndIf

// Set lc before including the base .geo.  The base file uses lc as the default
// global mesh size; the final options below then replace the uniform min/max
// sizing with a curvature-sensitive range.
lc = lc_max;

Include "car_cabin_acoustics_transient_63_cleared.geo";

Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = curvature_refine;
Mesh.CharacteristicLengthExtendFromBoundary = 1;
Mesh.CharacteristicLengthMin = lc_min;
Mesh.CharacteristicLengthMax = lc_max;

// HXT avoids the ill-shaped tetrahedra warning observed with the default
// Delaunay path while preserving the imported surfaces and physical labels.
Mesh.Algorithm = 6;
Mesh.Algorithm3D = algo3d;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = optimize_netgen;
Mesh.OptimizeThreshold = optimize_threshold;
