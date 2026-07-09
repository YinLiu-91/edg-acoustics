SetFactory("OpenCASCADE");
Mesh.MshFileVersion = 2.2;

DefineConstant[
  lc = {0.0775, Name "Parameters/lc"}
];

Box(1) = {0.0, 0.0, 0.0, 5.0, 5.0, 5.0};

surfaces[] = Boundary{ Volume{1}; };
Physical Surface(11) = {surfaces[]}; // hard wall on all six faces
Physical Volume(1) = {1};

Mesh.Algorithm = 1;
Mesh.Algorithm3D = 1;
Mesh.Optimize = 1;
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
