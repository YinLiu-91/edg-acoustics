SetFactory("OpenCASCADE");

// COMSOL exported this STEP in metres, but Gmsh/OpenCASCADE otherwise reads it
// with millimetre-scale coordinates for this file.
Geometry.OCCTargetUnit = "M";

Mesh.MshFileVersion = 2.2;

// Parameters from the COMSOL wave-based room tutorial.
f0 = 700;        // signal centre frequency [Hz]
T0 = 1 / f0;     // signal period [s]
c0 = 343;        // speed of sound [m/s]
lam0 = c0 / f0;  // centre-frequency wavelength [m]

lc = lam0 / 3;
lc_min = 0.04;
eps = 1e-4;

imported_volumes[] = ShapeFromFile("room.step");

// The STEP file contains seven touching volumes. Fragment them first so Gmsh
// creates shared internal faces/nodes instead of duplicate hard boundaries.
all_volumes[] = BooleanFragments{ Volume{ imported_volumes[] }; Delete; }{};
all_points[] = PointsOf{ Volume{ all_volumes[] }; };
MeshSize{ all_points[] } = lc;

// Material groups for EDG acoustics. Surface IDs were generated after the
// BooleanFragments step with Gmsh 4.15.2. Only external surfaces are physical;
// the 19 internal shared surfaces are deliberately omitted.
other_surfaces[] = {
  20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
  122, 123, 124, 125, 126, 127, 154, 155, 156, 157,
  158, 159, 181, 182, 183, 184, 185, 186, 187, 188,
  189, 196, 197, 198, 199, 200, 201, 202, 203, 204,
  206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
  216, 217, 218, 219, 220, 221, 222, 223, 224, 225,
  226, 227, 228, 229, 230, 231, 232, 233, 234, 235,
  236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
  246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
  256, 257, 258, 259, 260, 261, 262, 263, 264, 265,
  266, 267, 268, 269, 270, 271, 272, 273, 274, 275,
  276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
  286, 287, 288, 289, 290, 291, 292, 293, 294, 295,
  296, 297, 298, 299, 300, 301, 302, 303, 304, 305,
  306, 307, 308, 309, 310, 311, 312, 313, 314, 315,
  316, 317, 318, 319, 320, 321, 322, 323, 324, 325,
  326, 327, 328, 329, 330, 331, 332, 333, 334, 335,
  336, 337, 338, 339, 340, 341, 342, 343, 348, 349,
  350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
  360, 361, 362, 363, 364, 365, 366, 367
};
walls[] = {
  138, 139, 140, 142, 143, 144, 160, 161, 162, 163,
  164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
  174, 175, 176, 177, 178, 179, 180, 190, 191, 192,
  193, 194, 195, 345, 346, 347
};
carpet[] = { 19, 146, 148, 150, 152, 368 };
ceiling[] = { 145, 344 };
sofa[] = {
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
  44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
  56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
  68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
  80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
  92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
  103, 104, 105, 106, 107, 108, 109, 110, 111, 128,
  129, 130, 131, 132, 133, 134, 135, 136, 137, 147,
  149, 151, 153
};

// COMSOL applies normal velocity on boundary 222. After fragmentation, the
// corresponding Gmsh surface is 205, identified by the original bbox.
source_comsol_222[] = { 205 };

Physical Surface(11) = { other_surfaces[] };     // default hard/other surfaces
Physical Surface(12) = { walls[] };              // walls
Physical Surface(13) = { carpet[] };             // carpet/floor
Physical Surface(14) = { ceiling[] };            // ceiling
Physical Surface(15) = { sofa[] };               // sofa
Physical Surface(16) = { source_comsol_222[] };  // normal-velocity source

Physical Volume(1) = { all_volumes[] };

Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1;
Mesh.Optimize = 1;
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthMin = lc_min;
Mesh.CharacteristicLengthMax = lc;
