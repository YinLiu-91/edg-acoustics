SetFactory("OpenCASCADE");

// The STEP declares metre units, but the imported OCC bounding box is car-sized
// after applying a 0.001 scale factor.
If (!Exists(scale))
  scale = 0.001;
EndIf

If (!Exists(lc))
  lc = 0.15;
EndIf

Geometry.OCCScaling = scale;
Geometry.OCCImportLabels = 1;

Merge "car_cabin_acoustics_transient_63_cleared.step";

volumes[] = Volume{:};
boundary_surfaces[] = Boundary{ Volume{volumes[]}; };

// Boundary groups recovered from the COMSOL 6.3 .mph file:
//   - named selections come from dmodel.xml SelectionFeature entities;
//   - active boundary conditions come from pressure-acoustics PhysicsFeature nodes;
//   - Door follows the active impedance feature imp2, which includes surfaces
//     298 and 302 in addition to the named "Doors" selection.
windows_surfaces[] = {2, 3, 48, 49, 297, 301, 418, 422, 445, 448, 450, 451};

dashboard_surfaces[] = {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
  21, 22, 23, 24, 25, 26, 27, 38, 39, 40, 41, 42, 43, 50, 51, 52, 53, 65, 66, 67,
  68, 69, 70, 71, 72, 73, 74, 75, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
  91, 94, 95, 96, 97, 98, 99, 100, 102, 103, 104, 106, 107, 110, 111, 112, 113,
  114, 189, 190, 191, 196, 197, 198, 199, 200, 202, 403, 446};

door_impedance_surfaces[] = {108, 109, 175, 176, 298, 302, 321, 322};

leather_seat_surfaces[] = {123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
  134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
  150, 151, 152, 165, 166, 167, 168, 171, 172, 173, 174, 177, 178, 179, 180, 181,
  182, 183, 184, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216,
  217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232,
  233, 234, 235, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
  255, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 272, 273, 274,
  275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290,
  291, 304, 305, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319,
  320, 323, 324, 330, 331, 332, 333, 334, 337, 338, 339, 340, 341, 343, 344, 345,
  346, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365,
  374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390,
  392, 394, 395, 396, 397, 398, 399, 400, 401, 404, 405, 406, 407, 408, 409, 410,
  411, 412, 413, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436,
  437, 438, 439, 440, 441, 442, 443};

carpet_floor_surfaces[] = {7};
roof_trim_surfaces[] = {156, 157, 292, 293, 366, 367};
tweeter_l_source_surfaces[] = {32, 33};
inactive_speaker_hard_surfaces[] = {34, 35, 115, 116, 117, 118, 119, 120, 121, 122};

non_default_surfaces[] = {};
non_default_surfaces[] += windows_surfaces[];
non_default_surfaces[] += dashboard_surfaces[];
non_default_surfaces[] += door_impedance_surfaces[];
non_default_surfaces[] += leather_seat_surfaces[];
non_default_surfaces[] += carpet_floor_surfaces[];
non_default_surfaces[] += roof_trim_surfaces[];
non_default_surfaces[] += tweeter_l_source_surfaces[];
non_default_surfaces[] += inactive_speaker_hard_surfaces[];

default_hard_wall_surfaces[] = boundary_surfaces[];
default_hard_wall_surfaces[] -= non_default_surfaces[];

Physical Surface("DefaultHardWall", 11) = {default_hard_wall_surfaces[]};
Physical Surface("Windows", 12) = {windows_surfaces[]};
Physical Surface("Dashboard", 13) = {dashboard_surfaces[]};
Physical Surface("Doors", 14) = {door_impedance_surfaces[]};
Physical Surface("LeatherSeats", 15) = {leather_seat_surfaces[]};
Physical Surface("CarpetFloor", 16) = {carpet_floor_surfaces[]};
Physical Surface("RoofTrim", 17) = {roof_trim_surfaces[]};
Physical Surface("TweeterLSource", 21) = {tweeter_l_source_surfaces[]};
Physical Surface("InactiveSpeakersHardWall", 22) = {inactive_speaker_hard_surfaces[]};
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
