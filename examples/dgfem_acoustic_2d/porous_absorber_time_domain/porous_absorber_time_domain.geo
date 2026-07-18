SetFactory("OpenCASCADE");

If (!Exists(w0))
  w0 = 0.05;
EndIf

L0 = 1.5;
layer = L0 / 5.0;
width = 3.0 * L0;
xmin = -1.5 * L0;
xmax = xmin + width;
ymin = -w0;
ymax = L0;
outer_xmin = xmin - layer;
outer_xmax = xmax + layer;
outer_ymax = ymax + layer;
lc = 343.0 / 4000.0 / 1.5;
eps = 1e-6;

Rectangle(1) = {outer_xmin, ymin, 0, outer_xmax - outer_xmin, outer_ymax - ymin, 0};
Rectangle(2) = {xmin, 0, 0, width, ymax, 0};
Rectangle(3) = {xmin, ymin, 0, width, w0, 0};

BooleanFragments{ Surface{1}; Delete; }{ Surface{2, 3}; Delete; }

c_left[] = Curve In BoundingBox {outer_xmin - eps, ymin - eps, -eps, outer_xmin + eps, outer_ymax + eps, eps};
c_right[] = Curve In BoundingBox {outer_xmax - eps, ymin - eps, -eps, outer_xmax + eps, outer_ymax + eps, eps};
c_top[] = Curve In BoundingBox {outer_xmin - eps, outer_ymax - eps, -eps, outer_xmax + eps, outer_ymax + eps, eps};
c_bottom_left[] = Curve In BoundingBox {outer_xmin - eps, ymin - eps, -eps, xmin + eps, ymin + eps, eps};
c_bottom_right[] = Curve In BoundingBox {xmax - eps, ymin - eps, -eps, outer_xmax + eps, ymin + eps, eps};
Physical Surface("Air", 1) = {2};
Physical Surface("Porous", 2) = {3};
Physical Surface("PML", 3) = {4};

Physical Curve("Outer", 11) = {c_left[], c_right[], c_top[], c_bottom_left[], c_bottom_right[]};
Physical Curve("Rigid", 12) = {12};

Mesh.MeshSizeMin = lc;
Mesh.MeshSizeMax = lc;
