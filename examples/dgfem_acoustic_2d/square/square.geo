SetFactory("OpenCASCADE");

lc = 0.1;
Rectangle(1) = {-5, -5, 0, 10, 10, 0};
MeshSize{ PointsOf{ Surface{1}; } } = lc;

Physical Surface("Domain", 1) = {1};
Physical Curve("Absorbing", 11) = {1, 2, 3, 4};
