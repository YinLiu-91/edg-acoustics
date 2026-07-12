"""2D triangular mesh support."""

from __future__ import annotations

import meshio
import torch

from edg_acoustics.mesh_base import SimplexMesh


class Mesh2D(SimplexMesh):
    """Mesh data structure for 2D triangular DG simulations."""

    dim = 2
    element_cell_type = "triangle"
    boundary_cell_type = "line"
    faces_per_element = 3
    face_vertex_ids = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.long)

    def __init__(self, filename: str, BC_labels: dict[str, int]):
        self._init_common(filename, BC_labels)
        if not filename.endswith(".msh"):
            raise ValueError("Mesh2D currently supports Gmsh .msh files.")
        self.init_from_mesh_file(filename, BC_labels)

    def init_from_mesh_file(self, filename: str, BC_labels: dict[str, int]):
        mesh_data = meshio.read(filename)
        if self.element_cell_type not in mesh_data.cells_dict:
            raise ValueError("Mesh2D requires triangle elements.")
        if self.boundary_cell_type not in mesh_data.cells_dict:
            raise ValueError("Mesh2D requires line boundary elements.")
        self._load_vertices(mesh_data)
        self._load_boundary_faces(mesh_data, BC_labels)
        self._load_elements(mesh_data)
        self.N_triangles = self.N_elements
        self.BC_lines = self.BC_faces
        self.N_BC_lines = self.N_BC_faces
