"""Shared simplex mesh helpers for dimension-specific DG meshes."""

from __future__ import annotations

import abc

import numpy
import torch

import edg_acoustics.device_ini as device_ini


class SimplexMesh(abc.ABC):
    """Base container for simplex meshes used by DG solvers."""

    dim: int
    element_cell_type: str
    boundary_cell_type: str
    faces_per_element: int
    face_vertex_ids: torch.Tensor

    def _init_common(
        self,
        filename: str,
        BC_labels: dict[str, int],
        domain_labels: dict[str, int] | None = None,
    ):
        self.filename = filename
        self.BC_labels = BC_labels
        self.domain_labels = {} if domain_labels is None else dict(domain_labels)

    def _load_vertices(self, mesh_data):
        self.N_vertices = mesh_data.points.shape[0]
        self.vertices = mesh_data.points.transpose()

    def _load_boundary_faces(self, mesh_data, BC_labels: dict[str, int]):
        physical = mesh_data.cell_data_dict["gmsh:physical"][self.boundary_cell_type]
        labels_in_mesh = sorted(numpy.unique(physical))
        labels_in_input = sorted(BC_labels.values())
        if labels_in_input != labels_in_mesh:
            raise ValueError(
                "[edg_acoustics.SimplexMesh] All BC labels must be present in the "
                "mesh and all mesh labels must be present in BC_labels."
            )

        boundary_cells = mesh_data.cells_dict[self.boundary_cell_type]
        self.N_BC_faces = {}
        self.BC_faces = {}
        for name, label in BC_labels.items():
            has_label = physical == label
            self.N_BC_faces[name] = int(has_label.sum())
            self.BC_faces[name] = torch.from_numpy(boundary_cells[has_label]).to(
                device_ini.device
            )

    def _load_elements(self, mesh_data):
        elements = mesh_data.cells_dict[self.element_cell_type]
        self.N_elements = elements.shape[0]
        self.EToV = torch.from_numpy(elements.transpose()).to(device_ini.device)
        self.EToE, self.EToF = self.compute_element_connectivity(self.EToV)
        physical = mesh_data.cell_data_dict["gmsh:physical"][self.element_cell_type]
        self.element_physical_labels = torch.from_numpy(physical).to(device_ini.device)
        self.N_element_labels = {
            int(label): int((physical == label).sum()) for label in numpy.unique(physical)
        }
        self.domain_elements = {}
        self.N_domain_elements = {}
        if self.domain_labels:
            labels_in_mesh = sorted(numpy.unique(physical))
            labels_in_input = sorted(self.domain_labels.values())
            missing = sorted(set(labels_in_input).difference(labels_in_mesh))
            if missing:
                raise ValueError(
                    "[edg_acoustics.SimplexMesh] Domain labels missing from the mesh: "
                    + ", ".join(str(label) for label in missing)
                )
            for name, label in self.domain_labels.items():
                selector = physical == label
                self.domain_elements[name] = torch.nonzero(
                    torch.from_numpy(selector), as_tuple=False
                ).reshape(-1).to(device_ini.device)
                self.N_domain_elements[name] = int(selector.sum())

    def compute_element_connectivity(self, EToV: torch.Tensor):
        """Build element-to-element and element-to-face adjacency."""

        elements_t = EToV.T
        n_elements = EToV.shape[1]
        n_face_vertices = self.face_vertex_ids.shape[1]
        face_vertices = torch.vstack(
            [elements_t[:, face.tolist()] for face in self.face_vertex_ids]
        ).to(device_ini.device)
        face_vertices, _ = torch.sort(face_vertices, dim=-1)
        face_indices = torch.arange(
            0, n_elements * self.faces_per_element, device=device_ini.device
        )
        vertex_base = int(EToV.max().item()) + 1
        multipliers = (
            torch.as_tensor(
                [vertex_base**power for power in range(n_face_vertices)],
                device=device_ini.device,
            )
            .reshape(1, -1)
            .to(face_vertices.dtype)
        )
        face_ids = torch.sum(face_vertices * multipliers, dim=1)
        sort_indices = torch.argsort(face_ids)
        face_ids = face_ids[sort_indices]
        face_indices = face_indices[sort_indices]

        EToE = (
            torch.arange(0, n_elements, device=device_ini.device)
            .reshape(1, -1)
            .repeat(self.faces_per_element, 1)
        )
        EToF = (
            torch.arange(0, self.faces_per_element, device=device_ini.device)
            .repeat_interleave(n_elements)
            .reshape(-1, n_elements)
        )

        interior = face_ids[:-1] == face_ids[1:]
        left = face_indices[:-1][interior]
        right = face_indices[1:][interior]
        left_element = torch.remainder(left, n_elements)
        right_element = torch.remainder(right, n_elements)
        left_face = torch.div(left, n_elements, rounding_mode="floor")
        right_face = torch.div(right, n_elements, rounding_mode="floor")
        EToE[left_face, left_element] = right_element
        EToF[left_face, left_element] = right_face
        EToE[right_face, right_element] = left_element
        EToF[right_face, right_element] = left_face
        return EToE, EToF
