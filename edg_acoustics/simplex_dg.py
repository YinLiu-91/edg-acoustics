"""Dimension-aware simplex DG helper functions."""

from __future__ import annotations

import math

import modepy
import numpy
import torch

import edg_acoustics.device_ini as device_ini


def simplex_num_nodes(dim: int, Nx: int):
    return math.comb(Nx + dim, dim)


def simplex_num_face_nodes(dim: int, Nx: int):
    return math.comb(Nx + dim - 1, dim - 1)


def triangle_collocation_nodes(EToV: torch.Tensor, vertices: torch.Tensor, Nx: int):
    rst = torch.from_numpy(modepy.warp_and_blend_nodes(2, Nx)).to(device_ini.device)
    xyz = torch.zeros(
        [2, rst.shape[1], EToV.shape[1]],
        device=device_ini.device,
        dtype=device_ini.dtype,
    )
    v0, v1, v2 = EToV[0], EToV[1], EToV[2]
    r = rst[0].reshape(-1, 1)
    s = rst[1].reshape(-1, 1)
    xyz[0] = 0.5 * (
        -(r + s) * vertices[0, v0]
        + (1.0 + r) * vertices[0, v1]
        + (1.0 + s) * vertices[0, v2]
    )
    xyz[1] = 0.5 * (
        -(r + s) * vertices[1, v0]
        + (1.0 + r) * vertices[1, v1]
        + (1.0 + s) * vertices[1, v2]
    )
    return rst.cpu().numpy(), xyz


def simplex_vandermonde(dim: int, Nx: int, rst: numpy.ndarray):
    basis = modepy.simplex_onb(dim, Nx)
    return torch.as_tensor(
        modepy.vandermonde(basis, rst),
        device=device_ini.device,
        dtype=device_ini.dtype,
    )


def simplex_derivative_matrices(dim: int, Nx: int, rst: numpy.ndarray):
    basis = modepy.simplex_onb(dim, Nx)
    grad_basis = modepy.grad_simplex_onb(dim, Nx)
    matrices = modepy.differentiation_matrices(basis, grad_basis, rst)
    return tuple(
        torch.as_tensor(matrix, device=device_ini.device, dtype=device_ini.dtype)
        for matrix in matrices
    )


def triangle_fmask(rst: torch.Tensor, node_tol: float):
    Nx = _triangle_degree_from_np(rst.shape[1])
    Nfp = simplex_num_face_nodes(2, Nx)
    Fmask = torch.zeros((3, Nfp), device=device_ini.device, dtype=torch.int32)
    Fmask[0] = torch.nonzero(torch.abs(1.0 + rst[1]) < node_tol).flatten()
    Fmask[1] = torch.nonzero(torch.abs(rst.sum(dim=0)) < node_tol).flatten()
    Fmask[2] = torch.nonzero(torch.abs(1.0 + rst[0]) < node_tol).flatten()
    return Fmask


def triangle_lift(V: torch.Tensor, rst: torch.Tensor, Fmask: torch.Tensor):
    Np = V.shape[1]
    Nx = _triangle_degree_from_np(Np)
    Nfp = simplex_num_face_nodes(2, Nx)
    Emat = torch.zeros((Np, 3 * Nfp), device=device_ini.device, dtype=device_ini.dtype)
    basis_1d = modepy.simplex_onb(1, Nx)
    face_coordinates = (
        rst[0, Fmask[0]],
        rst[0, Fmask[1]],
        rst[1, Fmask[2]],
    )
    for face, coordinate in enumerate(face_coordinates):
        v_face = modepy.vandermonde(basis_1d, coordinate.cpu().numpy().reshape(1, -1))
        mass_face = numpy.linalg.inv(v_face @ v_face.T)
        Emat[Fmask[face], face * Nfp : (face + 1) * Nfp] = torch.as_tensor(
            mass_face, device=device_ini.device, dtype=device_ini.dtype
        )
    return V @ (V.T @ Emat)


def triangle_geometric_factors(xyz: torch.Tensor, Dr: torch.Tensor, Ds: torch.Tensor):
    xr = Dr @ xyz[0]
    xs = Ds @ xyz[0]
    yr = Dr @ xyz[1]
    ys = Ds @ xyz[1]
    J = xr * ys - xs * yr
    rst_xyz = torch.zeros(
        (2, 2, xyz.shape[1], xyz.shape[2]),
        device=device_ini.device,
        dtype=device_ini.dtype,
    )
    rst_xyz[0, 0] = ys / J
    rst_xyz[0, 1] = -xs / J
    rst_xyz[1, 0] = -yr / J
    rst_xyz[1, 1] = xr / J
    return rst_xyz, torch.abs(J)


def triangle_normals(
    vertices: torch.Tensor,
    EToV: torch.Tensor,
    J: torch.Tensor,
    Fmask: torch.Tensor,
):
    n_elements = EToV.shape[1]
    Nfp = Fmask.shape[1]
    faces = torch.tensor([[0, 1], [1, 2], [0, 2]], device=device_ini.device)
    opposite = torch.tensor([2, 0, 1], device=device_ini.device)
    n_xy = torch.empty(
        (2, 3 * Nfp, n_elements), device=device_ini.device, dtype=device_ini.dtype
    )
    sJ = torch.empty(
        (3 * Nfp, n_elements), device=device_ini.device, dtype=device_ini.dtype
    )
    reference_lengths = torch.tensor(
        [2.0, 2.0**0.5 * 2.0, 2.0], device=device_ini.device, dtype=device_ini.dtype
    )

    xy = vertices[:2].to(device_ini.device, dtype=device_ini.dtype)
    for face in range(3):
        a = EToV[faces[face, 0]]
        b = EToV[faces[face, 1]]
        c = EToV[opposite[face]]
        va = xy[:, a]
        vb = xy[:, b]
        vc = xy[:, c]
        tangent = vb - va
        normal = torch.stack((tangent[1], -tangent[0]), dim=0)
        midpoint = 0.5 * (va + vb)
        inward_test = ((vc - midpoint) * normal).sum(dim=0)
        normal[:, inward_test > 0] *= -1.0
        edge_length = torch.linalg.vector_norm(tangent, dim=0)
        unit_normal = normal / edge_length.unsqueeze(0)
        rows = slice(face * Nfp, (face + 1) * Nfp)
        n_xy[:, rows, :] = unit_normal[:, None, :]
        sJ[rows, :] = (edge_length / reference_lengths[face]).reshape(1, -1)
    return n_xy, sJ


def simplex_build_maps(
    nodeids: torch.Tensor,
    xyz: torch.Tensor,
    EToE: torch.Tensor,
    EToF: torch.Tensor,
    Fmask: torch.Tensor,
    node_tol: float,
):
    n_faces, Nfp = Fmask.shape
    n_elements = EToE.shape[1]
    fmask = Fmask.to(device=device_ini.device, dtype=torch.long)
    vmapM = nodeids[fmask, :].contiguous()
    neighbor_faces = EToF.to(device=device_ini.device, dtype=torch.long)
    neighbor_elements = EToE.to(device=device_ini.device, dtype=torch.long)
    neighbor_vmap = vmapM[neighbor_faces, :, neighbor_elements]
    local_rows = vmapM.permute(0, 2, 1).reshape(-1, Nfp)
    neighbor_rows = neighbor_vmap.reshape(-1, Nfp)
    vmapP_rows = torch.empty_like(local_rows)
    coordinates = xyz.permute(1, 2, 0).reshape(-1, xyz.shape[0])
    tolerance_squared = node_tol * node_tol

    for start in range(0, local_rows.shape[0], 8192):
        end = min(start + 8192, local_rows.shape[0])
        local_ids = local_rows[start:end]
        neighbor_ids = neighbor_rows[start:end]
        local_xyz = coordinates[local_ids]
        neighbor_xyz = coordinates[neighbor_ids]
        distances = ((local_xyz[:, :, None, :] - neighbor_xyz[:, None, :, :]) ** 2).sum(
            dim=-1
        )
        matches = torch.argmin(distances, dim=2)
        min_distances = distances.gather(2, matches.unsqueeze(-1)).squeeze(-1)
        if bool(torch.any(min_distances > tolerance_squared)):
            raise ValueError("Failed to match simplex face nodes.")
        vmapP_rows[start:end] = neighbor_ids.gather(1, matches)

    vmapP = vmapP_rows.reshape(n_faces, n_elements, Nfp).permute(0, 2, 1).contiguous()
    return vmapM.to(torch.int32).reshape(-1), vmapP.to(torch.int32).reshape(-1)


def simplex_build_bcmaps(
    face_vertex_ids: torch.Tensor,
    BC_list: dict[str, int],
    EToV: torch.Tensor,
    vmapM: torch.Tensor,
    BC_faces: dict[str, torch.Tensor],
    Fmask: torch.Tensor,
):
    n_faces, Nfp = Fmask.shape
    n_elements = EToV.shape[1]
    bc_type = torch.zeros((n_faces, n_elements), device=device_ini.device, dtype=torch.int32)
    for name, label in BC_list.items():
        boundary, _ = BC_faces[name].sort(axis=1)
        for face in range(n_faces):
            face_vertices, _ = torch.sort(EToV[face_vertex_ids[face]], axis=0)
            is_boundary = _ismember_col(face_vertices, boundary)
            bc_type[face, is_boundary] = label
    bc_type = bc_type.repeat_interleave(Nfp, dim=0)
    nodes = []
    for _, label in BC_list.items():
        node = {"label": label}
        node["map"] = torch.nonzero(bc_type.reshape(-1) == label).t()[0].to(device_ini.device)
        node["vmap"] = vmapM[node["map"]]
        nodes.append(node)
    return nodes


def diameter_2d(Fscale: torch.Tensor):
    Nfp = Fscale.shape[0] // 3
    perimeter_to_area = Fscale[[0, Nfp, 2 * Nfp]].sum(axis=0)
    return (2.0 / perimeter_to_area).min().item()


def _triangle_degree_from_np(Np: int):
    return int(round((math.sqrt(8 * Np + 1) - 3) / 2))


def _ismember_col(a: torch.Tensor, b: torch.Tensor):
    rows = torch.cat((a.T, b), dim=0)
    _, inverse = torch.unique(rows, dim=0, return_inverse=True)
    return torch.isin(inverse[: a.shape[1]], inverse[a.shape[1] :])
