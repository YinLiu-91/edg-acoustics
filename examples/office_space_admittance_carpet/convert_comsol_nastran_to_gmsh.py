#!/usr/bin/env python3
"""Convert COMSOL office-space mesh1 NASTRAN export to EDG/Gmsh labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import meshio
import numpy

import office_space_boundary_groups as groups


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_MPH = CASE_DIR / "office_space_acoustics_64_cleared.mph"
DEFAULT_NASTRAN = CASE_DIR / "office_space_comsol_mesh1.nas"
DEFAULT_MSH = CASE_DIR / "office_space_comsol_mesh1.msh"
DEFAULT_POINT_SCALE = 0.01
VOLUME_LABEL = 1

# mesh1 uses COMSOL virtual geometry. Domain 48 remains in the physics and
# absorbing-layer selections, but has no exported CTETRA after mesh1.run().
REVIEWED_MISSING_VOLUME_REFS = {48}


def _cell_data(mesh: meshio.Mesh, key: str, cell_type: str) -> numpy.ndarray | None:
    values = mesh.cell_data_dict.get(key, {}).get(cell_type)
    if values is None:
        return None
    return numpy.asarray(values)


def _nastran_refs(mesh: meshio.Mesh, cell_type: str) -> numpy.ndarray:
    for key in ("nastran:ref", "gmsh:geometrical"):
        refs = _cell_data(mesh, key, cell_type)
        if refs is not None:
            return refs.astype(int, copy=False)
    raise ValueError(f"NASTRAN export has no geometric/property refs for {cell_type}")


def _entity_to_physical_label(boundary_model: dict[str, Any]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for group in boundary_model["physical_groups"]:
        label = int(group["label"])
        for entity in group["entities"]:
            entity = int(entity)
            if entity in mapping:
                raise ValueError(f"COMSOL boundary entity {entity} appears twice")
            mapping[entity] = label
    return mapping


def _cells_and_refs(mesh: meshio.Mesh, cell_type: str) -> tuple[numpy.ndarray, numpy.ndarray]:
    cells: list[numpy.ndarray] = []
    refs: list[numpy.ndarray] = []
    offsets: dict[str, int] = {}
    for block in mesh.cells:
        if block.type != cell_type:
            continue
        all_refs = _nastran_refs(mesh, cell_type)
        offset = offsets.get(cell_type, 0)
        cells.append(numpy.asarray(block.data, dtype=int))
        refs.append(all_refs[offset : offset + len(block.data)].astype(int, copy=False))
        offsets[cell_type] = offset + len(block.data)
    if not cells:
        width = 3 if cell_type == "triangle" else 4
        return numpy.empty((0, width), dtype=int), numpy.empty(0, dtype=int)
    return numpy.concatenate(cells), numpy.concatenate(refs)


def _exterior_shell_mask(
    triangles: numpy.ndarray,
    tetrahedra: numpy.ndarray,
) -> tuple[numpy.ndarray, dict[str, Any]]:
    tetra_faces = numpy.concatenate(
        (
            tetrahedra[:, [0, 1, 2]],
            tetrahedra[:, [0, 1, 3]],
            tetrahedra[:, [0, 2, 3]],
            tetrahedra[:, [1, 2, 3]],
        )
    )
    face_counts = Counter(map(tuple, numpy.sort(tetra_faces, axis=1).tolist()))
    sorted_shells = numpy.sort(triangles, axis=1)
    if len(numpy.unique(sorted_shells, axis=0)) != len(sorted_shells):
        raise ValueError("NASTRAN export contains duplicate shell triangles")

    multiplicity = numpy.asarray(
        [face_counts.get(tuple(face), 0) for face in sorted_shells], dtype=int
    )
    invalid = numpy.flatnonzero((multiplicity < 1) | (multiplicity > 2))
    if invalid.size:
        raise ValueError(
            "NASTRAN shell triangles must match one or two tetrahedron faces; "
            f"invalid shell indices: {invalid[:20].tolist()}"
        )

    topological_boundary_count = sum(count == 1 for count in face_counts.values())
    exterior_mask = multiplicity == 1
    if int(exterior_mask.sum()) != topological_boundary_count:
        raise ValueError(
            "NASTRAN shell export does not cover every topological tetrahedron boundary: "
            f"shell exterior={int(exterior_mask.sum())}, topology={topological_boundary_count}"
        )
    return exterior_mask, {
        "exported_shell_triangles": int(len(triangles)),
        "exterior_shell_triangles": int(exterior_mask.sum()),
        "discarded_internal_shell_triangles": int((multiplicity == 2).sum()),
        "topological_boundary_triangles": int(topological_boundary_count),
    }


def _volume_validation(tetra_refs: numpy.ndarray) -> dict[str, Any]:
    exported = set(map(int, tetra_refs.tolist()))
    expected = set(groups.ACOUSTIC_DOMAINS)
    missing = expected - exported
    unexpected = exported - expected
    unreviewed_missing = missing - REVIEWED_MISSING_VOLUME_REFS
    validation = {
        "ok": not unexpected and not unreviewed_missing,
        "expected_acoustic_domain_refs": sorted(expected),
        "exported_tetra_domain_refs": sorted(exported),
        "missing_expected_refs": sorted(missing),
        "reviewed_missing_virtual_geometry_refs": sorted(
            missing & REVIEWED_MISSING_VOLUME_REFS
        ),
        "unexpected_exported_refs": sorted(unexpected),
    }
    if not validation["ok"]:
        raise ValueError("Unexpected tetrahedron domain coverage: " + json.dumps(validation))
    return validation


def convert_nastran_to_gmsh(
    nastran_path: Path,
    output_path: Path,
    mph_path: Path = DEFAULT_MPH,
    point_scale: float = DEFAULT_POINT_SCALE,
) -> dict[str, Any]:
    mesh = meshio.read(nastran_path)
    triangles, triangle_refs = _cells_and_refs(mesh, "triangle")
    tetrahedra, tetra_refs = _cells_and_refs(mesh, "tetra")
    if triangles.size == 0 or tetrahedra.size == 0:
        raise ValueError("NASTRAN export must contain triangle shells and tetrahedron solids")

    exterior_mask, topology = _exterior_shell_mask(triangles, tetrahedra)
    internal_refs = triangle_refs[~exterior_mask]
    triangles = triangles[exterior_mask]
    triangle_refs = triangle_refs[exterior_mask]
    boundary_refs = sorted(set(map(int, triangle_refs.tolist())))
    volume_validation = _volume_validation(tetra_refs)
    boundary_model = groups.recover_boundary_model(mph_path, boundary_refs)
    validation = groups.validate_physical_groups(boundary_model, boundary_refs)
    if not validation["ok"]:
        raise ValueError(
            "Recovered physical groups do not cover the NASTRAN boundary refs: "
            + json.dumps(validation, sort_keys=True)
        )
    entity_to_label = _entity_to_physical_label(boundary_model)

    unknown_surface_refs: list[int] = []
    triangle_labels = numpy.empty(len(triangles), dtype=int)
    for index, entity in enumerate(triangle_refs):
        label = entity_to_label.get(int(entity))
        if label is None:
            unknown_surface_refs.append(int(entity))
            label = -1
        triangle_labels[index] = label

    if unknown_surface_refs:
        raise ValueError(
            "Boundary entities in NASTRAN export are not covered by recovered groups: "
            f"{sorted(set(unknown_surface_refs))[:40]}"
        )

    gmsh_mesh = meshio.Mesh(
        points=mesh.points * point_scale,
        cells=[("triangle", triangles), ("tetra", tetrahedra)],
        cell_data={
            "gmsh:physical": [
                triangle_labels,
                numpy.full(len(tetrahedra), VOLUME_LABEL, dtype=int),
            ],
            "gmsh:geometrical": [triangle_refs, tetra_refs],
        },
    )
    meshio.write(output_path, gmsh_mesh, file_format="gmsh22", binary=False)
    diagnostics = groups.mesh_diagnostics(output_path)
    return {
        "input": str(nastran_path),
        "output": str(output_path),
        "point_scale": point_scale,
        "boundary_refs": boundary_refs,
        "boundary_validation": validation,
        "volume_validation": volume_validation,
        "topology_validation": {
            **topology,
            "discarded_internal_entity_refs": sorted(set(map(int, internal_refs.tolist()))),
        },
        "diagnostics": diagnostics,
        "absorbing_layer_treatment": (
            "COMSOL absorbing-layer domains are retained as ordinary acoustic air; "
            "outer AL boundaries are mapped to RI=0."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nastran", type=Path, default=DEFAULT_NASTRAN)
    parser.add_argument("--mph", type=Path, default=DEFAULT_MPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MSH)
    parser.add_argument(
        "--point-scale",
        type=float,
        default=DEFAULT_POINT_SCALE,
        help="Scale COMSOL NASTRAN coordinates before writing Gmsh; office export is in cm.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = convert_nastran_to_gmsh(args.nastran, args.output, args.mph, args.point_scale)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
