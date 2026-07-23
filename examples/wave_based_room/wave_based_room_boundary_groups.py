#!/usr/bin/env python3
"""Recover and validate COMSOL wave-based-room boundary groups."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import meshio
import numpy


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_MPH = CASE_DIR / "wave_based_room.mph"

PHYSICAL_GROUP_LABELS = {
    "default_hard_wall": 11,
    "carpet": 12,
    "ceiling": 13,
    "sofa": 14,
    "wall": 15,
    "normal_velocity_source": 21,
}

PHYSICAL_GROUP_NAMES = {
    "default_hard_wall": "DefaultHardWall",
    "carpet": "Carpet",
    "ceiling": "Ceiling",
    "sofa": "Sofa",
    "wall": "Wall",
    "normal_velocity_source": "NormalVelocitySource",
}

BOUNDARY_KIND = {
    "default_hard_wall": "sound_hard",
    "carpet": "rational_admittance",
    "ceiling": "rational_admittance",
    "sofa": "rational_admittance",
    "wall": "rational_admittance",
    "normal_velocity_source": "normal_velocity_source",
}

PHYSICS_TO_GROUP = {
    "imp1": "carpet",
    "imp2": "ceiling",
    "imp3": "sofa",
    "imp4": "wall",
    "nvel1": "normal_velocity_source",
}

GMSH_PHYSICAL_GROUP_ORDER = (
    "default_hard_wall",
    "carpet",
    "ceiling",
    "sofa",
    "wall",
    "normal_velocity_source",
)

ACTIVE_GROUP_ENTITIES = {
    "carpet": [3, 75],
    "ceiling": [7, 77],
    "sofa": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29, 30, 31, 51, 52, 61, 68, 69, 70, 71],
    "wall": [1, 2, 4, 5, 8, 9, 74, 78, 262],
    "normal_velocity_source": [222],
}

ACOUSTIC_DOMAINS = [1, 2, 3, 4]
COMSOL_RECEIVER_POINT_IDS = [122, 121, 53, 35]
COMSOL_RECEIVER_COORDS = [
    [1.2, 0.2, -0.8, -1.8],
    [1.3125, 0.875, 0.4375, 0.0],
    [1.0, 1.0, 1.0, 1.0],
]


def physical_groups(exported_boundary_refs: list[int] | None = None) -> list[dict[str, Any]]:
    group_entities = {key: sorted(value) for key, value in ACTIVE_GROUP_ENTITIES.items()}
    if exported_boundary_refs is None:
        default_entities: list[int] = []
    else:
        active = set().union(*(set(values) for values in group_entities.values()))
        default_entities = sorted(set(exported_boundary_refs) - active)
    group_entities["default_hard_wall"] = default_entities

    groups = []
    for key in GMSH_PHYSICAL_GROUP_ORDER:
        groups.append(
            {
                "key": key,
                "name": PHYSICAL_GROUP_NAMES[key],
                "label": PHYSICAL_GROUP_LABELS[key],
                "kind": BOUNDARY_KIND[key],
                "entities": group_entities[key],
                "entity_count": len(group_entities[key]),
                "physics_feature": _physics_feature_for_group(key),
            }
        )
    return groups


def _physics_feature_for_group(group_key: str) -> str | None:
    inverse = {value: key for key, value in PHYSICS_TO_GROUP.items()}
    return inverse.get(group_key)


def recover_boundary_model(exported_boundary_refs: list[int] | None = None) -> dict[str, Any]:
    return {
        "metadata": {
            "model": str(DEFAULT_MPH),
            "comsol_model_version": "6.4.0.250",
            "recovered_from": "wave_based_room.mph dmodel.xml/smodel.json and COMSOL Java inspection",
        },
        "physical_groups": physical_groups(exported_boundary_refs),
        "acoustic_domains": ACOUSTIC_DOMAINS,
        "receiver_point_ids": COMSOL_RECEIVER_POINT_IDS,
        "receiver_coords": COMSOL_RECEIVER_COORDS,
        "source": {
            "kind": "normal_velocity",
            "feature": "nvel1",
            "boundary_entities": ACTIVE_GROUP_ENTITIES["normal_velocity_source"],
            "expression": "vn(t)",
            "amplitude": 1.0,
            "frequency_hz": 700.0,
            "delay_s": 2.0 / 700.0,
            "sigma_s": 0.5 / 700.0,
        },
        "study": {
            "dataset": "dset1",
            "output": "range(0,T0,30*T0)",
            "T0": 1.0 / 700.0,
            "Tend": 30.0 / 700.0,
        },
        "notes": [
            "DefaultHardWall is computed from exported exterior boundary refs minus active impedance and source features.",
            "COMSOL probe table tbl1 stores normalized listening-point pressure pate.p_t/(1[m/s]*pate.Z).",
            "EDG material files use reflection R=(1-rho0*c0*Y)/(1+rho0*c0*Y) converted from COMSOL PFF admittance.",
        ],
    }


def validate_physical_groups(boundary_model: dict[str, Any], boundary_refs: list[int]) -> dict[str, Any]:
    refs = set(boundary_refs)
    occurrences: Counter[int] = Counter()
    empty: list[str] = []
    missing: dict[str, list[int]] = {}
    for group in boundary_model["physical_groups"]:
        entities = set(group["entities"])
        if not entities:
            empty.append(group["key"])
        missing_entities = sorted(entities - refs)
        if missing_entities:
            missing[group["key"]] = missing_entities
        occurrences.update(entities)

    covered = set(occurrences)
    duplicate_entities = sorted(entity for entity, count in occurrences.items() if count > 1)
    uncovered = sorted(refs - covered)
    return {
        "ok": not missing and not empty and not duplicate_entities and not uncovered,
        "surface_count": len(boundary_refs),
        "covered_surface_count": len(covered),
        "uncovered_surface_count": len(uncovered),
        "uncovered_surfaces": uncovered,
        "duplicate_surfaces": duplicate_entities,
        "missing_surfaces_by_group": missing,
        "empty_groups": empty,
    }


def mesh_diagnostics(mesh_path: Path) -> dict[str, Any]:
    mesh = meshio.read(mesh_path)
    points = numpy.asarray(mesh.points)
    diagnostics: dict[str, Any] = {
        "points": int(points.shape[0]),
        "bbox_min": points.min(axis=0).tolist(),
        "bbox_max": points.max(axis=0).tolist(),
        "bbox_size": (points.max(axis=0) - points.min(axis=0)).tolist(),
        "cells": {block.type: int(len(block.data)) for block in mesh.cells},
        "physical_tags": {},
        "geometrical_tags": {},
    }
    for key, target in (
        ("gmsh:physical", diagnostics["physical_tags"]),
        ("gmsh:geometrical", diagnostics["geometrical_tags"]),
    ):
        for cell_type, data in mesh.cell_data_dict.get(key, {}).items():
            values, counts = numpy.unique(data, return_counts=True)
            target[cell_type] = {
                str(int(value)): int(count)
                for value, count in zip(values, counts)
            }

    tets = _cells_by_type(mesh, "tetra")
    if tets.size:
        volumes = _tet_volumes(points, tets)
        edges = _tet_edge_lengths(points, tets)
        insphere = _tet_insphere_diameters(points, tets, volumes)
        diagnostics["tetra_quality"] = {
            "min_volume": float(volumes.min()),
            "median_volume": float(numpy.median(volumes)),
            "max_volume": float(volumes.max()),
            "min_edge_length": float(edges.min()),
            "median_edge_length": float(numpy.median(edges)),
            "max_edge_length": float(edges.max()),
            "min_insphere_diameter": float(insphere.min()),
            "median_insphere_diameter": float(numpy.median(insphere)),
        }
    triangles = _cells_by_type(mesh, "triangle")
    if triangles.size:
        areas = _triangle_areas(points, triangles)
        diagnostics["triangle_quality"] = {
            "min_area": float(areas.min()),
            "median_area": float(numpy.median(areas)),
            "max_area": float(areas.max()),
        }
        diagnostics["boundary_topology"] = _boundary_topology(mesh, triangles, tets)
    return diagnostics


def _cells_by_type(mesh: Any, cell_type: str) -> numpy.ndarray:
    blocks = [block.data for block in mesh.cells if block.type == cell_type]
    if not blocks:
        return numpy.empty((0, 4 if cell_type == "tetra" else 3), dtype=int)
    return numpy.concatenate(blocks, axis=0)


def _tet_volumes(points: numpy.ndarray, tets: numpy.ndarray) -> numpy.ndarray:
    a = points[tets[:, 0]]
    b = points[tets[:, 1]]
    c = points[tets[:, 2]]
    d = points[tets[:, 3]]
    return numpy.abs(numpy.einsum("ij,ij->i", numpy.cross(b - a, c - a), d - a)) / 6.0


def _tet_edge_lengths(points: numpy.ndarray, tets: numpy.ndarray) -> numpy.ndarray:
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    return numpy.concatenate(
        [numpy.linalg.norm(points[tets[:, i]] - points[tets[:, j]], axis=1) for i, j in pairs]
    )


def _tet_insphere_diameters(points: numpy.ndarray, tets: numpy.ndarray, volumes: numpy.ndarray) -> numpy.ndarray:
    surface_area = numpy.zeros(len(tets), dtype=float)
    for i, j, k in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        a = points[tets[:, i]]
        b = points[tets[:, j]]
        c = points[tets[:, k]]
        surface_area += numpy.linalg.norm(numpy.cross(b - a, c - a), axis=1) / 2.0
    return 6.0 * volumes / surface_area


def _triangle_areas(points: numpy.ndarray, triangles: numpy.ndarray) -> numpy.ndarray:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return numpy.linalg.norm(numpy.cross(b - a, c - a), axis=1) / 2.0


def _boundary_topology(mesh: meshio.Mesh, triangles: numpy.ndarray, tets: numpy.ndarray) -> dict[str, Any]:
    if not tets.size:
        return {}
    tetra_faces = numpy.concatenate(
        (
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        )
    )
    face_counts = Counter(map(tuple, numpy.sort(tetra_faces, axis=1).tolist()))
    multiplicity = Counter(
        face_counts.get(tuple(face), 0)
        for face in numpy.sort(triangles, axis=1).tolist()
    )
    return {
        "all_shells_are_exterior": set(multiplicity) <= {1},
        "shell_face_multiplicity": {str(key): int(value) for key, value in sorted(multiplicity.items())},
        "topological_boundary_faces": int(sum(count == 1 for count in face_counts.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    boundary_refs = None
    diagnostics = None
    if args.mesh is not None:
        diagnostics = mesh_diagnostics(args.mesh)
        boundary_refs = sorted(
            int(value)
            for value in diagnostics.get("geometrical_tags", {}).get("triangle", {})
        )
    model = recover_boundary_model(boundary_refs)
    report: dict[str, Any] = {"boundary_model": model}
    if boundary_refs is not None:
        report["boundary_validation"] = validate_physical_groups(model, boundary_refs)
        report["diagnostics"] = diagnostics
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if not report.get("boundary_validation") or report["boundary_validation"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
