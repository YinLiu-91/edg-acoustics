#!/usr/bin/env python3
"""Recover and validate COMSOL office-space boundary groups.

The office-space model uses COMSOL virtual geometry and an Absorbing Layer, so
the preferred mesh path is COMSOL ``mesh1`` exported to NASTRAN and converted
to Gmsh with explicit physical labels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_MPH = CASE_DIR / "office_space_acoustics_64_cleared.mph"

PHYSICAL_GROUP_LABELS = {
    "default_hard_wall": 11,
    "closed_windows": 12,
    "doors": 13,
    "brick_wall": 14,
    "carpet": 15,
    "ceiling": 16,
    "gypsum": 17,
    "open_window_absorbing_layer": 18,
}

PHYSICAL_GROUP_NAMES = {
    "default_hard_wall": "DefaultHardWall",
    "closed_windows": "ClosedWindows",
    "doors": "Doors",
    "brick_wall": "BrickWall",
    "carpet": "Carpet",
    "ceiling": "Ceiling",
    "gypsum": "Gypsum",
    "open_window_absorbing_layer": "OpenWindowAbsorbingLayer",
}

BOUNDARY_KIND = {
    "default_hard_wall": "sound_hard",
    "closed_windows": "constant_impedance",
    "doors": "constant_impedance",
    "brick_wall": "constant_impedance",
    "carpet": "rational_admittance",
    "ceiling": "rational_admittance",
    "gypsum": "rational_admittance",
    "open_window_absorbing_layer": "matched_absorbing_boundary_baseline",
}

PHYSICS_TO_GROUP = {
    "imp1": "closed_windows",
    "imp2": "doors",
    "imp4": "brick_wall",
    "imp3": "carpet",
    "imp5": "ceiling",
    "imp6": "gypsum",
    "imp7": "open_window_absorbing_layer",
}

GMSH_PHYSICAL_GROUP_ORDER = (
    "default_hard_wall",
    "closed_windows",
    "doors",
    "brick_wall",
    "carpet",
    "ceiling",
    "gypsum",
    "open_window_absorbing_layer",
)

# Reviewed active COMSOL selections from dmodel.xml/smodel.json.  The typo
# "Cypsum" is preserved by COMSOL, but this reproduction uses "gypsum".
ACTIVE_GROUP_ENTITIES = {
    "closed_windows": [13, 18, 42, 46, 54, 56, 58, 62, 70, 72, 74, 78, 86],
    "doors": [381, 389],
    "brick_wall": [383],
    "carpet": [35],
    "ceiling": [41],
    "gypsum": [33, 34, 88, 374, 375, 382],
    "open_window_absorbing_layer": [5, 6, 7, 8],
}

ACOUSTIC_DOMAINS = [48, 1, 2, 3, 4, 5, 6, 7, 8, 9, 46, 47]
ABSORBING_LAYER_DOMAINS = [48, 1, 2, 3, 4]
STORE_ON_POINT_IDS = [2, 230, 231, 232, 233, 455, 456, 467, 520]
RESPONSE_POINT_IDS = [230, 233, 467]

SELECTED_PARAMETER_NAMES = (
    "fc",
    "f0",
    "c0",
    "xs",
    "ys",
    "zs",
    "B",
    "S0",
    "rho0",
    "alpha_win",
    "Z_win",
    "alpha_door",
    "Z_door",
    "alpha_brick",
    "Z_brick",
    "T0",
    "T_ir",
    "hmin",
)


def _read_zip_text(path: Path, member: str) -> str:
    with ZipFile(path) as archive:
        return archive.read(member).decode("utf-8", errors="replace")


def _parse_entities(value: str | None) -> list[int]:
    return [int(item) for item in re.findall(r"-?\d+", value or "") if int(item) > 0]


def _parse_comsol_value_list(value: str | None) -> list[str]:
    if value is None or value == "":
        return []
    quoted = re.findall(r"'([^']*)'", value)
    if quoted:
        return quoted
    return [value]


def _feature_params(feature: ET.Element) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for param in feature.findall("param"):
        name = param.attrib.get("param")
        if not name:
            continue
        values = _parse_comsol_value_list(param.attrib.get("value"))
        params[name] = values[0] if len(values) == 1 else values
    return params


def _selection_entity_list(feature: ET.Element) -> list[int]:
    output = feature.find("outputSelection")
    if output is not None:
        for explicit in output.iter("explicit"):
            if explicit.attrib.get("dim") in {"2", "3", "0"}:
                return _parse_entities(explicit.attrib.get("entities"))
    return []


def _selection_ref_or_entities(
    feature: ET.Element,
    selections: dict[str, dict[str, Any]],
) -> tuple[str | None, list[int] | None]:
    selection = feature.find("selection")
    if selection is None:
        return None, []
    named = selection.find("named")
    if named is not None and (named.text or "").strip():
        ref = (named.text or "").strip()
        tag = ref.rsplit("/", 1)[-1]
        entities = selections.get(tag, {}).get("entities")
        return tag, list(entities) if entities is not None else []
    explicit = selection.find("explicit")
    if explicit is None:
        return None, []
    entities_attr = explicit.attrib.get("entities")
    if entities_attr is None:
        return None, None
    return None, _parse_entities(entities_attr)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_model_metadata(mph_path: Path = DEFAULT_MPH) -> dict[str, Any]:
    smodel = json.loads(_read_zip_text(mph_path, "smodel.json"))
    metadata = {
        key: smodel[key]
        for key in (
            "title",
            "name",
            "displayLabel",
            "lastComputationVersion",
            "lastComputationDate",
            "lastComputationTime",
        )
        if key in smodel
    }
    try:
        info = ET.fromstring(_read_zip_text(mph_path, "modelinfo.xml"))
        metadata["modelinfo_root"] = info.tag
    except Exception as exc:  # pragma: no cover - diagnostic field
        metadata["modelinfo_error"] = str(exc)
    return metadata


def load_parameters(mph_path: Path = DEFAULT_MPH) -> dict[str, dict[str, Any]]:
    smodel = json.loads(_read_zip_text(mph_path, "smodel.json"))
    parameters: dict[str, dict[str, Any]] = {}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            name = node.get("name")
            if name and {"value", "scalarReal"} <= set(node):
                parameters.setdefault(
                    name,
                    {
                        "value": node.get("value"),
                        "scalar_real": _maybe_float(node.get("scalarReal")),
                        "scalar_imag": _maybe_float(node.get("scalarImag")),
                        "description": node.get("description"),
                    },
                )
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(smodel)
    return parameters


def load_selections(mph_path: Path = DEFAULT_MPH) -> dict[str, dict[str, Any]]:
    root = ET.fromstring(_read_zip_text(mph_path, "dmodel.xml"))
    selections: dict[str, dict[str, Any]] = {}
    for feature in root.iter("SelectionFeature"):
        tag = feature.attrib.get("tag")
        if not tag:
            continue
        entities = _selection_entity_list(feature)
        if not entities:
            continue
        selections[tag] = {
            "tag": tag,
            "name": feature.attrib.get("name"),
            "op": feature.attrib.get("op"),
            "entities": entities,
            "entity_count": len(entities),
        }
    return selections


def load_physics_features(
    mph_path: Path = DEFAULT_MPH,
    selections: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    selections = selections or load_selections(mph_path)
    root = ET.fromstring(_read_zip_text(mph_path, "dmodel.xml"))
    features: dict[str, dict[str, Any]] = {}
    for feature in root.iter("PhysicsFeature"):
        op = feature.attrib.get("op")
        if op not in {"Impedance", "SoundHard", "PressureAcousticsTimeExplicitModel"}:
            continue
        tag = feature.attrib.get("tag")
        if not tag:
            continue
        selection_tag, entities = _selection_ref_or_entities(feature, selections)
        features[tag] = {
            "tag": tag,
            "name": feature.attrib.get("name"),
            "op": op,
            "selection_tag": selection_tag,
            "entities": entities,
            "entity_count": None if entities is None else len(entities),
            "params": _feature_params(feature),
        }
    return features


def selected_parameters(mph_path: Path = DEFAULT_MPH) -> dict[str, dict[str, Any]]:
    parameters = load_parameters(mph_path)
    return {name: parameters[name] for name in SELECTED_PARAMETER_NAMES if name in parameters}


def physical_groups(
    exported_boundary_refs: list[int] | None = None,
) -> list[dict[str, Any]]:
    group_entities = {key: sorted(values) for key, values in ACTIVE_GROUP_ENTITIES.items()}
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


def recover_boundary_model(
    mph_path: Path = DEFAULT_MPH,
    exported_boundary_refs: list[int] | None = None,
) -> dict[str, Any]:
    selections = load_selections(mph_path)
    physics = load_physics_features(mph_path, selections)
    return {
        "metadata": load_model_metadata(mph_path),
        "parameters": selected_parameters(mph_path),
        "selections": {
            tag: selections[tag]
            for tag in sorted(selections)
            if tag in {"sel1", "sel2", "sel3", "sel4", "sel5", "sel6", "sel7", "sel8", "sel9"}
        },
        "physics_features": {
            tag: physics[tag]
            for tag in sorted(physics)
            if tag in {"shb1", "imp1", "imp2", "imp3", "imp4", "imp5", "imp6", "imp7"}
        },
        "physical_groups": physical_groups(exported_boundary_refs),
        "acoustic_domains": ACOUSTIC_DOMAINS,
        "absorbing_layer_domains": ABSORBING_LAYER_DOMAINS,
        "response_point_ids": RESPONSE_POINT_IDS,
        "store_on_point_ids": STORE_ON_POINT_IDS,
        "source": {
            "kind": "initial_pressure_gaussian",
            "expression": "S0*exp(-log(2)*((x-xs)^2+(y-ys)^2+(z-zs)^2)/B^2)",
            "zero_initial_velocity": True,
        },
        "studies": {
            "std1": {"output": "range(0,T0/2,20*T0)", "purpose": "Store on Boundaries"},
            "std2": {"output": "range(0,T0/30,T_ir)", "purpose": "Store in Points"},
        },
        "notes": [
            "DefaultHardWall is computed from exported boundary refs minus active COMSOL physics groups.",
            "COMSOL Absorbing Layer domains are retained as ordinary air in the baseline EDG mesh; "
            "only the outer boundaries 5,6,7,8 are mapped to a matched RI=0 boundary.",
            "A full-fidelity match requires 3D absorbing-layer/PML support in EDG.",
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
        "ok": not empty and not missing and not duplicate_entities and not uncovered,
        "boundary_ref_count": len(refs),
        "covered_ref_count": len(covered),
        "uncovered_refs": uncovered,
        "duplicate_refs": duplicate_entities,
        "missing_refs_by_group": missing,
        "empty_groups": empty,
    }


def mesh_diagnostics(mesh_path: Path) -> dict[str, Any]:
    import meshio

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
            target[cell_type] = {str(int(v)): int(c) for v, c in zip(values, counts)}

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
        if tets.size:
            face_counts = Counter(
                map(
                    tuple,
                    numpy.sort(
                        numpy.concatenate(
                            (
                                tets[:, [0, 1, 2]],
                                tets[:, [0, 1, 3]],
                                tets[:, [0, 2, 3]],
                                tets[:, [1, 2, 3]],
                            )
                        ),
                        axis=1,
                    ).tolist(),
                )
            )
            shell_multiplicity = Counter(
                face_counts.get(tuple(face), 0)
                for face in numpy.sort(triangles, axis=1).tolist()
            )
            diagnostics["boundary_topology"] = {
                "shell_face_multiplicity": {
                    str(key): int(value) for key, value in sorted(shell_multiplicity.items())
                },
                "topological_boundary_faces": int(
                    sum(value == 1 for value in face_counts.values())
                ),
                "all_shells_are_exterior": shell_multiplicity == Counter({1: len(triangles)}),
            }
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


def _tet_insphere_diameters(
    points: numpy.ndarray,
    tets: numpy.ndarray,
    volumes: numpy.ndarray,
) -> numpy.ndarray:
    faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    area = numpy.zeros(len(tets), dtype=float)
    for i, j, k in faces:
        a = points[tets[:, i]]
        b = points[tets[:, j]]
        c = points[tets[:, k]]
        area += numpy.linalg.norm(numpy.cross(b - a, c - a), axis=1) / 2.0
    return 6.0 * volumes / area


def _triangle_areas(points: numpy.ndarray, triangles: numpy.ndarray) -> numpy.ndarray:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return numpy.linalg.norm(numpy.cross(b - a, c - a), axis=1) / 2.0


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, numpy.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mph", type=Path, default=DEFAULT_MPH)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--boundary-refs", type=int, nargs="*", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    refs = args.boundary_refs
    report: dict[str, Any] = {
        "mph": str(args.mph),
        "boundary_model": recover_boundary_model(args.mph, refs),
    }
    if refs is not None:
        report["validation"] = validate_physical_groups(report["boundary_model"], refs)
    if args.mesh is not None:
        report["mesh"] = str(args.mesh)
        report["mesh_diagnostics"] = mesh_diagnostics(args.mesh)

    text = json.dumps(report, indent=2, sort_keys=True, default=_json_default)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    validation = report.get("validation")
    return 0 if not validation or validation.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
