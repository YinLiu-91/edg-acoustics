#!/usr/bin/env python3
"""Recover and validate COMSOL car-cabin boundary groups.

The COMSOL ``.mph`` file is a ZIP archive.  For this example the named
boundary selections and pressure-acoustics boundary-condition features are
available in ``dmodel.xml`` and evaluated parameter values are available in
``smodel.json``.  This module extracts those definitions without requiring a
COMSOL installation, then maps them to stable Gmsh physical surface labels used
by the EDG mesh.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_MPH = EXAMPLE_DIR / "car_cabin_acoustics_transient_63_cleared.mph"
DEFAULT_STEP = EXAMPLE_DIR / "car_cabin_acoustics_transient_63_cleared.step"
DEFAULT_GEO = EXAMPLE_DIR / "car_cabin_acoustics_transient_63_cleared.geo"

SELECTION_TAGS = {
    "sel2": "windows",
    "sel3": "dashboard",
    "sel4": "carpet_floor",
    "sel5": "doors_selection",
    "sel6": "leather_seats",
    "sel7": "roof_trim",
    "sel8": "speaker_covers",
    "sel9": "midwoofer_l",
    "sel10": "midwoofer_r",
    "sel11": "tweeter_r",
    "sel12": "tweeter_l",
    "dif1": "comsol_sound_hard_surfaces_selection",
    "uni1": "all_speakers",
}

PHYSICAL_GROUP_LABELS = {
    "default_hard_wall": 11,
    "windows": 12,
    "dashboard": 13,
    "doors": 14,
    "leather_seats": 15,
    "carpet_floor": 16,
    "roof_trim": 17,
    "tweeter_l_source": 21,
    "inactive_speakers_hard_wall": 22,
}

PHYSICAL_GROUP_NAMES = {
    "default_hard_wall": "DefaultHardWall",
    "windows": "Windows",
    "dashboard": "Dashboard",
    "doors": "Doors",
    "leather_seats": "LeatherSeats",
    "carpet_floor": "CarpetFloor",
    "roof_trim": "RoofTrim",
    "tweeter_l_source": "TweeterLSource",
    "inactive_speakers_hard_wall": "InactiveSpeakersHardWall",
}

BOUNDARY_KIND = {
    "default_hard_wall": "sound_hard",
    "windows": "constant_impedance",
    "dashboard": "constant_impedance",
    "doors": "constant_impedance",
    "leather_seats": "rational_approximation_impedance",
    "carpet_floor": "rational_approximation_impedance",
    "roof_trim": "rational_approximation_impedance",
    "tweeter_l_source": "normal_velocity_source",
    "inactive_speakers_hard_wall": "sound_hard",
}

PHYSICS_TO_GROUP = {
    "imp1": "dashboard",
    "imp2": "doors",
    "imp3": "windows",
    "imp4": "leather_seats",
    "imp5": "carpet_floor",
    "imp6": "roof_trim",
    "nvel1": "tweeter_l_source",
}

GMSH_PHYSICAL_GROUP_ORDER = (
    "default_hard_wall",
    "windows",
    "dashboard",
    "doors",
    "leather_seats",
    "carpet_floor",
    "roof_trim",
    "tweeter_l_source",
    "inactive_speakers_hard_wall",
)


def _read_zip_text(path: Path, member: str) -> str:
    with ZipFile(path) as archive:
        return archive.read(member).decode("utf-8", errors="replace")


def _parse_entities(value: str | None) -> list[int]:
    """Parse a COMSOL explicit entity list and ignore sentinel negative ids."""

    return [int(item) for item in re.findall(r"-?\d+", value or "") if int(item) > 0]


def _parse_comsol_value_list(value: str | None) -> list[str]:
    """Parse COMSOL compact parameter values like ``2|1,'a'|1,'b'``."""

    if value is None or value == "":
        return []
    quoted = re.findall(r"'([^']*)'", value)
    if quoted:
        return quoted
    return [value]


def _single_value(value: str | None) -> str | None:
    values = _parse_comsol_value_list(value)
    if not values:
        return None
    return values[0]


def _selection_entity_list(feature: ET.Element) -> list[int]:
    output = feature.find("outputSelection")
    if output is None:
        return []
    for explicit in output.iter("explicit"):
        if explicit.attrib.get("dim") == "2":
            return _parse_entities(explicit.attrib.get("entities"))
    return []


def _feature_params(feature: ET.Element) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for param in feature.findall("param"):
        name = param.attrib.get("param")
        if not name:
            continue
        values = _parse_comsol_value_list(param.attrib.get("value"))
        params[name] = values[0] if len(values) == 1 else values
    return params


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
        # COMSOL uses an explicit selection without an entity list for the
        # default SoundHard feature.  It applies to all exterior boundaries and
        # is later overridden by more specific impedance/source features.
        return None, None
    return None, _parse_entities(entities_attr)


def load_model_metadata(mph_path: Path = DEFAULT_MPH) -> dict[str, Any]:
    """Return model title/version metadata from ``smodel.json`` and modelinfo."""

    metadata: dict[str, Any] = {}
    try:
        smodel = json.loads(_read_zip_text(mph_path, "smodel.json"))
        for key in (
            "title",
            "name",
            "lastComputationVersion",
            "lastComputationDate",
            "lastComputationTime",
        ):
            if key in smodel:
                metadata[key] = smodel[key]
    except Exception as exc:  # pragma: no cover - defensive report field
        metadata["smodel_error"] = str(exc)

    try:
        info = ET.fromstring(_read_zip_text(mph_path, "modelinfo.xml"))
        metadata["modelinfo_root"] = info.tag
    except Exception as exc:  # pragma: no cover - defensive report field
        metadata["modelinfo_error"] = str(exc)

    return metadata


def load_parameters(mph_path: Path = DEFAULT_MPH) -> dict[str, dict[str, Any]]:
    """Load evaluated scalar model parameters from ``smodel.json``."""

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


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_selections(mph_path: Path = DEFAULT_MPH) -> dict[str, dict[str, Any]]:
    """Load named COMSOL 2D boundary selections used by this example."""

    root = ET.fromstring(_read_zip_text(mph_path, "dmodel.xml"))
    selections: dict[str, dict[str, Any]] = {}
    for feature in root.iter("SelectionFeature"):
        tag = feature.attrib.get("tag")
        if tag not in SELECTION_TAGS:
            continue
        entities = _selection_entity_list(feature)
        selections[tag] = {
            "key": SELECTION_TAGS[tag],
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
    """Load pressure-acoustics boundary-condition feature definitions."""

    selections = selections or load_selections(mph_path)
    root = ET.fromstring(_read_zip_text(mph_path, "dmodel.xml"))
    features: dict[str, dict[str, Any]] = {}
    for feature in root.iter("PhysicsFeature"):
        op = feature.attrib.get("op")
        if op not in {"Impedance", "NormalVelocity", "SoundHard"}:
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


def gmsh_step_surface_tags(step_path: Path = DEFAULT_STEP, scale: float = 0.001) -> list[int]:
    """Return OCC surface tags after importing the STEP file with Gmsh."""

    import gmsh  # imported lazily so .mph parsing works without libGL/Gmsh setup

    gmsh.initialize([])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.OCCScaling", scale)
        gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
        gmsh.model.add("car_cabin_boundary_check")
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()
        return sorted(tag for dim, tag in gmsh.model.getEntities(2))
    finally:
        gmsh.finalize()


def recover_boundary_model(
    mph_path: Path = DEFAULT_MPH,
    surface_tags: list[int] | None = None,
) -> dict[str, Any]:
    """Recover EDG physical boundary groups from COMSOL selections/features."""

    selections = load_selections(mph_path)
    physics = load_physics_features(mph_path, selections)
    parameters = load_parameters(mph_path)

    group_entities: dict[str, list[int]] = {
        "windows": selections["sel2"]["entities"],
        "dashboard": selections["sel3"]["entities"],
        "doors": physics["imp2"]["entities"] or [],
        "leather_seats": selections["sel6"]["entities"],
        "carpet_floor": selections["sel4"]["entities"],
        "roof_trim": selections["sel7"]["entities"],
        "tweeter_l_source": selections["sel12"]["entities"],
        "inactive_speakers_hard_wall": sorted(
            set(selections["uni1"]["entities"]) - set(selections["sel12"]["entities"])
        ),
    }

    non_default = set().union(*(set(values) for values in group_entities.values()))
    if surface_tags is not None:
        default_entities = sorted(set(surface_tags) - non_default)
    else:
        # Without a STEP import, this is limited to the first 454 COMSOL ids.
        default_entities = sorted(
            set(selections["dif1"]["entities"]) - set(physics["imp2"]["entities"] or [])
        )
    group_entities["default_hard_wall"] = default_entities

    physical_groups = []
    for key in GMSH_PHYSICAL_GROUP_ORDER:
        entities = sorted(set(group_entities[key]))
        physics_feature = _physics_feature_for_group(key, physics)
        physical_groups.append(
            {
                "key": key,
                "name": PHYSICAL_GROUP_NAMES[key],
                "label": PHYSICAL_GROUP_LABELS[key],
                "kind": BOUNDARY_KIND[key],
                "entities": entities,
                "entity_count": len(entities),
                "physics_feature": physics_feature,
            }
        )

    return {
        "metadata": load_model_metadata(mph_path),
        "parameters": _selected_parameters(parameters),
        "selections": selections,
        "physics_features": physics,
        "physical_groups": physical_groups,
        "notes": [
            "default_hard_wall is computed as all imported STEP boundary surfaces "
            "minus impedance, active source, and separately tracked inactive speaker surfaces.",
            "COMSOL selection dif1 contains surfaces 298 and 302, but the active Door "
            "impedance feature imp2 also selects them; EDG follows the active physics feature.",
        ],
    }


def _physics_feature_for_group(
    group_key: str,
    physics: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    inverse = {value: key for key, value in PHYSICS_TO_GROUP.items()}
    tag = inverse.get(group_key)
    if tag is None:
        return None
    feature = physics[tag]
    params = feature.get("params", {})
    keep = {
        "ImpedanceModel",
        "Zn",
        "Y_inf",
        "R",
        "xi",
        "Q",
        "zeta",
        "ApproximantFunctionReference",
        "nvel",
    }
    return {
        "tag": feature["tag"],
        "name": feature["name"],
        "op": feature["op"],
        "selection_tag": feature["selection_tag"],
        "params": {key: params[key] for key in keep if key in params},
    }


def _selected_parameters(parameters: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    names = (
        "f0",
        "fmax",
        "c0",
        "rho0",
        "alpha_win",
        "Z_win",
        "alpha_dash",
        "Z_dash",
        "alpha_door",
        "Z_door",
        "d_carp",
        "Rf_carp",
        "d_roof",
        "Rf_roof",
        "T0",
        "Tend",
    )
    return {name: parameters[name] for name in names if name in parameters}


def validate_physical_groups(
    boundary_model: dict[str, Any],
    surface_tags: list[int],
) -> dict[str, Any]:
    """Validate that recovered physical groups cover the imported STEP boundary."""

    surface_set = set(surface_tags)
    occurrences: Counter[int] = Counter()
    missing: dict[str, list[int]] = {}
    empty: list[str] = []
    for group in boundary_model["physical_groups"]:
        entities = set(group["entities"])
        if not entities:
            empty.append(group["key"])
        not_in_step = sorted(entities - surface_set)
        if not_in_step:
            missing[group["key"]] = not_in_step
        occurrences.update(entities)

    covered = set(occurrences)
    duplicate_entities = sorted(entity for entity, count in occurrences.items() if count > 1)
    uncovered = sorted(surface_set - covered)

    return {
        "ok": not missing and not empty and not duplicate_entities and not uncovered,
        "surface_count": len(surface_tags),
        "covered_surface_count": len(covered),
        "uncovered_surface_count": len(uncovered),
        "uncovered_surfaces": uncovered,
        "duplicate_surfaces": duplicate_entities,
        "missing_surfaces_by_group": missing,
        "empty_groups": empty,
    }


def mesh_diagnostics(mesh_path: Path) -> dict[str, Any]:
    """Return basic mesh quality and physical-tag diagnostics."""

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
        data_by_type = mesh.cell_data_dict.get(key, {})
        for cell_type, data in data_by_type.items():
            values, counts = numpy.unique(data, return_counts=True)
            target[cell_type] = {
                str(int(value)): int(count)
                for value, count in zip(values, counts)
            }

    tets = _cells_by_type(mesh, "tetra")
    if tets.size:
        volumes = _tet_volumes(points, tets)
        edges = _tet_edge_lengths(points, tets)
        insphere_diameters = _tet_insphere_diameters(points, tets, volumes)
        diagnostics["tetra_quality"] = {
            "min_volume": float(volumes.min()),
            "median_volume": float(numpy.median(volumes)),
            "max_volume": float(volumes.max()),
            "min_edge_length": float(edges.min()),
            "median_edge_length": float(numpy.median(edges)),
            "max_edge_length": float(edges.max()),
            "min_insphere_diameter": float(insphere_diameters.min()),
            "median_insphere_diameter": float(numpy.median(insphere_diameters)),
        }

    triangles = _cells_by_type(mesh, "triangle")
    if triangles.size:
        areas = _triangle_areas(points, triangles)
        diagnostics["triangle_quality"] = {
            "min_area": float(areas.min()),
            "median_area": float(numpy.median(areas)),
            "max_area": float(areas.max()),
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
    face_indices = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    surface_area = numpy.zeros(len(tets), dtype=float)
    for i, j, k in face_indices:
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


def build_report(
    mph_path: Path = DEFAULT_MPH,
    step_path: Path | None = DEFAULT_STEP,
    mesh_path: Path | None = None,
    scale: float = 0.001,
) -> dict[str, Any]:
    """Build a JSON-serializable boundary and mesh validation report."""

    surface_tags = gmsh_step_surface_tags(step_path, scale=scale) if step_path else None
    model = recover_boundary_model(mph_path, surface_tags=surface_tags)
    report: dict[str, Any] = {
        "mph": str(mph_path),
        "step": str(step_path) if step_path else None,
        "scale": scale,
        "boundary_model": model,
    }
    if surface_tags is not None:
        report["step_surface_tags"] = {
            "count": len(surface_tags),
            "min": min(surface_tags) if surface_tags else None,
            "max": max(surface_tags) if surface_tags else None,
        }
        report["validation"] = validate_physical_groups(model, surface_tags)
    if mesh_path is not None:
        report["mesh"] = str(mesh_path)
        report["mesh_diagnostics"] = mesh_diagnostics(mesh_path)
    return report


def generate_mesh_for_check(
    geo_path: Path = DEFAULT_GEO,
    lc: float = 0.45,
    gmsh_bin: str = "gmsh",
) -> Path:
    """Generate a temporary mesh from the grouped ``.geo`` for diagnostics."""

    tmp_path = Path(tempfile.mkdtemp(prefix="car_cabin_gmsh_"))
    mesh_path = tmp_path / "car_cabin_check.msh"
    cmd = [
        gmsh_bin,
        "-3",
        str(geo_path),
        "-setnumber",
        "lc",
        str(lc),
        "-format",
        "msh2",
        "-o",
        str(mesh_path),
    ]
    subprocess.run(cmd, check=True, cwd=geo_path.parent)
    return mesh_path


def write_boundary_preview(mesh_path: Path, output_path: Path) -> None:
    """Write a simple physical-group colored boundary PNG for manual inspection."""

    import matplotlib.pyplot as plt
    import meshio
    from matplotlib.patches import Patch

    mesh = meshio.read(mesh_path)
    triangles = _cells_by_type(mesh, "triangle")
    if not triangles.size:
        raise ValueError(f"{mesh_path} has no triangle boundary cells")

    physical = mesh.cell_data_dict.get("gmsh:physical", {}).get("triangle")
    if physical is None:
        raise ValueError(f"{mesh_path} has no triangle gmsh:physical tags")

    centroids = mesh.points[triangles].mean(axis=1)
    labels = {value: key for key, value in PHYSICAL_GROUP_LABELS.items()}
    color_index = {label: index for index, label in enumerate(sorted(set(physical)))}
    colors = [color_index[int(tag)] for tag in physical]

    fig = plt.figure(figsize=(11, 6), dpi=180)
    axis = fig.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        c=colors,
        cmap="tab10",
        s=2,
        linewidths=0,
    )
    del scatter
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title("Car cabin boundary physical groups")
    handles = [
        Patch(
            facecolor=plt.get_cmap("tab10")(color_index[label] % 10),
            label=f"{label}: {labels.get(int(label), 'unknown')}",
        )
        for label in sorted(color_index)
    ]
    axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, numpy.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mph", type=Path, default=DEFAULT_MPH)
    parser.add_argument("--step", type=Path, default=DEFAULT_STEP)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--scale", type=float, default=0.001)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--generate-mesh",
        action="store_true",
        help="Generate a temporary mesh from the grouped .geo and include diagnostics.",
    )
    parser.add_argument("--geo", type=Path, default=DEFAULT_GEO)
    parser.add_argument("--lc", type=float, default=0.45)
    parser.add_argument("--gmsh-bin", default="gmsh")
    parser.add_argument(
        "--preview-out",
        type=Path,
        default=None,
        help="Write a physical-group colored boundary preview PNG.",
    )
    args = parser.parse_args(argv)

    mesh_path = args.mesh
    if args.generate_mesh:
        mesh_path = generate_mesh_for_check(args.geo, lc=args.lc, gmsh_bin=args.gmsh_bin)

    report = build_report(args.mph, args.step, mesh_path=mesh_path, scale=args.scale)
    if args.preview_out is not None:
        if mesh_path is None:
            mesh_path = generate_mesh_for_check(args.geo, lc=args.lc, gmsh_bin=args.gmsh_bin)
        write_boundary_preview(mesh_path, args.preview_out)
        report["preview"] = str(args.preview_out)

    output = json.dumps(report, indent=2, sort_keys=True, default=_json_default)
    if args.json_out is not None:
        args.json_out.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    validation = report.get("validation")
    return 0 if not validation or validation.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
