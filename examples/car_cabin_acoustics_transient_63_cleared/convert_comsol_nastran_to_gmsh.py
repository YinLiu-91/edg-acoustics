#!/usr/bin/env python3
"""Convert COMSOL mesh2 NASTRAN export to EDG/Gmsh physical tags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import meshio
import numpy

import car_cabin_boundary_groups as groups


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_MPH = CASE_DIR / "car_cabin_acoustics_transient_63_cleared.mph"
DEFAULT_NASTRAN = CASE_DIR / "car_cabin_comsol_virtual_hmax0p114_hmin0p02.nas"
DEFAULT_MSH = CASE_DIR / "car_cabin_comsol_virtual_hmax0p114_hmin0p02.msh"


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
            if entity in mapping:
                raise ValueError(f"COMSOL boundary entity {entity} appears twice")
            mapping[int(entity)] = label
    return mapping


def convert_nastran_to_gmsh(
    nastran_path: Path,
    output_path: Path,
    mph_path: Path = DEFAULT_MPH,
) -> dict[str, Any]:
    mesh = meshio.read(nastran_path)
    boundary_model = groups.recover_boundary_model(mph_path)
    entity_to_label = _entity_to_physical_label(boundary_model)

    cells = []
    physical_data = []
    geometrical_data = []
    unknown_surface_refs: list[int] = []
    ref_offsets: dict[str, int] = {}

    for block in mesh.cells:
        if block.type not in {"triangle", "tetra"}:
            continue
        cells.append((block.type, block.data))
        refs = _nastran_refs(mesh, block.type)
        offset = ref_offsets.get(block.type, 0)
        block_refs = refs[offset : offset + len(block.data)].astype(int, copy=False)
        ref_offsets[block.type] = offset + len(block.data)
        geometrical_data.append(block_refs)
        if block.type == "tetra":
            physical_data.append(numpy.ones(len(block.data), dtype=int))
        else:
            labels = numpy.empty(len(block.data), dtype=int)
            for index, entity in enumerate(block_refs):
                label = entity_to_label.get(int(entity))
                if label is None:
                    unknown_surface_refs.append(int(entity))
                    label = 11
                labels[index] = label
            physical_data.append(labels)

    if unknown_surface_refs:
        unique = sorted(set(unknown_surface_refs))
        raise ValueError(
            "Boundary entities in NASTRAN export are not covered by MPH groups: "
            f"{unique[:20]}"
        )

    gmsh_mesh = meshio.Mesh(
        points=mesh.points,
        cells=cells,
        cell_data={
            "gmsh:physical": physical_data,
            "gmsh:geometrical": geometrical_data,
        },
    )
    meshio.write(output_path, gmsh_mesh, file_format="gmsh22", binary=False)
    diagnostics = groups.mesh_diagnostics(output_path)
    return {
        "input": str(nastran_path),
        "output": str(output_path),
        "diagnostics": diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nastran", type=Path, default=DEFAULT_NASTRAN)
    parser.add_argument("--mph", type=Path, default=DEFAULT_MPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MSH)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = convert_nastran_to_gmsh(args.nastran, args.output, args.mph)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
