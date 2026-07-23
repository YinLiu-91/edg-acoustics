#!/usr/bin/env python3
"""Extract COMSOL receiver coordinates from ExportOfficeSpaceReceiverPoints log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


POINT_IDS = [230, 233, 467]
PREFIX = "OFFICE_RECEIVER_COORDINATE,"


def extract_receiver_points(log_path: Path) -> dict:
    coordinates: dict[int, list[float]] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(PREFIX):
            continue
        fields = line.split(",")
        if len(fields) != 6 or fields[-1] != "m":
            raise ValueError(f"Malformed receiver coordinate record: {line}")
        point_id = int(fields[1])
        coordinates[point_id] = [float(value) for value in fields[2:5]]

    if sorted(coordinates) != POINT_IDS:
        raise ValueError(
            f"Expected COMSOL point ids {POINT_IDS}, found {sorted(coordinates)} in {log_path}"
        )
    xyz_by_point = [coordinates[point_id] for point_id in POINT_IDS]
    return {
        "point_ids": POINT_IDS,
        "coords": [list(axis) for axis in zip(*xyz_by_point)],
        "geometry_length_unit": "cm",
        "coordinate_unit": "m",
        "source": "COMSOL comp1/geom1.getVertexCoord()",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = extract_receiver_points(args.log)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote receiver coordinates: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
