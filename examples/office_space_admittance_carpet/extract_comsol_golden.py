#!/usr/bin/env python3
"""Convert ExportOfficeSpaceGolden batch-log records to canonical CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy


POINT_IDS = [230, 233, 467]
EXPECTED_NSAMPLES = 9001
OUTPUT_DT = (1.0 / 750.0) / 30.0


def parse_golden_log(log_path: Path) -> tuple[numpy.ndarray, numpy.ndarray, dict[int, list[float]]]:
    samples: list[list[float]] = []
    coordinates: dict[int, list[float]] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("OFFICE_GOLDEN_SAMPLE,"):
            fields = line.split(",")
            if len(fields) != 5:
                raise ValueError(f"Malformed golden sample record: {line}")
            samples.append([float(value) for value in fields[1:]])
        elif line.startswith("OFFICE_GOLDEN_RECEIVER,"):
            fields = line.split(",")
            if len(fields) != 6 or fields[-1] != "m":
                raise ValueError(f"Malformed golden receiver record: {line}")
            coordinates[int(fields[1])] = [float(value) for value in fields[2:5]]

    if len(samples) != EXPECTED_NSAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_NSAMPLES} golden samples, found {len(samples)} in {log_path}"
        )
    if sorted(coordinates) != POINT_IDS:
        raise ValueError(
            f"Expected receiver ids {POINT_IDS}, found {sorted(coordinates)} in {log_path}"
        )
    table = numpy.asarray(samples, dtype=float)
    expected_time = numpy.arange(EXPECTED_NSAMPLES, dtype=float) * OUTPUT_DT
    expected_time[-1] = 0.4
    if not numpy.allclose(table[:, 0], expected_time, rtol=0.0, atol=1.0e-12):
        raise ValueError("COMSOL golden sample times do not match range(0,T0/30,0.4)")
    if not numpy.all(numpy.isfinite(table)):
        raise ValueError("COMSOL golden contains NaN or Inf")
    return table[:, 0], table[:, 1:].T, coordinates


def write_golden_csv(
    output_path: Path,
    time: numpy.ndarray,
    pressure: numpy.ndarray,
    coordinates: dict[int, list[float]],
) -> None:
    with output_path.open("w", encoding="utf-8") as output:
        output.write("# COMSOL Response in Points pg5/ptgr1\n")
        output.write("# dataset,dset2\n")
        output.write("# expression,pate.p_t\n")
        output.write("# unit,Pa\n")
        output.write("# receiver_point_ids,230,233,467\n")
        output.write("# receiver_coordinate_unit,m\n")
        for point_id in POINT_IDS:
            x, y, z = coordinates[point_id]
            output.write(f"# receiver_coordinate,{point_id},{x:.17g},{y:.17g},{z:.17g}\n")
        output.write("time,p230,p233,p467\n")
        for sample, sample_time in enumerate(time):
            output.write(
                f"{sample_time:.17g},{pressure[0, sample]:.17g},"
                f"{pressure[1, sample]:.17g},{pressure[2, sample]:.17g}\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receiver-json", type=Path, required=True)
    args = parser.parse_args()

    time, pressure, coordinates = parse_golden_log(args.log)
    receiver_data = json.loads(args.receiver_json.read_text(encoding="utf-8"))
    expected = numpy.asarray(receiver_data["coords"], dtype=float)
    recovered = numpy.asarray([coordinates[point_id] for point_id in POINT_IDS], dtype=float).T
    if not numpy.allclose(expected, recovered, rtol=0.0, atol=1.0e-12):
        raise ValueError("Golden receiver coordinates do not match receiver JSON")
    write_golden_csv(args.output, time, pressure, coordinates)
    print(f"Wrote COMSOL golden CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
