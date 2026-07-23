#!/usr/bin/env python3
"""Convert ExportWaveBasedRoomGolden batch-log records to CSV and MAT golden files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy
import scipy.io


POINT_IDS = [122, 121, 53, 35]
TEND = 30.0 / 700.0
Z0 = 1.2 * 343.0


def parse_golden_log(log_path: Path) -> dict:
    pressure_rows: list[list[float]] = []
    normalized_rows: list[list[float]] = []
    stored_rows: list[list[float]] = []
    coordinates: dict[int, list[float]] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("WAVE_GOLDEN_SAMPLE,"):
            fields = line.split(",")
            if len(fields) != 6:
                raise ValueError(f"Malformed pressure sample record: {line}")
            pressure_rows.append([float(value) for value in fields[1:]])
        elif line.startswith("WAVE_GOLDEN_NORMALIZED,"):
            fields = line.split(",")
            if len(fields) != 6:
                raise ValueError(f"Malformed normalized sample record: {line}")
            normalized_rows.append([float(value) for value in fields[1:]])
        elif line.startswith("WAVE_GOLDEN_STORED_PA,"):
            fields = line.split(",")
            if len(fields) != 6:
                raise ValueError(f"Malformed stored pressure record: {line}")
            stored_rows.append([float(value) for value in fields[1:]])
        elif line.startswith("WAVE_GOLDEN_RECEIVER,"):
            fields = line.split(",")
            if len(fields) != 6 or fields[-1] != "m":
                raise ValueError(f"Malformed receiver record: {line}")
            coordinates[int(fields[1])] = [float(value) for value in fields[2:5]]

    if set(coordinates) != set(POINT_IDS):
        raise ValueError(f"Expected receiver ids {POINT_IDS}, found {sorted(coordinates)}")
    if not pressure_rows or len(pressure_rows) != len(normalized_rows):
        raise ValueError("Golden log is missing pressure or normalized probe records.")

    pressure_table = numpy.asarray(pressure_rows, dtype=float)
    normalized_table = numpy.asarray(normalized_rows, dtype=float)
    if not numpy.allclose(pressure_table[:, 0], normalized_table[:, 0], rtol=0.0, atol=1.0e-13):
        raise ValueError("Pressure and normalized golden times differ.")
    if not numpy.all(numpy.diff(pressure_table[:, 0]) > 0.0):
        raise ValueError("Golden times must be strictly increasing.")
    if pressure_table[0, 0] < -1.0e-12 or pressure_table[-1, 0] > TEND + 1.0e-10:
        raise ValueError("Golden time range lies outside [0, 30*T0].")
    if not numpy.all(numpy.isfinite(pressure_table)) or not numpy.all(numpy.isfinite(normalized_table)):
        raise ValueError("Golden table contains NaN or Inf.")

    stored_table = numpy.asarray(stored_rows, dtype=float) if stored_rows else numpy.empty((0, 5))
    return {
        "time": pressure_table[:, 0],
        "pressure_pa": pressure_table[:, 1:].T,
        "pressure_normalized": normalized_table[:, 1:].T,
        "stored_time": stored_table[:, 0] if stored_table.size else numpy.empty(0),
        "stored_pressure_pa": stored_table[:, 1:].T if stored_table.size else numpy.empty((4, 0)),
        "coordinates": coordinates,
    }


def validate_receiver_json(receiver_json: Path, coordinates: dict[int, list[float]]) -> numpy.ndarray:
    data = json.loads(receiver_json.read_text(encoding="utf-8"))
    point_ids = numpy.asarray(data["point_ids"], dtype=numpy.int32)
    if point_ids.tolist() != POINT_IDS:
        raise ValueError(f"receiver JSON point ids {point_ids.tolist()} do not match {POINT_IDS}")
    expected = numpy.asarray(data["coords"], dtype=float)
    recovered = numpy.asarray([coordinates[point_id] for point_id in POINT_IDS], dtype=float).T
    if not numpy.allclose(expected, recovered, rtol=0.0, atol=1.0e-12):
        raise ValueError("Golden receiver coordinates do not match receiver JSON")
    return recovered


def write_csv(output_path: Path, data: dict, receiver: numpy.ndarray) -> None:
    with output_path.open("w", encoding="utf-8") as output:
        output.write("# COMSOL wave_based_room listening point golden\n")
        output.write("# table,tbl1\n")
        output.write("# expression,pate.p_t and pate.p_t/(1[m/s]*pate.Z)\n")
        output.write("# pressure_unit,Pa\n")
        output.write("# normalized_unit,1\n")
        output.write("# receiver_point_ids,122,121,53,35\n")
        output.write("# receiver_coordinate_unit,m\n")
        for index, point_id in enumerate(POINT_IDS):
            output.write(
                f"# receiver_coordinate,{point_id},{receiver[0, index]:.17g},"
                f"{receiver[1, index]:.17g},{receiver[2, index]:.17g}\n"
            )
        output.write("time,p122,p121,p53,p35,pn122,pn121,pn53,pn35\n")
        pressure = data["pressure_pa"]
        normalized = data["pressure_normalized"]
        for sample, sample_time in enumerate(data["time"]):
            values = list(pressure[:, sample]) + list(normalized[:, sample])
            output.write(
                f"{sample_time:.17g}," + ",".join(f"{value:.17g}" for value in values) + "\n"
            )


def write_mat(output_path: Path, data: dict, receiver: numpy.ndarray) -> None:
    scipy.io.savemat(
        output_path,
        {
            "time": data["time"],
            "pressure_pa": data["pressure_pa"],
            "pressure_normalized": data["pressure_normalized"],
            "stored_time": data["stored_time"],
            "stored_pressure_pa": data["stored_pressure_pa"],
            "receiver_point_ids": numpy.asarray(POINT_IDS, dtype=numpy.int32),
            "receiver_coords": receiver,
            "rho0": 1.2,
            "c0": 343.0,
            "normalization_impedance": Z0,
            "source_table": "COMSOL tbl1",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--receiver-json", type=Path, required=True)
    args = parser.parse_args()

    data = parse_golden_log(args.log)
    receiver = validate_receiver_json(args.receiver_json, data["coordinates"])
    write_csv(args.csv, data, receiver)
    write_mat(args.mat, data, receiver)
    print(f"Wrote COMSOL golden CSV: {args.csv}")
    print(f"Wrote COMSOL golden MAT: {args.mat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
