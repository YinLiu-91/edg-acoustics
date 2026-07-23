#!/usr/bin/env python3
"""Compare EDG receiver histories against COMSOL office-space golden data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import scipy.io


COMSOL_POINT_IDS = numpy.array([230, 233, 467], dtype=numpy.int32)


def load_comsol_golden(
    path: Path,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None]:
    receiver_rows: dict[int, list[float]] = {}
    with path.open() as handle:
        lines = list(handle)
    table_lines = [
        line for line in lines if line.strip() and not line.lstrip().startswith("#")
    ]
    for line in lines:
        fields = [field.strip() for field in line.lstrip("# ").split(",")]
        if fields[0] == "receiver_coordinate" and len(fields) == 5:
            receiver_rows[int(fields[1])] = [float(value) for value in fields[2:]]
    if not table_lines:
        raise ValueError(f"{path} does not contain a CSV table.")

    data = numpy.genfromtxt(table_lines, delimiter=",", names=True)
    data = numpy.atleast_1d(data)
    expected_columns = ("time", "p230", "p233", "p467")
    if data.dtype.names != expected_columns:
        raise ValueError(
            f"Unexpected COMSOL golden columns {data.dtype.names}; expected {expected_columns}."
        )
    time = numpy.asarray(data["time"], dtype=float)
    pressure = numpy.vstack(
        [
            numpy.asarray(data["p230"], dtype=float),
            numpy.asarray(data["p233"], dtype=float),
            numpy.asarray(data["p467"], dtype=float),
        ]
    )
    if time.ndim != 1 or pressure.shape[1] != time.size:
        raise ValueError("Invalid COMSOL golden time/pressure shape.")
    if not numpy.all(numpy.diff(time) > 0.0):
        raise ValueError("COMSOL golden times must be strictly increasing.")
    receiver = None
    if receiver_rows:
        if set(receiver_rows) != set(COMSOL_POINT_IDS.tolist()):
            raise ValueError(
                f"COMSOL golden receiver metadata has point ids {sorted(receiver_rows)}; "
                f"expected {COMSOL_POINT_IDS.tolist()}."
            )
        receiver = numpy.asarray(
            [receiver_rows[int(point_id)] for point_id in COMSOL_POINT_IDS], dtype=float
        ).T
    return time, pressure, receiver


def load_edg_result(path: Path) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray | None]:
    data = scipy.io.loadmat(path)
    if "prec" not in data:
        raise ValueError(f"{path} does not contain 'prec'.")

    pressure = numpy.asarray(data["prec"], dtype=float)
    if pressure.ndim == 1:
        pressure = pressure.reshape(1, -1)
    if pressure.ndim != 2:
        raise ValueError(f"Expected EDG 'prec' to be 2D, got shape {pressure.shape}.")
    if pressure.shape[0] != COMSOL_POINT_IDS.size and pressure.shape[1] == COMSOL_POINT_IDS.size:
        pressure = pressure.T

    if "prec_times" in data and data["prec_times"].size == pressure.shape[1]:
        time = numpy.asarray(data["prec_times"], dtype=float).reshape(-1)
    elif "dt" in data:
        dt = float(numpy.asarray(data["dt"]).reshape(-1)[0])
        time = (numpy.arange(pressure.shape[1], dtype=float) + 1.0) * dt
    else:
        raise ValueError(f"{path} does not contain 'prec_times' or 'dt'.")

    receiver = numpy.asarray(data.get("rec", numpy.empty((0, 0))), dtype=float)
    receiver = numpy.squeeze(receiver)
    if receiver.shape == (3,):
        receiver = receiver.reshape(3, 1)
    if receiver.shape == (1, 3):
        receiver = receiver.reshape(3, 1)

    point_ids = None
    if "receiver_point_ids" in data:
        point_ids = numpy.asarray(data["receiver_point_ids"], dtype=numpy.int32).reshape(-1)

    if pressure.shape[1] != time.size:
        raise ValueError(
            f"EDG pressure/time length mismatch: prec={pressure.shape}, prec_times={time.shape}."
        )
    return time, pressure, receiver, point_ids


def load_expected_receiver(path: Path | None) -> numpy.ndarray | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    point_ids = numpy.asarray(data["point_ids"], dtype=numpy.int32)
    if not numpy.array_equal(point_ids, COMSOL_POINT_IDS):
        raise ValueError(f"receiver JSON point ids {point_ids.tolist()} do not match [230, 233, 467]")
    coords = numpy.asarray(data["coords"], dtype=float)
    if coords.shape != (3, 3):
        raise ValueError(f"receiver JSON coords must have shape (3, 3), got {coords.shape}")
    if data.get("coordinate_unit") != "m":
        raise ValueError("receiver JSON coordinate_unit must be 'm'")
    if not numpy.all(numpy.isfinite(coords)):
        raise ValueError("receiver JSON coords must all be finite")
    return coords


def validate_edg_receiver(
    receiver: numpy.ndarray,
    point_ids: numpy.ndarray | None,
    expected_receiver: numpy.ndarray | None,
    golden_receiver: numpy.ndarray | None,
) -> None:
    if point_ids is None:
        raise ValueError("EDG result is missing receiver_point_ids metadata.")
    if not numpy.array_equal(point_ids, COMSOL_POINT_IDS):
        raise ValueError(
            f"EDG receiver_point_ids={point_ids.tolist()} do not match {COMSOL_POINT_IDS.tolist()}."
        )
    if golden_receiver is None:
        raise ValueError(
            "COMSOL golden is missing receiver_coordinate metadata; re-export it with "
            "ExportOfficeSpaceGolden.java."
        )
    if expected_receiver is not None and not numpy.allclose(
        expected_receiver, golden_receiver, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("receiver JSON coordinates do not match COMSOL golden metadata.")
    if receiver.shape != golden_receiver.shape:
        raise ValueError(
            f"EDG receiver shape {receiver.shape} does not match expected {golden_receiver.shape}."
        )
    if not numpy.allclose(receiver, golden_receiver, rtol=0.0, atol=1.0e-12):
        raise ValueError("EDG receiver coordinates do not match COMSOL golden metadata.")


def interpolate_comsol_to_edg(
    comsol_time: numpy.ndarray,
    comsol_pressure: numpy.ndarray,
    edg_time: numpy.ndarray,
) -> numpy.ndarray:
    tolerance = max(1.0e-12, 1.0e-9 * (comsol_time[-1] - comsol_time[0]))
    if edg_time[0] < comsol_time[0] - tolerance or edg_time[-1] > comsol_time[-1] + tolerance:
        raise ValueError(
            "EDG time range is outside COMSOL golden range: "
            f"EDG [{edg_time[0]}, {edg_time[-1]}], COMSOL [{comsol_time[0]}, {comsol_time[-1]}]."
        )
    return numpy.vstack(
        [
            numpy.interp(edg_time, comsol_time, comsol_pressure[index])
            for index in range(comsol_pressure.shape[0])
        ]
    )


def _metric_block(error: numpy.ndarray, reference: numpy.ndarray) -> dict:
    ref_norm = numpy.linalg.norm(reference)
    return {
        "rms_abs": float(numpy.sqrt(numpy.mean(error**2))),
        "max_abs": float(numpy.max(numpy.abs(error))),
        "relative_l2": float(numpy.linalg.norm(error) / max(ref_norm, 1.0e-30)),
    }


def compute_metrics(
    edg_time: numpy.ndarray,
    edg_pressure: numpy.ndarray,
    reference_pressure: numpy.ndarray,
) -> dict:
    error = edg_pressure - reference_pressure
    per_receiver = []
    for index, point_id in enumerate(COMSOL_POINT_IDS.tolist()):
        block = _metric_block(error[index], reference_pressure[index])
        block["point_id"] = point_id
        per_receiver.append(block)

    windows = []
    for name, start, end in (
        ("pre_arrival", 0.0, 0.010),
        ("direct_and_early", 0.010, 0.060),
        ("mid_reverberation", 0.060, 0.200),
        ("late_tail", 0.200, float(edg_time[-1]) + 1.0e-15),
    ):
        mask = (edg_time >= start) & (edg_time < end)
        if numpy.any(mask):
            block = _metric_block(error[:, mask], reference_pressure[:, mask])
            block.update({"name": name, "time_start": start, "time_end": end, "num_samples": int(mask.sum())})
            windows.append(block)

    return {
        "num_samples": int(edg_time.size),
        "time_start": float(edg_time[0]),
        "time_end": float(edg_time[-1]),
        "receiver_point_ids": COMSOL_POINT_IDS.tolist(),
        "per_receiver": per_receiver,
        "windows": windows,
        "global": _metric_block(error, reference_pressure),
    }


def plot_comparison(
    output_path: Path,
    edg_time: numpy.ndarray,
    edg_pressure: numpy.ndarray,
    reference_pressure: numpy.ndarray,
) -> None:
    fig, axes = plt.subplots(
        nrows=COMSOL_POINT_IDS.size,
        ncols=2,
        figsize=(12.0, 8.2),
        sharex=True,
        constrained_layout=True,
    )
    for index, point_id in enumerate(COMSOL_POINT_IDS.tolist()):
        response_axis = axes[index, 0]
        error_axis = axes[index, 1]
        response_axis.plot(edg_time, reference_pressure[index], label="COMSOL", lw=1.4)
        response_axis.plot(edg_time, edg_pressure[index], label="EDG", lw=1.1, ls="--")
        response_axis.set_ylabel(f"p{point_id} (Pa)")
        response_axis.grid(True, alpha=0.25)
        if index == 0:
            response_axis.set_title("Receiver response")
            response_axis.legend(loc="best", fontsize=8)

        error_axis.plot(edg_time, edg_pressure[index] - reference_pressure[index], lw=1.1)
        error_axis.set_ylabel("EDG-COMSOL (Pa)")
        error_axis.grid(True, alpha=0.25)
        if index == 0:
            error_axis.set_title("Error")

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comsol", type=Path, required=True, help="COMSOL golden CSV")
    parser.add_argument("--edg", type=Path, required=True, help="EDG result .mat")
    parser.add_argument("--receiver-json", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None, help="Optional comparison PNG")
    parser.add_argument("--metrics-out", type=Path, default=None, help="Optional metrics JSON")
    parser.add_argument(
        "--skip-receiver-check",
        action="store_true",
        help="Allow comparing files without matching receiver metadata.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comsol_time, comsol_pressure, golden_receiver = load_comsol_golden(args.comsol)
    edg_time, edg_pressure, receiver, point_ids = load_edg_result(args.edg)

    if not args.skip_receiver_check:
        validate_edg_receiver(
            receiver,
            point_ids,
            load_expected_receiver(args.receiver_json),
            golden_receiver,
        )
    if edg_pressure.shape[0] != COMSOL_POINT_IDS.size:
        raise ValueError(
            f"EDG 'prec' has {edg_pressure.shape[0]} receiver rows; expected {COMSOL_POINT_IDS.size}."
        )

    reference_pressure = interpolate_comsol_to_edg(comsol_time, comsol_pressure, edg_time)
    metrics = compute_metrics(edg_time, edg_pressure, reference_pressure)

    print(json.dumps(metrics, indent=2))
    if args.metrics_out is not None:
        args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if args.plot is not None:
        plot_comparison(args.plot, edg_time, edg_pressure, reference_pressure)
        print(f"Wrote comparison plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
