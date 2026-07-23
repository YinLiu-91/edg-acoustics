#!/usr/bin/env python3
"""Compare EDG receiver histories against COMSOL wave-based-room golden data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import scipy.io


COMSOL_POINT_IDS = numpy.array([122, 121, 53, 35], dtype=numpy.int32)


def load_comsol_golden(path: Path) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None]:
    receiver_rows: dict[int, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        lines = list(handle)
    for line in lines:
        fields = [field.strip() for field in line.lstrip("# ").split(",")]
        if fields[0] == "receiver_coordinate" and len(fields) == 5:
            receiver_rows[int(fields[1])] = [float(value) for value in fields[2:]]
    table_lines = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    data = numpy.genfromtxt(table_lines, delimiter=",", names=True)
    data = numpy.atleast_1d(data)
    expected_columns = ("time", "p122", "p121", "p53", "p35", "pn122", "pn121", "pn53", "pn35")
    if data.dtype.names != expected_columns:
        raise ValueError(f"Unexpected COMSOL golden columns {data.dtype.names}; expected {expected_columns}.")
    time = numpy.asarray(data["time"], dtype=float)
    pressure = numpy.vstack(
        [
            numpy.asarray(data["p122"], dtype=float),
            numpy.asarray(data["p121"], dtype=float),
            numpy.asarray(data["p53"], dtype=float),
            numpy.asarray(data["p35"], dtype=float),
        ]
    )
    if not numpy.all(numpy.diff(time) > 0.0):
        raise ValueError("COMSOL golden times must be strictly increasing.")
    receiver = None
    if receiver_rows:
        if set(receiver_rows) != set(COMSOL_POINT_IDS.tolist()):
            raise ValueError("COMSOL golden receiver metadata has unexpected point ids.")
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
    if pressure.shape[0] != COMSOL_POINT_IDS.size and pressure.shape[1] == COMSOL_POINT_IDS.size:
        pressure = pressure.T
    if "prec_times" in data and data["prec_times"].size == pressure.shape[1]:
        time = numpy.asarray(data["prec_times"], dtype=float).reshape(-1)
    elif "dt" in data:
        dt = float(numpy.asarray(data["dt"]).reshape(-1)[0])
        time = (numpy.arange(pressure.shape[1], dtype=float) + 1.0) * dt
    else:
        raise ValueError(f"{path} does not contain 'prec_times' or 'dt'.")
    receiver = numpy.squeeze(numpy.asarray(data.get("rec", numpy.empty((0, 0))), dtype=float))
    if receiver.shape == (4, 3):
        receiver = receiver.T
    point_ids = None
    if "receiver_point_ids" in data:
        point_ids = numpy.asarray(data["receiver_point_ids"], dtype=numpy.int32).reshape(-1)
    return time, pressure, receiver, point_ids


def load_expected_receiver(path: Path | None) -> numpy.ndarray | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    point_ids = numpy.asarray(data["point_ids"], dtype=numpy.int32)
    if not numpy.array_equal(point_ids, COMSOL_POINT_IDS):
        raise ValueError("receiver JSON point ids do not match COMSOL golden.")
    coords = numpy.asarray(data["coords"], dtype=float)
    if coords.shape != (3, 4):
        raise ValueError(f"receiver JSON coords must have shape (3, 4), got {coords.shape}")
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
        raise ValueError(f"EDG receiver_point_ids={point_ids.tolist()} do not match {COMSOL_POINT_IDS.tolist()}.")
    if golden_receiver is None:
        raise ValueError("COMSOL golden is missing receiver_coordinate metadata.")
    if expected_receiver is not None and not numpy.allclose(expected_receiver, golden_receiver, rtol=0.0, atol=1.0e-12):
        raise ValueError("receiver JSON coordinates do not match COMSOL golden metadata.")
    if receiver.shape != golden_receiver.shape:
        raise ValueError(f"EDG receiver shape {receiver.shape} does not match expected {golden_receiver.shape}.")
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
            f"EDG time range [{edg_time[0]}, {edg_time[-1]}] is outside COMSOL "
            f"golden range [{comsol_time[0]}, {comsol_time[-1]}]."
        )
    return numpy.vstack(
        [numpy.interp(edg_time, comsol_time, comsol_pressure[index]) for index in range(comsol_pressure.shape[0])]
    )


def _metric_block(error: numpy.ndarray, reference: numpy.ndarray) -> dict:
    ref_norm = numpy.linalg.norm(reference)
    return {
        "rms_abs": float(numpy.sqrt(numpy.mean(error**2))),
        "max_abs": float(numpy.max(numpy.abs(error))),
        "relative_l2": float(numpy.linalg.norm(error) / max(ref_norm, 1.0e-30)),
    }


def compute_metrics(edg_time: numpy.ndarray, edg_pressure: numpy.ndarray, reference_pressure: numpy.ndarray) -> dict:
    error = edg_pressure - reference_pressure
    per_receiver = []
    for index, point_id in enumerate(COMSOL_POINT_IDS.tolist()):
        block = _metric_block(error[index], reference_pressure[index])
        block["point_id"] = point_id
        per_receiver.append(block)
    return {
        "num_samples": int(edg_time.size),
        "time_start": float(edg_time[0]),
        "time_end": float(edg_time[-1]),
        "receiver_point_ids": COMSOL_POINT_IDS.tolist(),
        "per_receiver": per_receiver,
        "global": _metric_block(error, reference_pressure),
    }


def plot_comparison(output_path: Path, edg_time: numpy.ndarray, edg_pressure: numpy.ndarray, reference_pressure: numpy.ndarray) -> None:
    fig, axes = plt.subplots(
        nrows=COMSOL_POINT_IDS.size,
        ncols=2,
        figsize=(12.0, 9.0),
        sharex=True,
        constrained_layout=True,
    )
    for index, point_id in enumerate(COMSOL_POINT_IDS.tolist()):
        axes[index, 0].plot(edg_time, reference_pressure[index], label="COMSOL", lw=1.4)
        axes[index, 0].plot(edg_time, edg_pressure[index], label="EDG", lw=1.1, ls="--")
        axes[index, 0].set_ylabel(f"p{point_id} (Pa)")
        axes[index, 0].grid(True, alpha=0.25)
        if index == 0:
            axes[index, 0].legend(loc="best", fontsize=8)
        axes[index, 1].plot(edg_time, edg_pressure[index] - reference_pressure[index], lw=1.1)
        axes[index, 1].set_ylabel("EDG-COMSOL (Pa)")
        axes[index, 1].grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comsol", type=Path, required=True)
    parser.add_argument("--edg", type=Path, required=True)
    parser.add_argument("--receiver-json", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--skip-receiver-check", action="store_true")
    args = parser.parse_args()

    comsol_time, comsol_pressure, golden_receiver = load_comsol_golden(args.comsol)
    edg_time, edg_pressure, receiver, point_ids = load_edg_result(args.edg)
    if not args.skip_receiver_check:
        validate_edg_receiver(receiver, point_ids, load_expected_receiver(args.receiver_json), golden_receiver)
    if edg_pressure.shape[0] != COMSOL_POINT_IDS.size:
        raise ValueError(f"EDG 'prec' has {edg_pressure.shape[0]} receiver rows; expected {COMSOL_POINT_IDS.size}.")
    reference_pressure = interpolate_comsol_to_edg(comsol_time, comsol_pressure, edg_time)
    metrics = compute_metrics(edg_time, edg_pressure, reference_pressure)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.metrics_out is not None:
        args.metrics_out.write_text(text + "\n", encoding="utf-8")
    if args.plot is not None:
        plot_comparison(args.plot, edg_time, edg_pressure, reference_pressure)
        print(f"Wrote comparison plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
