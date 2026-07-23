"""Compare EDG receiver histories against COMSOL microphone golden data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import scipy.io


COMSOL_POINT_IDS = numpy.array([197, 391, 402], dtype=numpy.int32)
COMSOL_RECEIVER = numpy.array(
    [
        [2.0, 2.5, 2.5],
        [-0.05, -0.55, 0.55],
        [1.2, 1.2, 1.2],
    ],
    dtype=float,
)


def load_comsol_golden(path: Path) -> tuple[numpy.ndarray, numpy.ndarray]:
    with path.open() as handle:
        table_lines = [
            line
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not table_lines:
        raise ValueError(f"{path} does not contain a CSV table.")

    data = numpy.genfromtxt(table_lines, delimiter=",", names=True)
    data = numpy.atleast_1d(data)
    expected_columns = ("time", "p197", "p391", "p402")
    if data.dtype.names != expected_columns:
        raise ValueError(
            f"Unexpected COMSOL golden columns {data.dtype.names}; "
            f"expected {expected_columns}."
        )
    time = numpy.asarray(data["time"], dtype=float)
    pressure = numpy.vstack(
        [
            numpy.asarray(data["p197"], dtype=float),
            numpy.asarray(data["p391"], dtype=float),
            numpy.asarray(data["p402"], dtype=float),
        ]
    )
    if time.ndim != 1 or pressure.shape[1] != time.size:
        raise ValueError("Invalid COMSOL golden time/pressure shape.")
    if not numpy.all(numpy.diff(time) > 0.0):
        raise ValueError("COMSOL golden times must be strictly increasing.")
    return time, pressure


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
            f"EDG pressure/time length mismatch: prec={pressure.shape}, "
            f"prec_times={time.shape}."
        )
    return time, pressure, receiver, point_ids


def validate_edg_receiver(receiver: numpy.ndarray, point_ids: numpy.ndarray | None) -> None:
    if receiver.shape != COMSOL_RECEIVER.shape:
        raise ValueError(
            "EDG receiver shape does not match COMSOL microphone response points: "
            f"got {receiver.shape}, expected {COMSOL_RECEIVER.shape}. "
            "Regenerate results_on_the_run.mat with the current main.py."
        )
    if not numpy.allclose(receiver, COMSOL_RECEIVER, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "EDG receiver coordinates do not match COMSOL microphone response points."
        )
    if point_ids is not None and not numpy.array_equal(point_ids, COMSOL_POINT_IDS):
        raise ValueError(
            f"EDG receiver_point_ids={point_ids.tolist()} do not match "
            f"{COMSOL_POINT_IDS.tolist()}."
        )


def interpolate_comsol_to_edg(
    comsol_time: numpy.ndarray,
    comsol_pressure: numpy.ndarray,
    edg_time: numpy.ndarray,
) -> numpy.ndarray:
    tolerance = max(1.0e-12, 1.0e-9 * (comsol_time[-1] - comsol_time[0]))
    if edg_time[0] < comsol_time[0] - tolerance or edg_time[-1] > comsol_time[-1] + tolerance:
        raise ValueError(
            "EDG time range is outside COMSOL golden range: "
            f"EDG [{edg_time[0]}, {edg_time[-1]}], "
            f"COMSOL [{comsol_time[0]}, {comsol_time[-1]}]."
        )
    return numpy.vstack(
        [
            numpy.interp(edg_time, comsol_time, comsol_pressure[index])
            for index in range(comsol_pressure.shape[0])
        ]
    )


def compute_metrics(
    edg_time: numpy.ndarray,
    edg_pressure: numpy.ndarray,
    reference_pressure: numpy.ndarray,
) -> dict:
    error = edg_pressure - reference_pressure
    per_receiver = []
    for index, point_id in enumerate(COMSOL_POINT_IDS.tolist()):
        ref_norm = numpy.linalg.norm(reference_pressure[index])
        per_receiver.append(
            {
                "point_id": point_id,
                "rms_abs": float(numpy.sqrt(numpy.mean(error[index] ** 2))),
                "max_abs": float(numpy.max(numpy.abs(error[index]))),
                "relative_l2": float(
                    numpy.linalg.norm(error[index]) / max(ref_norm, 1.0e-30)
                ),
            }
        )

    ref_global_norm = numpy.linalg.norm(reference_pressure)
    return {
        "num_samples": int(edg_time.size),
        "time_start": float(edg_time[0]),
        "time_end": float(edg_time[-1]),
        "receiver_point_ids": COMSOL_POINT_IDS.tolist(),
        "per_receiver": per_receiver,
        "global": {
            "rms_abs": float(numpy.sqrt(numpy.mean(error**2))),
            "max_abs": float(numpy.max(numpy.abs(error))),
            "relative_l2": float(numpy.linalg.norm(error) / max(ref_global_norm, 1.0e-30)),
        },
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
            response_axis.set_title("Microphone response")
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
    comsol_time, comsol_pressure = load_comsol_golden(args.comsol)
    edg_time, edg_pressure, receiver, point_ids = load_edg_result(args.edg)

    if not args.skip_receiver_check:
        validate_edg_receiver(receiver, point_ids)
    if edg_pressure.shape[0] != COMSOL_POINT_IDS.size:
        raise ValueError(
            f"EDG 'prec' has {edg_pressure.shape[0]} receiver rows; "
            f"expected {COMSOL_POINT_IDS.size}."
        )

    reference_pressure = interpolate_comsol_to_edg(
        comsol_time, comsol_pressure, edg_time
    )
    metrics = compute_metrics(edg_time, edg_pressure, reference_pressure)

    print(json.dumps(metrics, indent=2))
    if args.metrics_out is not None:
        args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n")
    if args.plot is not None:
        plot_comparison(args.plot, edg_time, edg_pressure, reference_pressure)
        print(f"Wrote comparison plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
