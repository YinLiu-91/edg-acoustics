"""Plot receiver histories for the 2D ER porous absorber runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import scipy.io


CASE_DIR = Path(__file__).resolve().parent


def load_series(path: Path):
    data = scipy.io.loadmat(path, squeeze_me=False)
    if "prec" not in data or "time" not in data:
        raise ValueError(f"{path} does not contain receiver history data.")
    time = numpy.asarray(data["time"]).reshape(-1)
    pressure = numpy.asarray(data["prec"])
    if pressure.ndim == 2 and pressure.shape[0] == 1:
        pressure = pressure.reshape(-1)
    elif pressure.ndim == 2:
        pressure = pressure[0]
    else:
        pressure = pressure.reshape(-1)
    thickness = None
    if "thickness_m" in data:
        thickness = float(numpy.asarray(data["thickness_m"]).reshape(-1)[0])
    return time, pressure, thickness


def infer_thickness_from_path(path: Path):
    stem = path.stem.lower()
    if re.search(r"(^|[^0-9])15cm([^0-9]|$)", stem):
        return 0.15
    if re.search(r"(^|[^0-9])5cm([^0-9]|$)", stem):
        return 0.05
    return None


def load_golden_series(path: Path):
    try:
        data = numpy.loadtxt(path, comments="%", dtype=float)
    except Exception as exc:
        raise ValueError(f"Failed to parse COMSOL golden data from {path}") from exc
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns: time and pressure.")
    thickness = infer_thickness_from_path(path)
    return data[:, 0], data[:, 1], thickness


def discover_default_golden_files():
    candidates = []
    for name in ("5cm_er_comsol_golden.txt", "15cm_er_comsol_golden.txt"):
        path = CASE_DIR / name
        if path.exists():
            candidates.append(path)
    return candidates


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ER porous absorber receiver histories.")
    parser.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="One or more results_on_the_run.mat files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("receiver_history.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--zoom-start-ms",
        type=float,
        default=5.0,
        help="Zoom window start time in milliseconds.",
    )
    parser.add_argument(
        "--zoom-end-ms",
        type=float,
        default=10.0,
        help="Zoom window end time in milliseconds.",
    )
    parser.add_argument(
        "--golden",
        nargs="*",
        type=Path,
        default=None,
        help="Optional COMSOL golden data files to overlay as point markers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    result_colors: dict[float | str, str] = {}

    for path in args.results:
        time, pressure, thickness = load_series(path)
        label = path.parent.name if thickness is None else f"{thickness * 100:.0f} cm"
        line = axes[0].plot(time, pressure, label=label)[0]
        color = line.get_color()
        key = label if thickness is None else thickness
        result_colors[key] = color
        zoom_mask = (time >= args.zoom_start_ms * 1.0e-3) & (time <= args.zoom_end_ms * 1.0e-3)
        axes[1].plot(time[zoom_mask], pressure[zoom_mask], label=label, color=color)

    golden_paths = discover_default_golden_files() if args.golden is None else args.golden
    for path in golden_paths:
        time, pressure, thickness = load_golden_series(path)
        label = path.stem if thickness is None else f"{thickness * 100:.0f} cm COMSOL"
        key = label if thickness is None else thickness
        color = result_colors.get(key, None)
        axes[0].plot(
            time,
            pressure,
            linestyle="None",
            marker="o",
            markersize=3.0,
            markerfacecolor="none",
            color=color,
            label=label,
        )
        zoom_mask = (time >= args.zoom_start_ms * 1.0e-3) & (time <= args.zoom_end_ms * 1.0e-3)
        axes[1].plot(
            time[zoom_mask],
            pressure[zoom_mask],
            linestyle="None",
            marker="o",
            markersize=3.0,
            markerfacecolor="none",
            color=color,
            label=label,
        )

    axes[0].set_title("Receiver Pressure History")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Pressure")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title(
        f"Zoomed View ({args.zoom_start_ms:.1f} ms to {args.zoom_end_ms:.1f} ms)"
    )
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Pressure")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
