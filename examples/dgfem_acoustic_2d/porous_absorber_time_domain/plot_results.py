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


def interpolate_series(source_time, source_values, target_time):
    source_time = numpy.asarray(source_time).reshape(-1)
    source_values = numpy.asarray(source_values).reshape(-1)
    target_time = numpy.asarray(target_time).reshape(-1)
    if target_time.size == 0:
        return numpy.empty((0,), dtype=float)
    return numpy.interp(target_time, source_time, source_values)


def compute_error_series(
    sim_time,
    sim_pressure,
    golden_time,
    golden_pressure,
    *,
    relative_floor_ratio: float,
):
    sim_time = numpy.asarray(sim_time).reshape(-1)
    sim_pressure = numpy.asarray(sim_pressure).reshape(-1)
    golden_time = numpy.asarray(golden_time).reshape(-1)
    golden_pressure = numpy.asarray(golden_pressure).reshape(-1)
    overlap_mask = (golden_time >= sim_time[0]) & (golden_time <= sim_time[-1])
    if not numpy.any(overlap_mask):
        raise ValueError("Simulation and COMSOL golden data do not overlap in time.")
    aligned_time = golden_time[overlap_mask]
    aligned_golden_pressure = golden_pressure[overlap_mask]
    aligned_sim_pressure = interpolate_series(sim_time, sim_pressure, aligned_time)
    absolute_error = numpy.abs(aligned_sim_pressure - aligned_golden_pressure)
    golden_peak = float(numpy.max(numpy.abs(aligned_golden_pressure)))
    denominator_floor = relative_floor_ratio * golden_peak
    denominator = numpy.maximum(numpy.abs(aligned_golden_pressure), denominator_floor)
    relative_error = numpy.divide(
        absolute_error,
        denominator,
        out=numpy.zeros_like(absolute_error),
        where=denominator > 0.0,
    )
    return {
        "time": aligned_time,
        "sim_pressure": aligned_sim_pressure,
        "golden_pressure": aligned_golden_pressure,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "relative_error_percent": 100.0 * relative_error,
        "relative_error_floor": denominator_floor,
    }


def compute_error_metrics(absolute_error, golden_pressure):
    absolute_error = numpy.asarray(absolute_error).reshape(-1)
    golden_pressure = numpy.asarray(golden_pressure).reshape(-1)
    rmse = float(numpy.sqrt(numpy.mean(absolute_error**2)))
    max_abs = float(numpy.max(absolute_error))
    golden_norm = float(numpy.linalg.norm(golden_pressure))
    rel_l2 = float(numpy.linalg.norm(absolute_error) / golden_norm) if golden_norm > 0.0 else 0.0
    return {
        "rmse": rmse,
        "max_abs": max_abs,
        "rel_l2": rel_l2,
    }


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
    parser.add_argument(
        "--relative-error-floor-ratio",
        type=float,
        default=1.0e-3,
        help=(
            "Pointwise relative error floor ratio. Relative error is computed as "
            "|e| / max(|p_ref|, ratio * max|p_ref|)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    golden_paths = discover_default_golden_files() if args.golden is None else args.golden
    has_golden = len(golden_paths) > 0
    nrows = 4 if has_golden else 2
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(12, 14 if has_golden else 9),
        constrained_layout=True,
    )
    axes = numpy.atleast_1d(axes)
    history_ax = axes[0]
    zoom_ax = axes[1]
    absolute_error_ax = axes[2] if has_golden else None
    relative_error_ax = axes[3] if has_golden else None
    result_colors: dict[float | str, str] = {}
    result_series: dict[float | str, dict[str, numpy.ndarray | str | float | None]] = {}

    for path in args.results:
        time, pressure, thickness = load_series(path)
        label = path.parent.name if thickness is None else f"{thickness * 100:.0f} cm"
        line = history_ax.plot(time, pressure, label=label)[0]
        color = line.get_color()
        key = label if thickness is None else thickness
        result_colors[key] = color
        result_series[key] = {
            "time": time,
            "pressure": pressure,
            "label": label,
            "thickness": thickness,
        }
        zoom_mask = (time >= args.zoom_start_ms * 1.0e-3) & (time <= args.zoom_end_ms * 1.0e-3)
        zoom_ax.plot(time[zoom_mask], pressure[zoom_mask], label=label, color=color)

    for path in golden_paths:
        time, pressure, thickness = load_golden_series(path)
        label = path.stem if thickness is None else f"{thickness * 100:.0f} cm COMSOL"
        key = label if thickness is None else thickness
        color = result_colors.get(key, None)
        history_ax.plot(
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
        zoom_ax.plot(
            time[zoom_mask],
            pressure[zoom_mask],
            linestyle="None",
            marker="o",
            markersize=3.0,
            markerfacecolor="none",
            color=color,
            label=label,
        )
        if not has_golden or key not in result_series:
            continue
        error_series = compute_error_series(
            result_series[key]["time"],
            result_series[key]["pressure"],
            time,
            pressure,
            relative_floor_ratio=args.relative_error_floor_ratio,
        )
        metrics = compute_error_metrics(
            error_series["absolute_error"], error_series["golden_pressure"]
        )
        absolute_error_ax.plot(
            error_series["time"],
            error_series["absolute_error"],
            color=color,
            label=f"{result_series[key]['label']} |e|",
        )
        relative_error_ax.plot(
            error_series["time"],
            error_series["relative_error_percent"],
            color=color,
            label=f"{result_series[key]['label']} rel (L2={metrics['rel_l2']:.2%})",
        )

    history_ax.set_title("Receiver Pressure History")
    history_ax.set_xlabel("Time [s]")
    history_ax.set_ylabel("Pressure")
    history_ax.grid(True, alpha=0.3)
    history_ax.legend()

    zoom_ax.set_title(
        f"Zoomed View ({args.zoom_start_ms:.1f} ms to {args.zoom_end_ms:.1f} ms)"
    )
    zoom_ax.set_xlabel("Time [s]")
    zoom_ax.set_ylabel("Pressure")
    zoom_ax.grid(True, alpha=0.3)
    zoom_ax.legend()

    if has_golden:
        absolute_error_ax.set_title("Absolute Error vs COMSOL")
        absolute_error_ax.set_xlabel("Time [s]")
        absolute_error_ax.set_ylabel("|p - p_ref|")
        absolute_error_ax.set_ylim(bottom=0.0)
        absolute_error_ax.grid(True, alpha=0.3)
        absolute_error_ax.legend()

        relative_error_ax.set_title(
            f"Relative Error vs COMSOL (floor={100.0 * args.relative_error_floor_ratio:.3f}% peak)"
        )
        relative_error_ax.set_xlabel("Time [s]")
        relative_error_ax.set_ylabel("Relative Error [%]")
        relative_error_ax.set_ylim(bottom=0.0)
        relative_error_ax.grid(True, alpha=0.3)
        relative_error_ax.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
