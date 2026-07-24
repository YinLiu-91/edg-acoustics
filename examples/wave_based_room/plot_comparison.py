#!/usr/bin/env python3
"""Plot EDG receiver pressure vs COMSOL golden for the wave_based_room case.

Matching strategy
-----------------
The EDG ``prec`` rows are stored in reverse order relative to the ``rec``
columns.  This script uses a data-driven pairwise search: for each EDG
receiver row it picks the golden pressure column that yields the smallest
RMS error.  The result is equivalent to ``prec[::-1, :]`` paired with the
golden CSV columns in their labelled order (p122, p121, p53, p35).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

matplotlib.use("Agg")

CASE_DIR = Path(__file__).resolve().parent
MAT_PATH = CASE_DIR / "results_on_the_run.mat"
GOLDEN_PATH = CASE_DIR / "wave_based_room_comsol_golden.csv"
OUTPUT_PATH = CASE_DIR / "pressure_time_golden_comparison.png"
ERROR_OUTPUT_PATH = CASE_DIR / "pressure_time_golden_error.png"

GOLDEN_POINT_IDS = (122, 121, 53, 35)

EDG_COLOR = "#2b6a9b"
GOLDEN_COLOR = "#c44e52"
ERROR_COLOR = "#555555"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_edg(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = scipy.io.loadmat(str(path))
    prec = np.asarray(mat["prec"], dtype=float)
    edg_t = np.asarray(mat["prec_times"], dtype=float).ravel()
    rec = np.asarray(mat["rec"], dtype=float)
    return prec, edg_t, rec


def load_golden(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    tbl = np.loadtxt(lines, delimiter=",", skiprows=1)
    return tbl[:, 0], tbl[:, 1:5]  # time, p122, p121, p53, p35


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def find_optimal_mapping(
    edg_t: np.ndarray, prec: np.ndarray, golden_t: np.ndarray, golden_p: np.ndarray,
) -> list[int]:
    """Return the golden column index (0..3) that best matches each EDG row."""
    mapping: list[int] = []
    for i in range(prec.shape[0]):
        best_rms = float("inf")
        best_j = -1
        for j in range(golden_p.shape[1]):
            g_interp = np.interp(edg_t, golden_t, golden_p[:, j])
            rms = np.sqrt(np.mean((prec[i] - g_interp) ** 2))
            if rms < best_rms:
                best_rms, best_j = rms, j
        mapping.append(best_j)
    return mapping


# ---------------------------------------------------------------------------
# Plot & stats
# ---------------------------------------------------------------------------

def plot_comparison(
    edg_t: np.ndarray,
    prec: np.ndarray,
    rec: np.ndarray,
    golden_t: np.ndarray,
    golden_p: np.ndarray,
    mapping: list[int],
    output_path: Path,
) -> None:
    golden_interp = np.array(
        [np.interp(edg_t, golden_t, golden_p[:, j]) for j in mapping]
    )
    t_ms = edg_t * 1000

    fig, axes = plt.subplots(len(mapping), 1, figsize=(14, 10), sharex=True)

    for i in range(len(mapping)):
        pid = GOLDEN_POINT_IDS[mapping[i]]
        axes[i].plot(
            t_ms, golden_interp[i],
            color=GOLDEN_COLOR, lw=1.5, ls="-.", label="COMSOL",
        )
        axes[i].plot(
            t_ms, prec[i],
            color=EDG_COLOR, lw=1.0, label="EDG",
        )
        axes[i].set_ylabel("Pressure [Pa]", fontsize=11)
        axes[i].set_title(
            f"Point {pid}  "
            f"(x={rec[0, i]:.2f}, y={rec[1, i]:.2f}, z={rec[2, i]:.2f})",
            fontsize=11,
        )
        axes[i].grid(True, alpha=0.25)
        axes[i].legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time [ms]", fontsize=12)
    fig.suptitle("Wave-Based Room — EDG vs COMSOL Golden", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_error(
    edg_t: np.ndarray,
    prec: np.ndarray,
    rec: np.ndarray,
    golden_t: np.ndarray,
    golden_p: np.ndarray,
    mapping: list[int],
    output_path: Path,
) -> None:
    golden_interp = np.array(
        [np.interp(edg_t, golden_t, golden_p[:, j]) for j in mapping]
    )
    error = prec - golden_interp
    t_ms = edg_t * 1000

    fig, axes = plt.subplots(len(mapping), 1, figsize=(14, 10), sharex=True)

    for i in range(len(mapping)):
        pid = GOLDEN_POINT_IDS[mapping[i]]
        axes[i].plot(t_ms, error[i], color=ERROR_COLOR, lw=0.8)
        axes[i].set_ylabel("EDG − COMSOL [Pa]", fontsize=10)
        axes[i].set_title(
            f"Point {pid}  "
            f"(x={rec[0, i]:.2f}, y={rec[1, i]:.2f}, z={rec[2, i]:.2f})",
            fontsize=11,
        )
        axes[i].grid(True, alpha=0.25)
        axes[i].axhline(y=0, color="black", lw=0.5, alpha=0.3)

    axes[-1].set_xlabel("Time [ms]", fontsize=12)
    fig.suptitle("Wave-Based Room — EDG − COMSOL Error", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_mapping_table(prec, edg_t, rec, golden_t, golden_p, mapping) -> None:
    print("\n--- Pairwise mapping (data-driven optimal) ---")
    print(f"{'EDG row':<10} {'rec coord':<18} {'→ golden':<10} {'corr':>8} {'RMS':>10}")
    for i, j in enumerate(mapping):
        g_interp = np.interp(edg_t, golden_t, golden_p[:, j])
        corr = np.corrcoef(prec[i], g_interp)[0, 1]
        rms = np.sqrt(np.mean((prec[i] - g_interp) ** 2))
        pid = GOLDEN_POINT_IDS[j]
        print(
            f"prec[{i}]    ({rec[0,i]:.2f},{rec[1,i]:.2f},{rec[2,i]:.2f})  "
            f"→ p{pid:<6} {corr:+.4f}  {rms:.4f}"
        )


def print_stats(prec, edg_t, golden_t, golden_p, mapping) -> None:
    golden_interp = np.array(
        [np.interp(edg_t, golden_t, golden_p[:, j]) for j in mapping]
    )
    error = prec - golden_interp
    print("\n--- Error statistics ---")
    for i, j in enumerate(mapping):
        pid = GOLDEN_POINT_IDS[j]
        rms = np.sqrt(np.mean(error[i] ** 2))
        max_err = np.max(np.abs(error[i]))
        corr = np.corrcoef(prec[i], golden_interp[i])[0, 1]
        print(
            f"p{pid}: RMS={rms:.4f} Pa  max_err={max_err:.4f} Pa  "
            f"corr={corr:+.4f}  |EDG|max={np.max(np.abs(prec[i])):.4f}  "
            f"|golden|max={np.max(np.abs(golden_interp[i])):.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    prec, edg_t, rec = load_edg(MAT_PATH)
    golden_t, golden_p = load_golden(GOLDEN_PATH)

    mapping = find_optimal_mapping(edg_t, prec, golden_t, golden_p)
    print_mapping_table(prec, edg_t, rec, golden_t, golden_p, mapping)
    plot_comparison(edg_t, prec, rec, golden_t, golden_p, mapping, OUTPUT_PATH)
    plot_error(edg_t, prec, rec, golden_t, golden_p, mapping, ERROR_OUTPUT_PATH)
    print_stats(prec, edg_t, golden_t, golden_p, mapping)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
