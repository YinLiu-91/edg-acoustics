#!/usr/bin/env python3
"""Fit COMSOL admittance tables to EDG multi-pole rational parameters.

Converts frequency-dependent acoustic admittance Y(ω) → normal-incidence
reflection coefficient R(ω), then fits the EDG complex-pole (CP) model
via nonlinear least-squares optimization.

Usage:
    python fit_admittance.py [--Ncp N] [--plot]

Output .mat files are written to the car_cabin directory with EDG-compatible
parameters (AS, lambdaS, BS, CS, alphaS, betaS, RI).
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from scipy.io import savemat
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO0 = 1.213
C0 = 343.0
Z0 = RHO0 * C0

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAR_CABIN_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "car_cabin_acoustics_transient_63_cleared",
)

MATERIALS = {
    "seat": os.path.join(CAR_CABIN_DIR, "seat_admittance_63.txt"),
    "carpet": os.path.join(CAR_CABIN_DIR, "carpet_admittance_63.txt"),
    "roof": os.path.join(CAR_CABIN_DIR, "roof_admittance_63.txt"),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def parse_admittance(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a COMSOL admittance table → (freq_Hz, Y_complex)."""
    rows = []
    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except (ValueError, IndexError):
                continue
    data = np.array(rows)
    return data[:, 0], data[:, 1] + 1j * data[:, 2]


def admittance_to_R(freq_Hz: np.ndarray,
                     Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Y → R(ω) = (Z_norm - 1)/(Z_norm + 1) with Z_norm = 1/(Y·Z0)."""
    Z_norm = 1.0 / (Y * Z0)
    R = (Z_norm - 1.0) / (Z_norm + 1.0)
    return 2.0 * np.pi * freq_Hz, R


# ---------------------------------------------------------------------------
# EDG CP model
# ---------------------------------------------------------------------------
# R(ω) = RI + Σ_j ½[(B_j + iC_j)/(α_j + iβ_j + iω) + (B_j - iC_j)/(α_j - iβ_j + iω)]
#
# Parameter vector x (length 1 + 4·Ncp):
#   [RI, B_0, C_0, α_0, β_0,  B_1, C_1, α_1, β_1,  ...]
#
# Constraints:   0 ≤ RI ≤ 1,  α_j > 0,  β_j > 0

def _unpack(x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray]:
    """Unpack parameter vector into RI and CP arrays."""
    RI = x[0]
    N = (len(x) - 1) // 4
    B = x[1 + 0*N:1 + 1*N]
    C = x[1 + 1*N:1 + 2*N]
    alpha = x[1 + 2*N:1 + 3*N]
    beta = x[1 + 3*N:1 + 4*N]
    return RI, B, C, alpha, beta


def _pack(RI: float, B: np.ndarray, C: np.ndarray,
          alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Pack parameters into a 1D vector."""
    return np.concatenate([[RI], B, C, alpha, beta])


def _residuals(x: np.ndarray, omega: np.ndarray,
               R_target: np.ndarray) -> np.ndarray:
    """Complex residuals: R_model(ω) - R_target(ω), with passivity penalty."""
    RI, B, C, alpha, beta = _unpack(x)
    s = 1j * omega
    R_model = np.full_like(s, RI, dtype=complex)
    for b, c, a_val, be in zip(B, C, alpha, beta):
        R_model += 0.5 * ((b + 1j * c) / (a_val + 1j * be + s)
                          + (b - 1j * c) / (a_val - 1j * be + s))

    # Core residual: R_model - R_target (passivity enforced post-hoc)
    res = R_model - R_target
    return np.concatenate([np.real(res), np.imag(res)])


def _initial_guess(omega: np.ndarray, R: np.ndarray, Ncp: int) -> np.ndarray:
    """Generate an initial guess for the EDG CP model parameters.

    Places CP pairs at log-spaced frequencies in the data band.
    For resonant materials, places extra poles near absorption minima.
    """
    w_min = max(omega.min(), 2 * np.pi * 1.0)
    w_max = omega.max()

    # Find absorption peaks (where |R| has local minima)
    R_mag = np.abs(R)
    betas_init = []

    # Find local minima of |R| (absorption peaks)
    for i in range(1, len(R_mag) - 1):
        if R_mag[i] < R_mag[i - 1] and R_mag[i] < R_mag[i + 1]:
            if R_mag[i] < 0.95:  # significant absorption
                betas_init.append(omega[i])

    # Fill remaining with log-spaced poles
    n_remaining = Ncp - len(betas_init)
    if n_remaining > 0:
        logspace_poles = np.logspace(np.log10(w_min), np.log10(w_max),
                                     n_remaining)
        betas_init.extend(logspace_poles)

    betas_init = np.array(sorted(betas_init[:Ncp]))
    alphas_init = 0.15 * betas_init  # moderate damping
    B_init = np.zeros(Ncp)
    C_init = np.zeros(Ncp)

    # RI: use high-frequency |R| as initial estimate
    RI_init = float(np.clip(np.abs(R[-1]), 0.0, 1.0))

    # Scale to ensure initial model is roughly in range
    return _pack(RI_init, B_init, C_init, alphas_init, betas_init)


# ---------------------------------------------------------------------------
# EDG evaluation (matching boundary_condition.py exactly)
# ---------------------------------------------------------------------------
def compute_Re_edg(omega: np.ndarray, params: dict) -> np.ndarray:
    """Evaluate R(ω) using EDG parameter dict (matching boundary_condition.py)."""
    Re = np.ones(len(omega), dtype=complex)
    for polekey in params:
        if polekey == "RI":
            Re = Re * params["RI"]
        elif polekey == "RP":
            Re = Re * params["RI"]
            for a, z in zip(params["RP"][0, :], params["RP"][1, :]):
                Re = Re + a / (1j * omega + z)
        elif polekey == "CP":
            Re = Re * params["RI"]
            CP = params["CP"]
            for j in range(CP.shape[1]):
                b, c, a_val, be = CP[0, j], CP[1, j], CP[2, j], CP[3, j]
                Re = Re + 0.5 * (
                    (b + 1j * c) / (a_val + 1j * be + 1j * omega)
                    + (b - 1j * c) / (a_val - 1j * be + 1j * omega)
                )
    return Re


def x_to_edg_params(x: np.ndarray) -> dict:
    """Convert solution vector to EDG parameter dict (CP format).

    Returns:
      {"RI": float, "CP": 4×Ncp array}
    """
    RI, B, C, alpha, beta = _unpack(x)
    RI = float(np.clip(RI, 0.0, 1.0))
    # Ensure positive damping
    alpha = np.maximum(alpha, 1e-6)
    beta = np.maximum(beta, 1e-6)
    CP = np.array([B, C, alpha, beta])
    return {"RI": RI, "CP": CP}


def enforce_passivity(edg_params: dict,
                       max_freq_rad_s: float = 2 * np.pi * 2000.0) -> dict:
    """Post-process EDG params to strictly enforce |R(ω)| ≤ 1.

    Checks |R| over a dense grid from 1 to max_freq_rad_s.
    If passivity is violated, scales down CP residues (B, C) by a factor
    to bring |R| below 1.  RI is left unchanged.

    The scaling preserves the frequency-dependent shape of the reflection
    coefficient while reducing its overall magnitude.
    """
    # Dense frequency check grid: 1 to max_freq rad/s
    omega_check = np.linspace(1.0, max_freq_rad_s, 5000)

    R_check = compute_Re_edg(omega_check, edg_params)
    max_R = np.max(np.abs(R_check))

    if max_R <= 1.0:
        return edg_params  # already passive

    # Find scaling factor via binary search
    RI = edg_params["RI"]
    CP = edg_params.get("CP")

    if CP is None:
        # RI-only model: just clamp
        return {"RI": min(RI, 1.0)}

    def eval_scaled(scale, omega):
        """Evaluate R with CP residues scaled by `scale`."""
        Re = np.full(len(omega), RI, dtype=complex)
        B = CP[0, :] * scale
        C = CP[1, :] * scale
        alpha = CP[2, :]
        beta = CP[3, :]
        for b, c, a_val, be in zip(B, C, alpha, beta):
            Re += 0.5 * (
                (b + 1j * c) / (a_val + 1j * be + 1j * omega)
                + (b - 1j * c) / (a_val - 1j * be + 1j * omega)
            )
        return Re

    # Binary search for the largest scale ≤ 1 that preserves passivity
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        R_scaled = eval_scaled(mid, omega_check)
        if np.max(np.abs(R_scaled)) <= 1.0:
            lo = mid  # this scale works, try larger
        else:
            hi = mid  # too large, reduce

    scale = lo
    CP_scaled = np.array([CP[0, :] * scale, CP[1, :] * scale,
                          CP[2, :], CP[3, :]])

    return {"RI": RI, "CP": CP_scaled}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def print_fit_quality(name: str, R_true: np.ndarray, R_fit: np.ndarray):
    err = np.abs(R_fit - R_true)
    rms = np.sqrt(np.mean(err**2))
    rms_mag = np.sqrt(np.mean((np.abs(R_fit) - np.abs(R_true))**2))
    print(f"  {name}: RMS = {rms:.4e}, RMS(|R|) = {rms_mag:.4e}, "
          f"max |err| = {np.max(err):.4e}")


def plot_fit(name: str, freq: np.ndarray, omega: np.ndarray,
             R_true: np.ndarray, R_fit_edg: np.ndarray, outdir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{name} — Reflection Coefficient Fit", fontsize=14)

    ax = axes[0, 0]
    ax.semilogx(freq, np.abs(R_true), "b-", label="True", linewidth=1.5)
    ax.semilogx(freq, np.abs(R_fit_edg), "r--", label="EDG fit", linewidth=1.5)
    ax.set_xlabel("Hz"); ax.set_ylabel("|R|"); ax.set_title("Magnitude")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogx(freq, np.abs(R_fit_edg) - np.abs(R_true), "k-", linewidth=1.0)
    ax.set_xlabel("Hz"); ax.set_ylabel("Δ|R|"); ax.set_title("Magnitude Error")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogx(freq, np.angle(R_true, deg=True), "b-", label="True", linewidth=1.5)
    ax.semilogx(freq, np.angle(R_fit_edg, deg=True), "r--", label="EDG fit", linewidth=1.5)
    ax.set_xlabel("Hz"); ax.set_ylabel("Phase [deg]"); ax.set_title("Phase")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(np.real(R_true), np.imag(R_true), "b-", label="True", linewidth=1.0)
    ax.plot(np.real(R_fit_edg), np.imag(R_fit_edg), "r--", label="EDG fit", linewidth=1.0)
    ax.set_xlabel("Re(R)"); ax.set_ylabel("Im(R)"); ax.set_title("Nyquist")
    ax.legend(); ax.axis("equal"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(outdir, f"{name}_fit_diagnostics.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Plot saved to {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fit COMSOL admittance to EDG CP parameters via least-squares"
    )
    parser.add_argument("--Ncp", type=int, default=None,
                        help="Number of complex pole pairs (default: auto)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots")
    parser.add_argument("--materials", nargs="*",
                        default=["seat", "carpet", "roof"],
                        help="Materials to fit")
    parser.add_argument("--drop-dc", action="store_true", default=True,
                        help="Drop DC (0 Hz) data points")
    args = parser.parse_args()

    print(f"EDG CP fitting (auto Ncp per material)")
    print(f"ρ₀={RHO0}, c₀={C0}, Z₀={Z0:.1f}")
    print()

    for name in args.materials:
        if name not in MATERIALS:
            print(f"Unknown material: {name} — skipping")
            continue
        filepath = MATERIALS[name]
        if not os.path.exists(filepath):
            print(f"File not found: {filepath} — skipping")
            continue

        print(f"{'='*60}")
        print(f"Fitting: {name}")
        print(f"{'='*60}")

        # 1. Load & convert
        freq_Hz, Y = parse_admittance(filepath)
        print(f"  Loaded {len(freq_Hz)} points ({freq_Hz[0]:.0f}–{freq_Hz[-1]:.0f} Hz)")

        omega, R_true = admittance_to_R(freq_Hz, Y)

        # Drop DC/near-DC points
        nonzero = omega > 1.0
        if not nonzero.all():
            n_dropped = np.sum(~nonzero)
            print(f"  Dropping {n_dropped} DC/near-DC point(s)")
            omega = omega[nonzero]; R_true = R_true[nonzero]
            freq_Hz = freq_Hz[nonzero]

        print(f"  |R| range: {np.min(np.abs(R_true)):.4f} – "
              f"{np.max(np.abs(R_true)):.4f}")

        # Auto-select Ncp if not specified
        Ncp = args.Ncp
        if Ncp is None:
            R_range = np.max(np.abs(R_true)) - np.min(np.abs(R_true))
            if R_range > 0.5:
                Ncp = 5
            elif R_range > 0.05:
                Ncp = 3
            else:
                Ncp = 2

        print(f"  Using Ncp={Ncp}")

        # Initial guess and optimize (multi-start for highly absorbing materials)
        best_result = None
        best_cost = np.inf
        n_starts = 5 if Ncp >= 5 else 1  # multi-start for complex materials

        for start_i in range(n_starts):
            if start_i == 0:
                x0 = _initial_guess(omega, R_true, Ncp)
            else:
                # Random perturbation of the initial guess
                x0_base = _initial_guess(omega, R_true, Ncp)
                noise = np.random.randn(len(x0_base)) * 0.1 * np.abs(x0_base)
                x0 = x0_base + noise
                x0[0] = np.clip(x0[0], 0.0, 1.0)
                _, _, _, alpha, beta = _unpack(x0)
                alpha = np.maximum(alpha, 1e-6)
                beta = np.maximum(beta, 1e-6)
                x0 = _pack(x0[0], x0[1:Ncp+1], x0[1+Ncp:1+2*Ncp], alpha, beta)

            result = least_squares(
                _residuals, x0,
                args=(omega, R_true),
                method="trf",
                loss="soft_l1",
                f_scale=0.01,
                max_nfev=5000 // n_starts,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                verbose=0,
            )

            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
                if n_starts > 1:
                    print(f"  start {start_i}: cost={result.cost:.4e}, nfev={result.nfev}")

        result = best_result
        print(f"\n  Optimizing {len(result.x)} parameters via least_squares "
              f"({n_starts} starts) ...")
        print(f"  Status: {result.message}")
        print(f"  Best cost: {result.cost:.4e},  nfev: {result.nfev}")

        # 4. Build EDG params and enforce passivity
        edg_params = x_to_edg_params(result.x)
        edg_params = enforce_passivity(edg_params)

        # 5. Print and check
        print(f"\n  EDG parameters (CP model):")
        print(f"    RI = {edg_params['RI']:.6f}")
        CP = edg_params["CP"]
        for j in range(CP.shape[1]):
            print(f"    CP[{j}]: BS={CP[0,j]:.6e}, CS={CP[1,j]:.6e}, "
                  f"α={CP[2,j]:.6e}, β={CP[3,j]:.6e}")

        R_fit_edg = compute_Re_edg(omega, edg_params)
        print(f"\n  Quality (data band {freq_Hz[0]:.0f}–{freq_Hz[-1]:.0f} Hz):")
        print_fit_quality(name, R_true, R_fit_edg)

        # Passivity over full validation range (1 to 2π·2000 rad/s)
        omega_val = np.linspace(1.0, 2.0 * np.pi * 2000.0, 5000)
        R_val = compute_Re_edg(omega_val, edg_params)
        max_mag = np.max(np.abs(R_val))
        ok = "✓" if max_mag <= 1.0 + 1e-6 else "✗ VIOLATION"
        print(f"  max |R(ω)| (1–{2*np.pi*2000:.0f} rad/s) = {max_mag:.6f} {ok}")

        # Stability
        alphas = edg_params["CP"][2, :]
        betas = edg_params["CP"][3, :]
        stable = np.all(alphas > 0) and np.all(betas > 0)
        print(f"  CP stable: {stable}")

        # 6. Save .mat (individual arrays for main.py loading)
        savemat_kwargs = {"RI": np.array([edg_params["RI"]])}
        if "CP" in edg_params:
            savemat_kwargs["BS"] = CP[0, :]
            savemat_kwargs["CS"] = CP[1, :]
            savemat_kwargs["alphaS"] = CP[2, :]
            savemat_kwargs["betaS"] = CP[3, :]
        savemat_kwargs["freq"] = freq_Hz
        savemat_kwargs["ApproxValue"] = R_fit_edg
        savemat_kwargs["trueValue"] = R_true
        savemat_kwargs["rms_error"] = np.sqrt(np.mean(
            np.abs(R_fit_edg - R_true)**2))

        mat_path = os.path.join(CAR_CABIN_DIR, f"{name}.mat")
        savemat(mat_path, savemat_kwargs)
        print(f"  Saved: {mat_path}")

        if args.plot:
            plot_fit(name, freq_Hz, omega, R_true, R_fit_edg, CAR_CABIN_DIR)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
