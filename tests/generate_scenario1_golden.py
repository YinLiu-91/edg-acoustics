"""Generate compact fp64 scenario 1 golden data.

Run from the repository root:

    python tests/generate_scenario1_golden.py
"""

from __future__ import annotations

import numpy

import edg_acoustics
from scenario1_utils import GOLDEN_DIR, build_scenario1_simulation, clone_bcvar, tensor_to_numpy


GOLDEN_FILE = GOLDEN_DIR / "scenario1_fp64_short.npz"
POSTPROCESSED_GOLDEN_FILE = GOLDEN_DIR / "scenario1_postprocessed.npz"
N_TIME_STEPS = 20
POSTPROCESS_TOTAL_TIME = 0.005


def collect_bc_state(prefix: str, bcvar: list[dict], output: dict):
    for index, state in enumerate(bcvar):
        for key, value in state.items():
            if hasattr(value, "detach"):
                output[f"{prefix}_bc{index}_{key}"] = tensor_to_numpy(value)


def main():
    rhs_sim = build_scenario1_simulation(device="cpu")
    rhs_bcvar = clone_bcvar(rhs_sim.BC.BCvar)
    rhs_p, rhs_vx, rhs_vy, rhs_vz, rhs_bcvar = rhs_sim.RHS_operator(
        rhs_sim.P,
        rhs_sim.Vx,
        rhs_sim.Vy,
        rhs_sim.Vz,
        rhs_bcvar,
    )

    time_sim = build_scenario1_simulation(device="cpu")
    time_sim.time_integration(n_time_steps=N_TIME_STEPS)

    data = {
        "n_time_steps": numpy.array(N_TIME_STEPS),
        "dtype": numpy.array("float64"),
        "nx": numpy.array(time_sim.Nx),
        "nt": numpy.array(time_sim.time_integrator.Nt),
        "cfl": numpy.array(time_sim.time_integrator.CFL),
        "dt": numpy.array(time_sim.time_integrator.dt),
        "rhs_p": tensor_to_numpy(rhs_p),
        "rhs_vx": tensor_to_numpy(rhs_vx),
        "rhs_vy": tensor_to_numpy(rhs_vy),
        "rhs_vz": tensor_to_numpy(rhs_vz),
        "prec": tensor_to_numpy(time_sim.prec),
        "final_p": tensor_to_numpy(time_sim.P),
        "final_vx": tensor_to_numpy(time_sim.Vx),
        "final_vy": tensor_to_numpy(time_sim.Vy),
        "final_vz": tensor_to_numpy(time_sim.Vz),
    }
    collect_bc_state("rhs", rhs_bcvar, data)
    collect_bc_state("final", time_sim.BC.BCvar, data)

    numpy.savez_compressed(GOLDEN_FILE, **data)
    print(f"Wrote {GOLDEN_FILE}")

    postprocess_sim = build_scenario1_simulation(device="cpu")
    postprocess_sim.time_integration(total_time=POSTPROCESS_TOTAL_TIME)
    postprocessor = edg_acoustics.Monopole_postprocessor(postprocess_sim, 1)
    postprocessor.apply_correction()
    numpy.savez_compressed(
        POSTPROCESSED_GOLDEN_FILE,
        TR=postprocessor.TR,
        IRnew=postprocessor.IRnew,
    )
    print(f"Wrote {POSTPROCESSED_GOLDEN_FILE}")


if __name__ == "__main__":
    main()
