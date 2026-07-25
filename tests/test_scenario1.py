"""End-to-end postprocessed receiver regression for scenario 1."""

from __future__ import annotations

import numpy

import edg_acoustics
from tests.scenario1_utils import GOLDEN_DIR, build_scenario1_simulation


POSTPROCESSED_GOLDEN_FILE = GOLDEN_DIR / "scenario1_postprocessed.npz"


def test_scenario1_simulation():
    """Keep the full postprocessed receiver response deterministic on CPU."""
    golden = numpy.load(POSTPROCESSED_GOLDEN_FILE)
    sim = build_scenario1_simulation(device="cpu")
    sim.time_integration(total_time=0.005)
    results = edg_acoustics.Monopole_postprocessor(sim, 1)
    results.apply_correction()

    numpy.testing.assert_allclose(results.TR, golden["TR"], rtol=1.0e-10, atol=1.0e-10)
    numpy.testing.assert_allclose(
        results.IRnew, golden["IRnew"], rtol=1.0e-10, atol=1.0e-10
    )
