"""Regression tests for temporary gmsh snapshot export."""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy

from tests.scenario1_utils import build_scenario1_simulation


def test_reference_tetra_vertex_node_indices():
    sim = build_scenario1_simulation(device="cpu")
    assert sim._reference_tetra_vertex_node_indices() == (0, 4, 14, 34)


def test_save_mesh_results_on_the_run(tmp_path: Path):
    sim = build_scenario1_simulation(device="cpu")
    sim.save_mesh_results_on_the_run(
        output_dir=str(tmp_path),
        step_index=12,
        real_time=1.234567e-4,
    )

    saved_files = list(tmp_path.glob("*.msh"))
    assert len(saved_files) == 1
    saved_file = saved_files[0]
    assert "step000012" in saved_file.name
    assert "t1.234567e-04" in saved_file.name

    mesh = meshio.read(saved_file)
    assert mesh.points.shape[0] == sim.mesh.N_vertices
    assert set(mesh.point_data) == {"P", "Vx", "Vy", "Vz"}
    for key in ("P", "Vx", "Vy", "Vz"):
        assert mesh.point_data[key].shape == (sim.mesh.N_vertices,)
        assert numpy.isfinite(mesh.point_data[key]).all()
