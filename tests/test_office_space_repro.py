"""Regression tests for the COMSOL office-space reproduction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy
import pytest
import scipy.io
import torch

import edg_acoustics


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "examples" / "office_space_admittance_carpet"


def load_case_module(name: str, filename: str):
    path = CASE_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(CASE_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(CASE_DIR))
    return module


def test_nastran_converter_discards_internal_shell_faces():
    converter = load_case_module("office_converter", "convert_comsol_nastran_to_gmsh.py")
    tetrahedra = numpy.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=int)
    triangles = numpy.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0, 1, 4],
            [0, 2, 4],
            [1, 2, 4],
        ],
        dtype=int,
    )

    mask, diagnostics = converter._exterior_shell_mask(triangles, tetrahedra)

    assert mask.tolist() == [False, True, True, True, True, True, True]
    assert diagnostics == {
        "exported_shell_triangles": 7,
        "exterior_shell_triangles": 6,
        "discarded_internal_shell_triangles": 1,
        "topological_boundary_triangles": 6,
    }


def test_brute_force_receiver_location_accepts_exact_shared_face_point():
    node_coordinates = numpy.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -1.0],
        ]
    )
    element_to_vertex = numpy.array([[0, 0], [1, 1], [2, 2], [3, 4]], dtype=int)
    receiver = numpy.array([[0.25], [0.25], [0.0]])

    containing = edg_acoustics.AcousticsSimulation.locate_simplex(
        node_coordinates,
        element_to_vertex,
        receiver,
        "brute_force",
    )

    assert containing.shape == (1,)
    assert containing[0] in (0, 1)


def test_generated_mesh_report_has_only_topological_exterior_shells():
    report = json.loads(
        (CASE_DIR / "office_space_mesh_conversion_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["boundary_validation"]["ok"] is True
    assert report["volume_validation"]["ok"] is True
    assert report["volume_validation"]["missing_expected_refs"] == [48]
    assert report["topology_validation"]["discarded_internal_shell_triangles"] == 2824
    assert report["diagnostics"]["boundary_topology"] == {
        "all_shells_are_exterior": True,
        "shell_face_multiplicity": {"1": 37232},
        "topological_boundary_faces": 37232,
    }
    assert report["diagnostics"]["physical_tags"]["triangle"] == {
        "11": 25836,
        "12": 1853,
        "13": 1040,
        "14": 386,
        "15": 2946,
        "16": 1978,
        "17": 2113,
        "18": 1080,
    }


def test_receiver_export_and_runtime_coordinates_match_comsol_geometry():
    case_main = load_case_module("office_main", "main.py")
    receiver = case_main.load_receivers(CASE_DIR / "office_space_receiver_points.json")

    numpy.testing.assert_allclose(
        receiver,
        numpy.array(
            [[1.5, 1.5, 4.0], [1.7, 7.3, 6.0], [1.0, 1.0, 1.0]],
            dtype=float,
        ),
        rtol=0.0,
        atol=0.0,
    )
    source = (CASE_DIR / "ExportOfficeSpaceReceiverPoints.java").read_text(
        encoding="utf-8"
    )
    assert "geometry.getVertexCoord()" in source
    assert "POINT_IDS = new int[] {230, 233, 467}" in source


def test_receiver_log_extractor_reproduces_runtime_json():
    extractor = load_case_module("office_receiver_extract", "extract_receiver_points.py")
    extracted = extractor.extract_receiver_points(CASE_DIR / "office_space_receiver_export.log")
    committed = json.loads(
        (CASE_DIR / "office_space_receiver_points.json").read_text(encoding="utf-8")
    )

    assert extracted == committed


def test_comsol_output_grid_reaches_requested_end_exactly():
    case_main = load_case_module("office_main_output", "main.py")
    times = case_main.comsol_output_times(0.4)

    assert times.size == 9001
    assert times[0] == 0.0
    assert times[-1] == 0.4
    numpy.testing.assert_allclose(numpy.diff(times), case_main.OUTPUT_DT)


@pytest.mark.parametrize("name", ["carpet", "ceiling", "gypsum"])
def test_material_fits_target_comsol_partial_fraction_functions(name: str):
    data = scipy.io.loadmat(CASE_DIR / f"{name}.mat")
    source = "".join(numpy.asarray(data["target_source"], dtype=str).reshape(-1))

    assert source == "COMSOL partial-fraction admittance"
    assert float(numpy.asarray(data["rms_error"]).reshape(-1)[0]) < 1.0e-10
    assert float(numpy.asarray(data["max_abs_R"]).reshape(-1)[0]) <= 1.0 + 1.0e-8
    assert float(
        numpy.asarray(data["table_pff_reflection_rms"]).reshape(-1)[0]
    ) > 1.0e-4


def test_comparison_requires_matching_golden_receiver_metadata(tmp_path: Path):
    compare = load_case_module("office_compare", "compare_receiver_response.py")
    golden_path = tmp_path / "golden.csv"
    golden_path.write_text(
        "\n".join(
            [
                "# receiver_point_ids,230,233,467",
                "# receiver_coordinate_unit,m",
                "# receiver_coordinate,230,1.5,1.7,1.0",
                "# receiver_coordinate,233,1.5,7.3,1.0",
                "# receiver_coordinate,467,4.0,6.0,1.0",
                "time,p230,p233,p467",
                "0.0,0.0,0.0,0.0",
                "0.1,1.0,2.0,3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _, _, golden_receiver = compare.load_comsol_golden(golden_path)
    expected = numpy.array(
        [[1.5, 1.5, 4.0], [1.7, 7.3, 6.0], [1.0, 1.0, 1.0]], dtype=float
    )
    compare.validate_edg_receiver(
        expected,
        compare.COMSOL_POINT_IDS.copy(),
        expected.copy(),
        golden_receiver,
    )

    mismatched = expected.copy()
    mismatched[0, 0] += 0.01
    with pytest.raises(ValueError, match="do not match COMSOL golden"):
        compare.validate_edg_receiver(
            expected,
            compare.COMSOL_POINT_IDS.copy(),
            mismatched,
            golden_receiver,
        )


def test_golden_log_extractor_enforces_comsol_time_grid(tmp_path: Path):
    extractor = load_case_module("office_golden_extract", "extract_comsol_golden.py")
    log_path = tmp_path / "golden.log"
    records = [
        "OFFICE_GOLDEN_RECEIVER,230,1.5,1.7,1.0,m",
        "OFFICE_GOLDEN_RECEIVER,233,1.5,7.3,1.0,m",
        "OFFICE_GOLDEN_RECEIVER,467,4.0,6.0,1.0,m",
    ]
    for sample in range(extractor.EXPECTED_NSAMPLES):
        time = sample * extractor.OUTPUT_DT
        records.append(f"OFFICE_GOLDEN_SAMPLE,{time:.17g},0,0,0")
    log_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    time, pressure, coordinates = extractor.parse_golden_log(log_path)

    assert time.size == extractor.EXPECTED_NSAMPLES
    assert time[-1] == pytest.approx(0.4)
    assert pressure.shape == (3, extractor.EXPECTED_NSAMPLES)
    assert coordinates[467] == [4.0, 6.0, 1.0]


def test_checkpoint_converts_tensor_metadata_and_keeps_step_file(tmp_path: Path):
    from types import SimpleNamespace

    sim = edg_acoustics.AcousticsSimulation.__new__(edg_acoustics.AcousticsSimulation)
    sim.IC = SimpleNamespace(
        metadata={"source_xyz": torch.tensor([4.0, 7.0, 1.5])},
        source_xyz=torch.tensor([4.0, 7.0, 1.5]),
        halfwidth=0.15,
    )
    sim.BC = SimpleNamespace(BCpara=[{"label": 11, "RI": torch.tensor(1.0)}])
    sim.prec = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    sim.prec_times = numpy.arange(1, 5, dtype=float) * 0.1
    sim.rec = numpy.zeros((3, 3))
    sim.time_integrator = SimpleNamespace(dt=0.1, Nt=3, CFL=0.5)
    sim.Ntimesteps = 4
    sim.Np = 20
    sim.N_tets = 1
    sim.rho0 = 1.2
    sim.c0 = 343.0
    sim.mesh = SimpleNamespace(filename="office.msh")
    sim.Nx = 3

    sim.save_results_on_the_run(
        output_dir=tmp_path,
        format="mat",
        step_index=4,
    )

    step_path = tmp_path / "results_step000004_t4.000000e-01.mat"
    assert step_path.exists()
    assert (tmp_path / "results_on_the_run.mat").exists()
    data = scipy.io.loadmat(step_path)
    numpy.testing.assert_allclose(data["source_xyz"].reshape(-1), [4.0, 7.0, 1.5])
    assert int(data["current_step"].item()) == 4
