"""Regression tests for the COMSOL wave-based-room reproduction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy
import pytest
import scipy.io
import torch

import edg_acoustics


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "examples" / "wave_based_room"


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


def test_wave_based_room_boundary_groups_match_active_comsol_features():
    groups = load_case_module("wave_groups", "wave_based_room_boundary_groups.py")
    model = groups.recover_boundary_model(exported_boundary_refs=list(range(1, 263)))
    by_key = {entry["key"]: entry for entry in model["physical_groups"]}

    assert by_key["carpet"]["entities"] == [3, 75]
    assert by_key["ceiling"]["entities"] == [7, 77]
    assert by_key["normal_velocity_source"]["entities"] == [222]
    assert by_key["wall"]["entities"] == [1, 2, 4, 5, 8, 9, 74, 78, 262]
    assert by_key["sofa"]["entity_count"] == 25
    assert model["receiver_point_ids"] == [122, 121, 53, 35]


def test_wave_based_room_runtime_parameters_match_comsol_case():
    case_main = load_case_module("wave_main", "main.py")
    receivers = case_main.load_receivers(CASE_DIR / "wave_based_room_receiver_points.json")

    assert case_main.F0 == 700.0
    assert case_main.TEND == pytest.approx(30.0 / 700.0)
    numpy.testing.assert_allclose(
        receivers,
        numpy.array(
            [
                [1.2, 0.2, -0.8, -1.8],
                [1.3125, 0.875, 0.4375, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
    )
    params = {entry["label"]: entry for entry in case_main.build_bc_parameters()}
    assert params[21]["normal_velocity"]["frequency"] == 700.0
    assert params[21]["normal_velocity"]["delay"] == pytest.approx(2.0 / 700.0)
    assert params[21]["normal_velocity"]["sigma"] == pytest.approx(0.5 / 700.0)


def test_wave_based_room_output_grid_reaches_comsol_end():
    case_main = load_case_module("wave_main_output", "main.py")
    times = case_main.comsol_output_times(case_main.TEND)

    assert times.size == 31
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(30.0 / 700.0)
    numpy.testing.assert_allclose(numpy.diff(times), 1.0 / 700.0)


@pytest.mark.parametrize("name", ["carpet", "ceiling", "sofa", "wall"])
def test_wave_based_room_material_files_are_passive(name: str):
    mat_path = CASE_DIR / f"{name}.mat"
    if not mat_path.exists():
        pytest.skip(f"{mat_path} has not been generated")
    data = scipy.io.loadmat(mat_path)
    params = {"label": 0, "RI": float(numpy.asarray(data["RI"]).reshape(-1)[0])}
    if "AS" in data and "lambdaS" in data and data["AS"].size:
        params["RP"] = numpy.vstack(
            (
                numpy.asarray(data["AS"]).reshape(-1),
                numpy.asarray(data["lambdaS"]).reshape(-1),
            )
        )
    if {"BS", "CS", "alphaS", "betaS"} <= set(data) and data["BS"].size:
        params["CP"] = numpy.vstack(
            (
                numpy.asarray(data["BS"]).reshape(-1),
                numpy.asarray(data["CS"]).reshape(-1),
                numpy.asarray(data["alphaS"]).reshape(-1),
                numpy.asarray(data["betaS"]).reshape(-1),
            )
        )

    omega = torch.linspace(1.0, 2.0 * numpy.pi * 2100.0, 5000)
    reflection = edg_acoustics.AbsorbBC.compute_Re(omega, params)

    assert float(numpy.asarray(data["max_abs_R"]).reshape(-1)[0]) <= 1.0 + 1.0e-8
    assert torch.max(torch.abs(reflection)).item() <= 1.0 + 1.0e-8
    if "RP" in params:
        assert numpy.all(params["RP"][1] > 0.0)
    if "CP" in params:
        assert numpy.all(params["CP"][2] > 0.0)


def test_wave_based_room_golden_parser_writes_pa_and_normalized(tmp_path: Path):
    extractor = load_case_module("wave_golden_extract", "extract_comsol_golden.py")
    log_path = tmp_path / "golden.log"
    records = [
        "WAVE_GOLDEN_RECEIVER,122,1.2,1.3125,1.0,m",
        "WAVE_GOLDEN_RECEIVER,121,0.2,0.875,1.0,m",
        "WAVE_GOLDEN_RECEIVER,53,-0.8,0.4375,1.0,m",
        "WAVE_GOLDEN_RECEIVER,35,-1.8,0.0,1.0,m",
        "WAVE_GOLDEN_SAMPLE,0.0,411.6,0.0,0.0,0.0",
        "WAVE_GOLDEN_NORMALIZED,0.0,1.0,0.0,0.0,0.0",
        "WAVE_GOLDEN_SAMPLE,0.001,0.0,411.6,0.0,0.0",
        "WAVE_GOLDEN_NORMALIZED,0.001,0.0,1.0,0.0,0.0",
    ]
    log_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    data = extractor.parse_golden_log(log_path)

    assert data["pressure_pa"].shape == (4, 2)
    assert data["pressure_normalized"].shape == (4, 2)
    assert data["pressure_pa"][0, 0] == pytest.approx(411.6)
    assert data["pressure_normalized"][0, 0] == pytest.approx(1.0)
