"""Regression tests for COMSOL car-cabin boundary recovery."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "car_cabin_acoustics_transient_63_cleared"
MODULE_PATH = EXAMPLE_DIR / "car_cabin_boundary_groups.py"
MPH_PATH = EXAMPLE_DIR / "car_cabin_acoustics_transient_63_cleared.mph"
STEP_PATH = EXAMPLE_DIR / "car_cabin_acoustics_transient_63_cleared.step"
GEO_PATH = EXAMPLE_DIR / "car_cabin_acoustics_transient_63_cleared.geo"

pytestmark = pytest.mark.skipif(
    not (MPH_PATH.exists() and STEP_PATH.exists() and GEO_PATH.exists()),
    reason="car cabin .mph/.step/.geo example files are required",
)


def load_boundary_module():
    spec = importlib.util.spec_from_file_location("car_cabin_boundary_groups", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mph_boundary_selections_and_physics_are_recovered():
    module = load_boundary_module()

    model = module.recover_boundary_model(MPH_PATH)
    selections = model["selections"]
    physics = model["physics_features"]
    groups = {group["key"]: group for group in model["physical_groups"]}

    assert selections["sel2"]["name"] == "Windows"
    assert selections["sel2"]["entity_count"] == 12
    assert selections["sel3"]["entity_count"] == 85
    assert selections["sel4"]["entities"] == [7]
    assert selections["sel6"]["entity_count"] == 226
    assert selections["sel7"]["entities"] == [156, 157, 292, 293, 366, 367]
    assert selections["sel12"]["entities"] == [32, 33]

    assert physics["imp2"]["entities"] == [108, 109, 175, 176, 298, 302, 321, 322]
    assert physics["imp4"]["params"]["ImpedanceModel"] == "RationalApproximation"
    assert physics["imp4"]["params"]["ApproximantFunctionReference"] == "pff1"
    assert physics["imp5"]["params"]["R"] == [
        "-38.03972583071767",
        "-0.10961576400452967",
        "-0.004058968542651085",
    ]
    assert physics["nvel1"]["selection_tag"] == "sel12"
    assert physics["nvel1"]["params"]["nvel"] == "vn(t)"

    assert groups["default_hard_wall"]["label"] == 11
    assert groups["default_hard_wall"]["entity_count"] == 104
    assert groups["doors"]["entity_count"] == 8
    assert {298, 302} <= set(groups["doors"]["entities"])
    assert {298, 302}.isdisjoint(groups["default_hard_wall"]["entities"])
    assert groups["inactive_speakers_hard_wall"]["entities"] == [
        34,
        35,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
    ]

    parameters = model["parameters"]
    assert parameters["c0"]["scalar_real"] == 343.0
    assert parameters["rho0"]["scalar_real"] == 1.2
    assert parameters["Z_win"]["scalar_real"] == pytest.approx(328456.28420971055)
    assert parameters["Z_dash"]["scalar_real"] == pytest.approx(163815.76582261728)
    assert parameters["Tend"]["scalar_real"] == pytest.approx(0.06)


def test_step_surface_tags_cover_recovered_physical_groups():
    pytest.importorskip("gmsh")
    module = load_boundary_module()

    surface_tags = module.gmsh_step_surface_tags(STEP_PATH, scale=0.001)
    model = module.recover_boundary_model(MPH_PATH, surface_tags=surface_tags)
    validation = module.validate_physical_groups(model, surface_tags)
    groups = {group["key"]: group for group in model["physical_groups"]}

    assert len(surface_tags) == 859
    assert surface_tags[0] == 1
    assert surface_tags[-1] == 859
    assert groups["default_hard_wall"]["entity_count"] == 509
    assert validation == {
        "ok": True,
        "surface_count": 859,
        "covered_surface_count": 859,
        "uncovered_surface_count": 0,
        "uncovered_surfaces": [],
        "duplicate_surfaces": [],
        "missing_surfaces_by_group": {},
        "empty_groups": [],
    }


@pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh is required")
def test_grouped_geo_generates_mesh_with_expected_physical_tags(tmp_path: Path):
    module = load_boundary_module()
    mesh_path = tmp_path / "car_cabin_grouped.msh"

    subprocess.run(
        [
            "gmsh",
            "-3",
            str(GEO_PATH),
            "-setnumber",
            "lc",
            "0.45",
            "-format",
            "msh2",
            "-o",
            str(mesh_path),
        ],
        cwd=EXAMPLE_DIR,
        check=True,
        text=True,
        capture_output=True,
        timeout=120,
    )

    diagnostics = module.mesh_diagnostics(mesh_path)
    triangle_tags = {
        int(tag) for tag in diagnostics["physical_tags"]["triangle"].keys()
    }
    tetra_tags = {int(tag) for tag in diagnostics["physical_tags"]["tetra"].keys()}

    assert triangle_tags == set(module.PHYSICAL_GROUP_LABELS.values())
    assert tetra_tags == {1}
    assert diagnostics["tetra_quality"]["min_volume"] > 0.0
    assert diagnostics["triangle_quality"]["min_area"] > 0.0
    assert sum(diagnostics["physical_tags"]["triangle"].values()) == diagnostics["cells"][
        "triangle"
    ]
