"""Generate the room mesh with the Gmsh Python API.

This reproduces the validated mesh topology used by the local ``main.py``.
The direct ``gmsh room.geo -3`` path currently over-refines this STEP import,
so this script is the reproducible way to refresh ``room.msh``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gmsh


RHO0 = 1.213
C0 = 343.0
F0 = 700.0
LC = C0 / F0 / 3.0
LC_MIN = 0.04
EPS = 1.0e-5


def _bbox(tag: int) -> tuple[float, float, float, float, float, float]:
    return gmsh.model.getBoundingBox(2, tag)


def _near_plane(tag: int, axis: int, value: float) -> bool:
    box = _bbox(tag)
    return abs(box[axis] - value) < EPS and abs(box[axis + 3] - value) < EPS


def build_mesh(step_path: Path, output_path: Path) -> None:
    gmsh.initialize()
    try:
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", LC_MIN)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", LC)

        gmsh.open(str(step_path))
        gmsh.model.occ.synchronize()
        gmsh.model.occ.fragment(
            gmsh.model.getEntities(3), [], removeObject=True, removeTool=True
        )
        gmsh.model.occ.synchronize()

        external = []
        for _, surface in gmsh.model.getEntities(2):
            up, _ = gmsh.model.getAdjacencies(2, surface)
            if len(up) == 1:
                external.append(surface)

        source = {205}
        carpet = {surface for surface in external if _near_plane(surface, 2, 0.0)}
        ceiling = {surface for surface in external if _near_plane(surface, 2, 2.6)}
        walls = {
            surface
            for surface in external
            if (
                _near_plane(surface, 0, -2.5)
                or _near_plane(surface, 0, 2.5)
                or _near_plane(surface, 1, -2.0)
                or _near_plane(surface, 1, 2.0)
            )
        }
        sofa = set()
        for surface in external:
            box = _bbox(surface)
            if (
                box[0] >= -2.12 - EPS
                and box[3] <= -0.11 + EPS
                and box[1] >= -1.47 - EPS
                and box[4] <= 1.47 + EPS
                and box[2] >= 0.19 - EPS
                and box[5] <= 0.80 + EPS
            ):
                sofa.add(surface)

        carpet -= source
        ceiling -= source | carpet
        walls -= source | carpet | ceiling
        sofa -= source | carpet | ceiling | walls
        other = sorted(set(external) - source - carpet - ceiling - walls - sofa)

        gmsh.model.addPhysicalGroup(2, other, 11, "hard_other")
        gmsh.model.addPhysicalGroup(2, sorted(walls), 12, "walls")
        gmsh.model.addPhysicalGroup(2, sorted(carpet), 13, "carpet")
        gmsh.model.addPhysicalGroup(2, sorted(ceiling), 14, "ceiling")
        gmsh.model.addPhysicalGroup(2, sorted(sofa), 15, "sofa")
        gmsh.model.addPhysicalGroup(2, sorted(source), 16, "source_comsol_222")
        gmsh.model.addPhysicalGroup(
            3, [tag for _, tag in gmsh.model.getEntities(3)], 1, "room_air"
        )

        gmsh.model.mesh.generate(3)
        gmsh.write(str(output_path))
    finally:
        gmsh.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        type=Path,
        default=Path(__file__).resolve().with_name("room.step"),
        help="Input STEP file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("room.msh"),
        help="Output MSH file.",
    )
    args = parser.parse_args()
    build_mesh(args.step.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
