"""Generate an intermediate scenario1 mesh from scenario1_fine.geo."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import edg_acoustics


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "scenario1"
BC_LABELS = {
    "hard wall": 11,
    "carpet": 13,
    "panel": 14,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=EXAMPLE_DIR / "scenario1_fine.geo",
        help="Source .geo file.",
    )
    parser.add_argument(
        "--lc",
        type=float,
        default=0.20,
        help="Characteristic length written to the top-level lc assignment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXAMPLE_DIR / "scenario1_profile_lc0p20.msh",
        help="Output .msh path.",
    )
    parser.add_argument(
        "--max-tets",
        type=int,
        default=100_000,
        help="Fail if generated mesh exceeds this tetrahedron count.",
    )
    return parser.parse_args()


def write_geo_with_lc(source: Path, destination: Path, lc: float):
    text = source.read_text()
    text, replacements = re.subn(
        r"(?m)^lc\s*=\s*[^;]+;",
        f"lc={lc};",
        text,
        count=1,
    )
    if replacements != 1:
        raise ValueError(f"Expected exactly one top-level lc assignment in {source}")
    destination.write_text(text)


def main():
    args = parse_args()
    gmsh = shutil.which("gmsh")
    if gmsh is None:
        raise FileNotFoundError("gmsh executable not found in PATH")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated_geo = args.output.with_suffix(".geo")
    write_geo_with_lc(args.source, generated_geo, args.lc)

    subprocess.run(
        [gmsh, "-3", str(generated_geo), "-format", "msh2", "-o", str(args.output)],
        check=True,
    )

    mesh = edg_acoustics.Mesh(str(args.output), BC_LABELS)
    n_tets = mesh.EToV.shape[1]
    if n_tets > args.max_tets:
        args.output.unlink(missing_ok=True)
        generated_geo.unlink(missing_ok=True)
        raise ValueError(
            f"Generated mesh has {n_tets} tetrahedra, above --max-tets={args.max_tets}"
        )

    print(f"source={args.source}")
    print(f"generated_geo={generated_geo}")
    print(f"output={args.output}")
    print(f"lc={args.lc}")
    print(f"vertices={mesh.vertices.shape[1]}")
    print(f"tets={n_tets}")


if __name__ == "__main__":
    main()
