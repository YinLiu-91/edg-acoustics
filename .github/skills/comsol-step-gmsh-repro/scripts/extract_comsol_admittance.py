#!/usr/bin/env python3
"""Extract one marked COMSOL admittance block from captured batch stdout."""

from __future__ import annotations

import argparse
from pathlib import Path


def extract_block(
    log_path: Path,
    marker: str,
    output_path: Path,
    expected_samples: int | None,
) -> int:
    text = log_path.read_text(encoding="utf-8")
    normalized_marker = marker.upper()
    begin = f"BEGIN_COMSOL_{normalized_marker}_ADMITTANCE"
    end = f"END_COMSOL_{normalized_marker}_ADMITTANCE"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise RuntimeError(
            f"Expected exactly one {begin}/{end} block in {log_path}"
        )

    block = text.split(begin, 1)[1].split(end, 1)[0].strip()
    output_lines: list[str] = []
    frequencies: list[float] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("%"):
            output_lines.append(line)
            continue
        columns = line.split()
        if len(columns) != 3:
            continue
        try:
            frequency, real_value, imaginary_value = map(float, columns)
        except ValueError:
            continue
        output_lines.append(
            f"{frequency:.17g} {real_value:.17g} {imaginary_value:.17g}"
        )
        frequencies.append(frequency)

    if expected_samples is not None and len(frequencies) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} samples, found {len(frequencies)}"
        )
    if not frequencies:
        raise RuntimeError(f"No numeric samples found in {begin}/{end}")
    if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
        raise RuntimeError("Exported frequencies must be strictly increasing")

    output_path.write_text(
        "\n".join(output_lines) + "\n", encoding="utf-8"
    )
    return len(frequencies)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = extract_block(
        args.log,
        args.marker,
        args.output,
        args.expected_samples,
    )
    print(f"Wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
