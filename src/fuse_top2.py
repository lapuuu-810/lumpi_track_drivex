#!/usr/bin/env python3
"""Class-wise, score-weighted two-model fusion."""

from __future__ import annotations

import argparse

from lumpi_repro.fusion import fuse_pair
from lumpi_repro.io import read_json, write_json


def parse_source(value: str) -> tuple[str, str, str]:
    try:
        class_name, paths = value.split("=", 1)
        primary, secondary = paths.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("source must be Class=primary.json:secondary.json") from exc
    return class_name.strip(), primary, secondary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", action="append", required=True, type=parse_source)
    parser.add_argument("--w-primary", type=float, default=0.6)
    parser.add_argument("--w-secondary", type=float, default=0.4)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    args = parser.parse_args()

    output = {}
    for class_name, primary_path, secondary_path in args.source:
        merged = fuse_pair(read_json(primary_path), read_json(secondary_path), class_name,
                            primary_weight=args.w_primary, secondary_weight=args.w_secondary,
                            iou_threshold=args.iou_thr)
        for frame_id, rows in merged.items():
            output.setdefault(frame_id, []).extend(rows)
    for rows in output.values():
        rows.sort(key=lambda row: -float(row["score"]))
    write_json(args.output, output)
    print(f"wrote {args.output}; frames={len(output)}")


if __name__ == "__main__":
    main()
