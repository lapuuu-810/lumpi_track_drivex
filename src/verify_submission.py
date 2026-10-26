#!/usr/bin/env python3
"""Validate LUMPI ZIP structure and, optionally, the published final fingerprint."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from lumpi_repro.constants import CLASS_SET, SUBMISSION_FRAME_COUNT
from lumpi_repro.io import read_json

EXPECTED_TOTAL = 13_532
EXPECTED_CLASS_COUNTS = {"Car": 5923, "Person": 3542, "Bicycle": 1468, "Truck": 1090, "Van": 471, "Unknown": 418, "Motorcycle": 348, "Bus": 272}
EXPECTED_FIRST_SYNTHETIC_MD5 = "4710765bd4e597c2e039f22243c129b0"


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--frame-ids-json", required=True)
    parser.add_argument("--compare-synthetic")
    parser.add_argument("--check-exact", action="store_true")
    args = parser.parse_args()
    frame_ids = set(read_json(args.frame_ids_json))
    with zipfile.ZipFile(args.zip) as archive:
        names = set(archive.namelist())
        synthetic = sorted(name for name in names if name.startswith("synthetic/") and not name.endswith("/"))
        if len(synthetic) != SUBMISSION_FRAME_COUNT:
            fail(f"expected {SUBMISSION_FRAME_COUNT} synthetic files, got {len(synthetic)}")
        if not {"predictions.json", "declaration.txt"}.issubset(names):
            fail("submission is missing predictions.json or declaration.txt")
        oversized = [name for name in synthetic if len(archive.read(name)) > 1.5 * 1024 * 1024]
        if oversized:
            fail(f"oversized synthetic frames: {oversized}")
        predictions = json.loads(archive.read("predictions.json"))
        if set(predictions) != frame_ids:
            fail("prediction frame IDs do not match --frame-ids-json")
        counts, invalid = Counter(), 0
        for rows in predictions.values():
            for row in rows:
                counts[row.get("class")] += 1
                invalid += row.get("class") not in CLASS_SET or len(row.get("box", [])) != 7
        print(f"frames={len(predictions)} dets={sum(counts.values())} bad_format={invalid}")
        print(f"class counts: {dict(counts)}")
        first_hash = hashlib.md5(archive.read(synthetic[0])).hexdigest()
        print(f"synthetic 0001 md5: {first_hash} (expected {EXPECTED_FIRST_SYNTHETIC_MD5})")
        if args.compare_synthetic:
            reference = Path(args.compare_synthetic)
            matching = sum(archive.read(name) == (reference / Path(name).name).read_bytes() for name in synthetic)
            print(f"synthetic byte-compare: {matching}/{len(synthetic)} identical")
    valid = not invalid
    if args.check_exact:
        valid = valid and sum(counts.values()) == EXPECTED_TOTAL and dict(counts) == EXPECTED_CLASS_COUNTS
        label = "exact final-answer"
    else:
        label = "structural"
    print(label + f" check: {'PASS' if valid else 'FAIL'}")
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
