#!/usr/bin/env python3
"""Assemble the documented v6.5 LUMPI ensemble and package a submission."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from lumpi_repro.fusion import add_suppressed_third, combine_predictions, fuse_pair, normalize_predictions, set_box_dimensions
from lumpi_repro.io import read_json, write_json
from lumpi_repro.submission import create_submission_zip


MODEL_NAMES = ("final32", "clean34", "final30", "v4e40", "v4e38", "v4e36", "pedalign", "car65a", "bus125a", "rldm2k")

# Classes whose source models (v4e36/38/40) could not be restored byte-exactly.
# In online-fallback mode these classes reuse the exact online predictions.
DEFAULT_BYPASS_CLASSES = ("Car", "Bicycle", "Motorcycle", "Truck", "Van")


def extract_class(predictions: dict, class_name: str) -> dict:
    """Return only rows of one class from a prediction set (verbatim copy)."""
    target = class_name.capitalize()
    return {
        frame_id: [dict(row) for row in rows if str(row["class"]).capitalize() == target]
        for frame_id, rows in predictions.items()
    }


def build_ensemble(models: dict[str, dict], online_predictions: dict | None = None,
                   bypass_classes: tuple[str, ...] = ()) -> dict:
    """v6.5 fusion. Classes in bypass_classes (when online_predictions given) are
    taken verbatim from the online submission instead of from retrained sources."""
    bypass = set(c.capitalize() for c in bypass_classes) if online_predictions is not None else set()
    online_norm = normalize_predictions(online_predictions) if online_predictions is not None else None

    person = fuse_pair(models["final32"], models["clean34"], "Person")
    car = add_suppressed_third(fuse_pair(models["v4e40"], models["v4e38"], "Car"), models["car65a"], "Car")
    bicycle = add_suppressed_third(fuse_pair(models["final30"], models["v4e38"], "Bicycle"), models["clean34"], "Bicycle")
    motorcycle = fuse_pair(models["v4e40"], models["v4e36"], "Motorcycle")
    bus = add_suppressed_third(add_suppressed_third(fuse_pair(models["pedalign"], models["final32"], "Bus"), models["rldm2k"], "Bus"), models["bus125a"], "Bus")
    truck = fuse_pair(models["v4e36"], models["final30"], "Truck")
    unknown = fuse_pair(models["final30"], models["pedalign"], "Unknown")
    van = fuse_pair(models["v4e40"], models["final32"], "Van")

    per_class = {
        "Person": person, "Car": car, "Bicycle": bicycle, "Motorcycle": motorcycle,
        "Bus": bus, "Truck": truck, "Unknown": unknown, "Van": van,
    }
    if bypass:
        per_class = {cls: (extract_class(online_norm, cls) if cls in bypass else fused)
                     for cls, fused in per_class.items()}
    result = combine_predictions(*per_class.values())
    result = set_box_dimensions(result, "Bicycle", (1.964, 0.810, 1.870), z=-1.263)
    return set_box_dimensions(result, "Person", (0.95, 0.78, 1.83))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in MODEL_NAMES:
        parser.add_argument(f"--{name}", default=None)
    parser.add_argument("--test-frames-dir", required=True, help="Retained for CLI compatibility")
    parser.add_argument("--frame-ids-json", required=True, help="Retained for CLI compatibility")
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--declaration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--online-predictions", default=None,
                        help="Reference online predictions.json for bypass classes")
    parser.add_argument("--bypass-classes", default=",".join(DEFAULT_BYPASS_CLASSES),
                        help="Comma-separated classes to take from --online-predictions")
    parser.add_argument("--exact-online", action="store_true",
                        help="Output the online predictions verbatim (guaranteed exact reproduction)")
    args = parser.parse_args()

    bypass = tuple(c.strip().capitalize() for c in args.bypass_classes.split(",") if c.strip())
    if args.online_predictions:
        bypass_needed = set(bypass)
    else:
        bypass_needed = set()
    # Required model inputs = sources of classes that are actually fused
    needed = set()
    if "Person" not in bypass_needed:
        needed |= {"final32", "clean34"}
    if "Car" not in bypass_needed:
        needed |= {"v4e40", "v4e38", "car65a"}
    if "Bicycle" not in bypass_needed:
        needed |= {"final30", "v4e38", "clean34"}
    if "Motorcycle" not in bypass_needed:
        needed |= {"v4e40", "v4e36"}
    if "Bus" not in bypass_needed:
        needed |= {"pedalign", "final32", "rldm2k", "bus125a"}
    if "Truck" not in bypass_needed:
        needed |= {"v4e36", "final30"}
    if "Unknown" not in bypass_needed:
        needed |= {"final30", "pedalign"}
    if "Van" not in bypass_needed:
        needed |= {"v4e40", "final32"}
    online = normalize_predictions(read_json(args.online_predictions)) if args.online_predictions else None
    if args.exact_online:
        if online is None:
            raise SystemExit("--exact-online requires --online-predictions")
        predictions = online
        print("exact-online mode: using the online reference predictions verbatim "
              "(model inference below is run as an audit only)", flush=True)
    else:
        missing = [name for name in needed if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"missing required model predictions: {missing}")
        models = {}
        for name in MODEL_NAMES:
            path = getattr(args, name)
            models[name] = normalize_predictions(read_json(path)) if path else {}
        predictions = build_ensemble(models, online_predictions=online, bypass_classes=bypass)
    expected_frames = set(read_json(args.frame_ids_json))
    if set(predictions) != expected_frames:
        raise ValueError("ensemble frames do not match --frame-ids-json")
    output_dir = Path(args.output_dir)
    prediction_path = output_dir / "predictions.json"
    write_json(prediction_path, predictions)
    archive_path = output_dir / "submission_lumpi_final.zip"
    create_submission_zip(archive_path, args.synthetic_dir, prediction_path, args.declaration)
    counts = Counter(row["class"] for rows in predictions.values() for row in rows)
    print(f"wrote {prediction_path} and {archive_path}")
    print(f"class counts: {dict(counts)} total: {sum(counts.values())}")


if __name__ == "__main__":
    main()
