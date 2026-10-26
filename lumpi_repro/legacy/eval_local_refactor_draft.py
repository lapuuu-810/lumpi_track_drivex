#!/usr/bin/env python3
"""Compute LUMPI R40 AP at 3D IoU 0.5 against label TXT files."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from lumpi_repro.constants import CLASSES
from lumpi_repro.fusion import canonical_class
from lumpi_repro.geometry import box3d_iou
from lumpi_repro.io import read_json


def load_labels(directory: str | Path) -> dict[str, tuple[np.ndarray, list[str]]]:
    result = {}
    for path in sorted(Path(directory).glob("*.txt")):
        boxes, names = [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 8:
                boxes.append([float(value) for value in fields[:7]])
                names.append(canonical_class(fields[7]))
        result[path.stem] = (np.asarray(boxes, dtype=np.float32).reshape(-1, 7), names)
    return result


def average_precision(scores: list[float], matches: list[int], ground_truth_count: int) -> float | None:
    if not ground_truth_count:
        return None
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    matches_array = np.asarray(matches, dtype=np.int64)[order]
    true_positive = np.cumsum(matches_array)
    false_positive = np.cumsum(1 - matches_array)
    recall = true_positive / ground_truth_count
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    return float(np.mean([(precision[recall >= level].max() if np.any(recall >= level) else 0.0) for level in np.linspace(0, 1, 41)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    args = parser.parse_args()

    labels = load_labels(args.labels_dir)
    records = {class_name: defaultdict(list) for class_name in CLASSES}
    ground_truth = {class_name: 0 for class_name in CLASSES}
    for frame_id, (boxes, names) in labels.items():
        for name in names:
            if name in ground_truth:
                ground_truth[name] += 1
        used: set[int] = set()
        for row in sorted(read_json(args.predictions).get(frame_id, []), key=lambda item: -float(item["score"])):
            class_name = canonical_class(row["class"])
            if class_name not in records:
                continue
            candidate, best_index, best_iou = np.asarray(row["box"], dtype=np.float32), -1, 0.0
            for index, (box, name) in enumerate(zip(boxes, names)):
                if index not in used and name == class_name:
                    iou = box3d_iou(candidate, box)
                    if iou > best_iou:
                        best_iou, best_index = iou, index
            matched = best_iou >= args.iou_thr
            if matched:
                used.add(best_index)
            records[class_name]["scores"].append(float(row["score"]))
            records[class_name]["matches"].append(int(matched))
    scores = {class_name: average_precision(records[class_name]["scores"], records[class_name]["matches"], ground_truth[class_name]) for class_name in CLASSES}
    valid = [value for value in scores.values() if value is not None]
    print(f"mAP: {np.mean(valid):.4f}" if valid else "mAP: n/a")
    print(" ".join(f"{name[:4]}={value:.4f}" for name, value in scores.items() if value is not None))


if __name__ == "__main__":
    main()
