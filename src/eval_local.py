#!/usr/bin/env python3
"""Compute LUMPI R40 AP at 3D IoU 0.5 against label TXT files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lumpi_repro.constants import CLASSES
from lumpi_repro.fusion import canonical_class
from lumpi_repro.geometry import box3d_iou
from lumpi_repro.io import read_json


def load_labels(directory: str | Path) -> dict[str, tuple[np.ndarray, list[str]]]:
    labels = {}
    for path in sorted(Path(directory).glob("*.txt")):
        rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
        boxes = [[float(value) for value in row[:7]] for row in rows if len(row) == 8]
        names = [canonical_class(row[7]) for row in rows if len(row) == 8]
        labels[path.stem] = (np.asarray(boxes, dtype=np.float32).reshape(-1, 7), names)
    return labels


def ap_r40(scores: list[float], matches: list[int], count: int) -> float | None:
    if not count:
        return None
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    matches = np.asarray(matches, dtype=np.int64)[order]
    tp, fp = np.cumsum(matches), np.cumsum(1 - matches)
    recall, precision = tp / count, tp / np.maximum(tp + fp, 1)
    return float(np.mean([precision[recall >= level].max() if np.any(recall >= level) else 0.0 for level in np.linspace(0, 1, 41)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    args = parser.parse_args()
    predictions, labels = read_json(args.predictions), load_labels(args.labels_dir)
    stats = {name: {"scores": [], "matches": [], "count": 0} for name in CLASSES}
    for boxes, names in labels.values():
        for name in names:
            if name in stats:
                stats[name]["count"] += 1
    for frame_id, rows in predictions.items():
        boxes, names = labels.get(frame_id, (np.empty((0, 7), dtype=np.float32), []))
        used: set[int] = set()
        for row in rows:
            name = canonical_class(row.get("class"))
            if name not in stats:
                continue
            candidate, best_iou, best_index = np.asarray(row["box"], dtype=np.float32), 0.0, -1
            for index, (box, target_name) in enumerate(zip(boxes, names)):
                if index not in used and target_name == name:
                    iou = box3d_iou(candidate, box)
                    if iou > best_iou:
                        best_iou, best_index = iou, index
            matched = best_iou >= args.iou_thr
            if matched:
                used.add(best_index)
            stats[name]["scores"].append(float(row["score"]))
            stats[name]["matches"].append(int(matched))
    scores = {name: ap_r40(stat["scores"], stat["matches"], stat["count"]) for name, stat in stats.items()}
    valid = [score for score in scores.values() if score is not None]
    print(f"mAP: {float(np.mean(valid)):.4f}" if valid else "mAP: n/a")
    print(" ".join(f"{name[:4]}={score:.4f}" for name, score in scores.items() if score is not None))


if __name__ == "__main__":
    main()
