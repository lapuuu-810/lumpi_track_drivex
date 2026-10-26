"""Detection parsing, class-wise fusion, and calibration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy

import numpy as np

from .geometry import box3d_iou

Predictions = dict[str, list[dict]]


def canonical_class(value: object) -> str:
    return str(value).capitalize()


def normalize_predictions(predictions: Mapping[str, Iterable[Mapping]]) -> Predictions:
    """Copy predictions, normalize class names, validate boxes, and sort scores."""
    result: Predictions = {}
    for frame_id, detections in predictions.items():
        rows = []
        for detection in detections:
            row = dict(detection)
            if len(row.get("box", [])) != 7:
                raise ValueError(f"{frame_id}: every detection box must contain 7 values")
            row["class"] = canonical_class(row.get("class"))
            row["score"] = float(row["score"])
            row["box"] = [float(value) for value in row["box"]]
            rows.append(row)
        result[str(frame_id)] = sorted(rows, key=lambda row: -row["score"])
    return result


def fuse_pair(primary: Predictions, secondary: Predictions, class_name: str, *, primary_weight: float = 0.6, secondary_weight: float = 0.4, iou_threshold: float = 0.5) -> Predictions:
    """Fuse one class while retaining unmatched detections from both models."""
    target = canonical_class(class_name)
    result: Predictions = {}
    for frame_id in sorted(set(primary) | set(secondary)):
        first = [row for row in primary.get(frame_id, []) if canonical_class(row["class"]) == target]
        second = [row for row in secondary.get(frame_id, []) if canonical_class(row["class"]) == target]
        used_secondary: set[int] = set()
        rows: list[dict] = []
        for candidate in first:
            candidate_box = np.asarray(candidate["box"], dtype=np.float32)
            best_score, best_index = -1.0, -1
            for index, other in enumerate(second):
                if index not in used_secondary:
                    iou = box3d_iou(candidate_box, np.asarray(other["box"], dtype=np.float32))
                    if iou > best_score:
                        best_score, best_index = iou, index
            merged = dict(candidate)
            if best_score >= iou_threshold:
                used_secondary.add(best_index)
                merged["score"] = primary_weight * candidate["score"] + secondary_weight * second[best_index]["score"]
            rows.append(merged)
        rows.extend(dict(row) for index, row in enumerate(second) if index not in used_secondary)
        result[frame_id] = sorted(rows, key=lambda row: -row["score"])
    return result


def add_suppressed_third(base: Predictions, third: Predictions, class_name: str, *, score_weight: float = 0.3, iou_threshold: float = 0.5) -> Predictions:
    """Append unmatched detections from a third model at a discounted score."""
    target = canonical_class(class_name)
    result: Predictions = {}
    for frame_id in sorted(set(base) | set(third)):
        rows = [dict(row) for row in base.get(frame_id, [])]
        boxes = [np.asarray(row["box"], dtype=np.float32) for row in rows]
        for candidate in third.get(frame_id, []):
            if canonical_class(candidate["class"]) != target:
                continue
            box = np.asarray(candidate["box"], dtype=np.float32)
            if all(box3d_iou(box, existing) < iou_threshold for existing in boxes):
                row = dict(candidate)
                row["score"] = float(row["score"]) * score_weight
                rows.append(row)
                boxes.append(box)
        result[frame_id] = sorted(rows, key=lambda row: -row["score"])
    return result


def combine_predictions(*groups: Predictions) -> Predictions:
    """Combine class-specific prediction maps into score-sorted frame maps."""
    result: Predictions = {}
    for frame_id in sorted(set().union(*(group.keys() for group in groups))):
        rows = [dict(row) for group in groups for row in group.get(frame_id, [])]
        result[frame_id] = sorted(rows, key=lambda row: -row["score"])
    return result


def set_box_dimensions(predictions: Predictions, class_name: str, dimensions: tuple[float, float, float], *, z: float | None = None) -> Predictions:
    """Return a copied prediction map with one class's dimensions calibrated."""
    target, result = canonical_class(class_name), deepcopy(predictions)
    for rows in result.values():
        for row in rows:
            if canonical_class(row["class"]) == target:
                row["box"][3:6] = dimensions
                if z is not None:
                    row["box"][2] = z
    return result
