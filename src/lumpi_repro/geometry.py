"""Dependency-free rotated 3D box geometry."""

from __future__ import annotations

import numpy as np


def rectangle_corners(box: np.ndarray) -> np.ndarray:
    """Return the four XY corners of a [x,y,z,dx,dy,dz,yaw] box."""
    x, y, _, dx, dy, _, yaw = np.asarray(box, dtype=np.float64)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    corners = np.array(((-dx / 2, -dy / 2), (dx / 2, -dy / 2), (dx / 2, dy / 2), (-dx / 2, dy / 2)))
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    return corners @ rotation.T + np.array((x, y))


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Clip a convex polygon using Sutherland-Hodgman clipping."""
    output = subject
    for index, start in enumerate(clip):
        end = clip[(index + 1) % len(clip)]
        edge = end - start
        if not len(output):
            break
        next_points: list[np.ndarray] = []
        previous = output[-1]
        previous_side = np.cross(edge, previous - start)
        previous_inside = previous_side >= 0
        for current in output:
            current_side = np.cross(edge, current - start)
            current_inside = current_side >= 0
            if current_inside != previous_inside:
                fraction = previous_side / (previous_side - current_side + 1e-12)
                next_points.append(previous + fraction * (current - previous))
            if current_inside:
                next_points.append(current)
            previous, previous_side, previous_inside = current, current_side, current_inside
        output = np.asarray(next_points, dtype=np.float64).reshape(-1, 2) if next_points else np.empty((0, 2))
    return output


def box3d_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Compute IoU for two rotated 3D boxes in LUMPI box convention."""
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape[-1] != 7 or second.shape[-1] != 7:
        raise ValueError("3D boxes must have seven values")
    intersection_area = polygon_area(_clip_polygon(rectangle_corners(first), rectangle_corners(second)))
    union_area = first[3] * first[4] + second[3] * second[4] - intersection_area
    if union_area <= 0:
        return 0.0
    first_low, first_high = first[2] - first[5] / 2, first[2] + first[5] / 2
    second_low, second_high = second[2] - second[5] / 2, second[2] + second[5] / 2
    intersection_height = max(0.0, min(first_high, second_high) - max(first_low, second_low))
    union_height = (first_high - first_low) + (second_high - second_low) - intersection_height
    return float(intersection_area / union_area * (intersection_height / union_height if union_height > 0 else 0.0))
