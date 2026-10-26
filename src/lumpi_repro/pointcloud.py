"""Point-cloud transforms and distribution statistics used by synthesis jobs."""

from __future__ import annotations

import math

import numpy as np

from .constants import POINT_CLOUD_RANGE


def crop_to_range(points: np.ndarray, bounds: np.ndarray = POINT_CLOUD_RANGE) -> np.ndarray:
    """Keep points inside the inclusive project XYZ bounds."""
    lower, upper = bounds[:3], bounds[3:]
    return points[np.all((points[:, :3] >= lower) & (points[:, :3] <= upper), axis=1)]


def sample_points(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count < 1:
        raise ValueError("sample count must be positive")
    if not len(points):
        raise ValueError("cannot sample an empty point cloud")
    indices = rng.choice(len(points), size=count, replace=len(points) < count)
    return np.ascontiguousarray(points[indices, :3], dtype=np.float32)


def moments(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = points[:, :3].astype(np.float64)
    return xyz.mean(axis=0), np.cov(xyz, rowvar=False)


def feature_vector(points: np.ndarray) -> np.ndarray:
    """Compute the scale-normalized cloud descriptor used for candidate ranking."""
    xyz = points[:, :3].astype(np.float64)
    mean, covariance = moments(xyz)
    covariance_values = np.array((covariance[0, 0], covariance[1, 1], covariance[2, 2], covariance[0, 1], covariance[0, 2], covariance[1, 2]))
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    z_hist, _ = np.histogram(xyz[:, 2], bins=np.linspace(POINT_CLOUD_RANGE[2], POINT_CLOUD_RANGE[5], 11), density=True)
    radius_hist, _ = np.histogram(radius, bins=np.linspace(0.0, math.hypot(40.0, 40.0), 11), density=True)
    return np.concatenate((mean, covariance_values, z_hist, radius_hist)).astype(np.float64)


def psd_matrix_power(matrix: np.ndarray, exponent: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-8, None)
    return (vectors * np.power(values, exponent)) @ vectors.T


def affine_match(points: np.ndarray, source_mean: np.ndarray, source_covariance: np.ndarray, target_mean: np.ndarray, target_covariance: np.ndarray) -> np.ndarray:
    transform = psd_matrix_power(source_covariance, -0.5) @ psd_matrix_power(target_covariance, 0.5)
    return ((points.astype(np.float64) - source_mean) @ transform + target_mean).astype(np.float32)
