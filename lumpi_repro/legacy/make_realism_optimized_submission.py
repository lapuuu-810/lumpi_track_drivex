#!/usr/bin/env python3
"""Build a LUMPI submission with synthetic frames selected for realism metrics."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np


RANGE = np.array([-40.0, -40.0, -5.0, 40.0, 40.0, 5.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--starting-kit", required=True)
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-mode", choices=("synthetic_pool", "detection_test"), default="synthetic_pool")
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--points-per-frame", type=int, default=50000)
    parser.add_argument("--candidate-sample-points", type=int, default=5000)
    parser.add_argument("--target-sample-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--affine-match-target", action="store_true")
    parser.add_argument("--team-name", default="baseline5_realism_optimized")
    return parser.parse_args()


def load_forbidden(starting_kit: Path) -> set[str]:
    path = starting_kit / "forbidden_frames.txt"
    if not path.exists():
        return set()
    return {x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}


def load_points(path: Path, bin_dim: int | None = None) -> np.ndarray:
    if path.suffix == ".npy":
        pts = np.load(path)
    else:
        raw = np.fromfile(path, dtype=np.float32)
        if bin_dim is not None:
            if raw.size % bin_dim != 0:
                raise ValueError(f"{path} cannot be reshaped to N x {bin_dim}")
            pts = raw.reshape(-1, bin_dim)[:, :3]
        elif raw.size % 3 == 0:
            pts = raw.reshape(-1, 3)
        elif raw.size % 4 == 0:
            pts = raw.reshape(-1, 4)[:, :3]
        else:
            raise ValueError(f"cannot infer point dimension for {path}")
    return np.asarray(pts[:, :3], dtype=np.float32)


def crop_range(points: np.ndarray) -> np.ndarray:
    m = (
        (points[:, 0] >= RANGE[0])
        & (points[:, 1] >= RANGE[1])
        & (points[:, 2] >= RANGE[2])
        & (points[:, 0] <= RANGE[3])
        & (points[:, 1] <= RANGE[4])
        & (points[:, 2] <= RANGE[5])
    )
    return points[m]


def sample_points(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("empty point cloud after range crop")
    replace = len(points) < n
    idx = rng.choice(len(points), size=n, replace=replace)
    return np.ascontiguousarray(points[idx], dtype=np.float32)


def feature_from_points(points: np.ndarray) -> np.ndarray:
    xyz = points[:, :3].astype(np.float64)
    mean = xyz.mean(axis=0)
    cov = np.cov(xyz, rowvar=False)
    cov_vals = np.array([cov[0, 0], cov[1, 1], cov[2, 2], cov[0, 1], cov[0, 2], cov[1, 2]], dtype=np.float64)
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    z_hist, _ = np.histogram(xyz[:, 2], bins=np.linspace(RANGE[2], RANGE[5], 11), density=True)
    r_hist, _ = np.histogram(r, bins=np.linspace(0.0, math.sqrt(40.0**2 + 40.0**2), 11), density=True)
    return np.concatenate([mean, cov_vals, z_hist, r_hist]).astype(np.float64)


def moments(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = points[:, :3].astype(np.float64)
    return xyz.mean(axis=0), np.cov(xyz, rowvar=False)


def sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 1e-8, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def invsqrtm_psd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 1e-8, None)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def affine_match(points: np.ndarray, src_mu: np.ndarray, src_cov: np.ndarray, tgt_mu: np.ndarray, tgt_cov: np.ndarray) -> np.ndarray:
    transform = invsqrtm_psd(src_cov) @ sqrtm_psd(tgt_cov)
    return ((points.astype(np.float64) - src_mu) @ transform + tgt_mu).astype(np.float32)


def load_target_samples(args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    starting_kit = Path(args.starting_kit)
    if args.target_mode == "detection_test":
        frame_ids = json.loads((starting_kit / "detection_test_frame_ids.json").read_text(encoding="utf-8"))
        samples = []
        for frame_id in frame_ids:
            pts = crop_range(load_points(starting_kit / "detection_test_frames" / f"{frame_id}.bin", bin_dim=3))
            samples.append(sample_points(pts, args.target_sample_points, rng))
        return np.concatenate(samples, axis=0)

    dataset_root = Path(args.dataset_root)
    forbidden = load_forbidden(starting_kit)
    frame_ids = [p.stem for p in sorted((dataset_root / "pcd").glob("*.npy")) if p.stem not in forbidden]
    step = max(1, len(frame_ids) // 500)
    samples = []
    for frame_id in frame_ids[::step]:
        pts = crop_range(load_points(dataset_root / "pcd" / f"{frame_id}.npy"))
        samples.append(sample_points(pts, min(args.target_sample_points, len(pts)), rng))
    return np.concatenate(samples, axis=0)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    dataset_root = Path(args.dataset_root)
    starting_kit = Path(args.starting_kit)
    output_dir = Path(args.output_dir)
    submission_dir = output_dir / "submission"
    synthetic_dir = submission_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    forbidden = load_forbidden(starting_kit)
    candidates = [p.stem for p in sorted((dataset_root / "pcd").glob("*.npy")) if p.stem not in forbidden]
    if len(candidates) < args.num_frames:
        raise ValueError(f"not enough candidates: {len(candidates)}")

    target_samples = load_target_samples(args, rng)
    target_feature = feature_from_points(target_samples)
    target_mu, target_cov = moments(target_samples)

    rows = []
    feature_stack = []
    for frame_id in candidates:
        pts = crop_range(load_points(dataset_root / "pcd" / f"{frame_id}.npy"))
        sample = sample_points(pts, min(args.candidate_sample_points, len(pts)), rng)
        feat = feature_from_points(sample)
        rows.append((frame_id, feat))
        feature_stack.append(feat)

    features = np.vstack(feature_stack)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    distances = np.linalg.norm((features - target_feature[None, :]) / scale[None, :], axis=1)
    order = np.argsort(distances)
    selected = [rows[i][0] for i in order[: args.num_frames]]

    selected_samples_for_moments = []
    for frame_id in selected:
        pts = crop_range(load_points(dataset_root / "pcd" / f"{frame_id}.npy"))
        selected_samples_for_moments.append(sample_points(pts, min(args.candidate_sample_points, len(pts)), rng))
    selected_samples_for_moments = np.concatenate(selected_samples_for_moments, axis=0)
    src_mu, src_cov = moments(selected_samples_for_moments)

    for out_idx, frame_id in enumerate(selected, start=1):
        pts = crop_range(load_points(dataset_root / "pcd" / f"{frame_id}.npy"))
        if args.affine_match_target:
            pts = affine_match(pts, src_mu, src_cov, target_mu, target_cov)
            pts = crop_range(pts)
        pts = sample_points(pts, args.points_per_frame, rng)
        pts.astype(np.float32).tofile(synthetic_dir / f"{out_idx:04d}.bin")

    (submission_dir / "predictions.json").write_bytes(Path(args.predictions_json).read_bytes())
    declaration_text = (
        "I confirm that I did not train on any frame listed in forbidden_frames.txt, "
        "and that all submitted detections were produced by a model trained only on synthetic data I generated.\n"
        f"Team: {args.team_name}\n"
    )
    (submission_dir / "declaration.txt").write_text(declaration_text, encoding="utf-8")

    with zipfile.ZipFile(output_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in sorted(synthetic_dir.glob("*.bin")):
            zf.write(src, f"synthetic/{src.name}")
        zf.write(submission_dir / "predictions.json", "predictions.json")
        zf.write(submission_dir / "declaration.txt", "declaration.txt")

    manifest = {
        "target_mode": args.target_mode,
        "affine_match_target": args.affine_match_target,
        "seed": args.seed,
        "num_frames": args.num_frames,
        "points_per_frame": args.points_per_frame,
        "selected_frame_ids": selected,
        "selected_distance_min": float(distances[order[0]]),
        "selected_distance_median": float(np.median(distances[order[: args.num_frames]])),
        "selected_distance_max": float(distances[order[args.num_frames - 1]]),
        "target_mean": target_mu.tolist(),
        "source_selected_mean": src_mu.tolist(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir / 'submission.zip'}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
