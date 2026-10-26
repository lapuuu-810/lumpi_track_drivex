#!/usr/bin/env python3
"""Build a submission whose synthetic frames are matched to public LUMPI real-frame statistics."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np


RANGE = np.array([-40.0, -40.0, -5.0, 40.0, 40.0, 5.0], dtype=np.float32)

PLY_DTYPE_MAP = {
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "uchar": "u1",
    "uint8": "u1",
    "ushort": "u2",
    "uint16": "u2",
    "uint": "u4",
    "uint32": "u4",
    "int": "i4",
    "int32": "i4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="UT-LUMPI synthetic root with pcd/*.npy")
    parser.add_argument("--starting-kit", required=True)
    parser.add_argument("--lumpi-measurement-dir", required=True, help="Directory containing lidar.zip.001")
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--points-per-frame", type=int, default=50000)
    parser.add_argument("--target-num-frames", type=int, default=120)
    parser.add_argument("--target-sample-points", type=int, default=2500)
    parser.add_argument("--candidate-sample-points", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--team-name", default="baseline10_lumpi_realstat")
    parser.add_argument("--no-affine", action="store_true")
    parser.add_argument("--post-crop-match-iters", type=int, default=0)
    parser.add_argument("--final-resample-mode", choices=("uniform", "z_hist", "r_hist", "rz_hist"), default="uniform")
    parser.add_argument("--hist-power", type=float, default=0.5)
    parser.add_argument("--hist-max-weight", type=float, default=8.0)
    parser.add_argument("--hist-min-radius", type=float, default=0.0)
    parser.add_argument("--hist-max-radius", type=float, default=1e9)
    return parser.parse_args()


def load_forbidden(starting_kit: Path) -> set[str]:
    path = starting_kit / "forbidden_frames.txt"
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def load_points(path: Path, bin_dim: int | None = None) -> np.ndarray:
    if path.suffix == ".npy":
        pts = np.load(path)
    else:
        raw = np.fromfile(path, dtype=np.float32)
        if bin_dim is not None:
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


def build_hist_profile(points: np.ndarray) -> dict:
    radius = np.sqrt(points[:, 0].astype(np.float64) ** 2 + points[:, 1].astype(np.float64) ** 2)
    z_bins = np.linspace(float(RANGE[2]), float(RANGE[5]), 31)
    r_bins = np.linspace(0.0, math.sqrt(40.0**2 + 40.0**2), 31)
    z_hist = np.histogram(points[:, 2], bins=z_bins)[0].astype(np.float64)
    r_hist = np.histogram(radius, bins=r_bins)[0].astype(np.float64)
    rz_hist = np.histogram2d(radius, points[:, 2], bins=(r_bins, z_bins))[0].astype(np.float64)
    z_hist = (z_hist + 1e-6) / (z_hist.sum() + 1e-6 * len(z_hist))
    r_hist = (r_hist + 1e-6) / (r_hist.sum() + 1e-6 * len(r_hist))
    rz_hist = (rz_hist + 1e-6) / (rz_hist.sum() + 1e-6 * rz_hist.size)
    return {
        "z_bins": z_bins,
        "r_bins": r_bins,
        "z_hist": z_hist,
        "r_hist": r_hist,
        "rz_hist": rz_hist,
    }


def weighted_sample_points(
    points: np.ndarray,
    n: int,
    rng: np.random.Generator,
    profile: dict,
    mode: str,
    power: float,
    max_weight: float,
    min_radius: float,
    max_radius: float,
) -> np.ndarray:
    if mode == "uniform":
        return sample_points(points, n, rng)
    if len(points) == 0:
        raise ValueError("empty point cloud after range crop")

    radius = np.sqrt(points[:, 0].astype(np.float64) ** 2 + points[:, 1].astype(np.float64) ** 2)
    z_idx = np.clip(np.digitize(points[:, 2], profile["z_bins"]) - 1, 0, len(profile["z_hist"]) - 1)
    r_idx = np.clip(np.digitize(radius, profile["r_bins"]) - 1, 0, len(profile["r_hist"]) - 1)
    weights = np.ones(len(points), dtype=np.float64)
    active = (radius >= min_radius) & (radius <= max_radius)

    if mode in {"z_hist", "rz_hist"}:
        current = np.bincount(z_idx, minlength=len(profile["z_hist"])).astype(np.float64)
        current = (current + 1e-6) / (current.sum() + 1e-6 * len(current))
        weights[active] *= profile["z_hist"][z_idx[active]] / current[z_idx[active]]
    if mode in {"r_hist", "rz_hist"}:
        current = np.bincount(r_idx, minlength=len(profile["r_hist"])).astype(np.float64)
        current = (current + 1e-6) / (current.sum() + 1e-6 * len(current))
        weights[active] *= profile["r_hist"][r_idx[active]] / current[r_idx[active]]
    if mode == "rz_hist":
        flat_idx = r_idx * len(profile["z_hist"]) + z_idx
        current = np.bincount(flat_idx, minlength=profile["rz_hist"].size).astype(np.float64).reshape(profile["rz_hist"].shape)
        current = (current + 1e-6) / (current.sum() + 1e-6 * current.size)
        weights[active] *= profile["rz_hist"][r_idx[active], z_idx[active]] / current[r_idx[active], z_idx[active]]

    weights = np.power(np.clip(weights, 1.0 / max_weight, max_weight), power)
    weights = weights / weights.sum()
    replace = len(points) < n
    idx = rng.choice(len(points), size=n, replace=replace, p=weights)
    return np.ascontiguousarray(points[idx], dtype=np.float32)


def parse_ply_bytes(data: bytes) -> np.ndarray:
    marker = b"end_header\n"
    header_end = data.find(marker)
    if header_end < 0:
        marker = b"end_header\r\n"
        header_end = data.find(marker)
    if header_end < 0:
        raise ValueError("missing PLY end_header")
    body_offset = header_end + len(marker)
    header = data[:body_offset].decode("ascii").splitlines()
    if header[0] != "ply" or "binary_little_endian" not in header[1]:
        raise ValueError(f"unsupported PLY header: {header[:2]}")

    vertex_count = None
    fields: list[tuple[str, str]] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
            continue
        if parts and parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
        if in_vertex and parts[:1] == ["property"]:
            fields.append((parts[2], PLY_DTYPE_MAP[parts[1]]))
    if vertex_count is None:
        raise ValueError("missing vertex count")
    arr = np.frombuffer(data, dtype=np.dtype(fields).newbyteorder("<"), count=vertex_count, offset=body_offset)
    return np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)


def list_ply_frames(measurement_dir: Path) -> list[str]:
    result = subprocess.run(
        ["7z", "l", "lidar.zip.001"],
        cwd=measurement_dir,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    frame_ids = []
    pat = re.compile(r"lidar/([0-9]{6})\.ply$")
    for line in result.stdout.splitlines():
        m = pat.search(line)
        if m:
            frame_ids.append(m.group(1))
    if not frame_ids:
        raise ValueError(f"no PLY frames found in {measurement_dir}")
    return sorted(set(frame_ids))


def load_ply_from_archive(measurement_dir: Path, frame_id: str) -> np.ndarray:
    rel = f"lidar/{frame_id}.ply"
    result = subprocess.run(
        ["7z", "e", "-so", "lidar.zip.001", rel],
        cwd=measurement_dir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return parse_ply_bytes(result.stdout)


def feature_from_points(points: np.ndarray) -> np.ndarray:
    xyz = points[:, :3].astype(np.float64)
    mean = xyz.mean(axis=0)
    cov = np.cov(xyz, rowvar=False)
    cov_vals = np.array([cov[0, 0], cov[1, 1], cov[2, 2], cov[0, 1], cov[0, 2], cov[1, 2]], dtype=np.float64)
    radius = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    z_hist, _ = np.histogram(xyz[:, 2], bins=np.linspace(RANGE[2], RANGE[5], 11), density=True)
    r_hist, _ = np.histogram(radius, bins=np.linspace(0.0, math.sqrt(40.0**2 + 40.0**2), 11), density=True)
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


def load_lumpi_target_samples(args: argparse.Namespace, forbidden: set[str], rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    measurement_dir = Path(args.lumpi_measurement_dir)
    frame_ids = [x for x in list_ply_frames(measurement_dir) if x not in forbidden]
    if len(frame_ids) < args.target_num_frames:
        raise ValueError(f"not enough non-forbidden LUMPI target frames: {len(frame_ids)}")
    pick_idx = np.linspace(0, len(frame_ids) - 1, args.target_num_frames, dtype=int)
    selected = [frame_ids[i] for i in pick_idx]
    samples = []
    for frame_id in selected:
        pts = crop_range(load_ply_from_archive(measurement_dir, frame_id))
        samples.append(sample_points(pts, args.target_sample_points, rng))
    return np.concatenate(samples, axis=0), selected


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
    target_samples, target_frame_ids = load_lumpi_target_samples(args, forbidden, rng)
    target_feature = feature_from_points(target_samples)
    target_mu, target_cov = moments(target_samples)
    target_hist_profile = build_hist_profile(target_samples)

    candidates = [p.stem for p in sorted((dataset_root / "pcd").glob("*.npy")) if p.stem not in forbidden]
    if len(candidates) < args.num_frames:
        raise ValueError(f"not enough synthetic candidates: {len(candidates)}")

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

    selected_samples = []
    for frame_id in selected:
        pts = crop_range(load_points(dataset_root / "pcd" / f"{frame_id}.npy"))
        selected_samples.append(sample_points(pts, min(args.candidate_sample_points, len(pts)), rng))
    selected_samples = np.concatenate(selected_samples, axis=0)
    src_mu, src_cov = moments(selected_samples)

    selected_clouds = []
    for frame_id in selected:
        pts = crop_range(load_points(dataset_root / "pcd" / f"{frame_id}.npy"))
        if not args.no_affine:
            pts = affine_match(pts, src_mu, src_cov, target_mu, target_cov)
            pts = crop_range(pts)
        selected_clouds.append(pts)

    post_crop_stats = []
    for _ in range(args.post_crop_match_iters):
        current_samples = []
        for pts in selected_clouds:
            current_samples.append(sample_points(pts, min(args.candidate_sample_points, len(pts)), rng))
        current_samples = np.concatenate(current_samples, axis=0)
        cur_mu, cur_cov = moments(current_samples)
        post_crop_stats.append(
            {
                "mean": cur_mu.tolist(),
                "cov_diag": np.diag(cur_cov).tolist(),
            }
        )
        selected_clouds = [crop_range(affine_match(pts, cur_mu, cur_cov, target_mu, target_cov)) for pts in selected_clouds]

    for out_idx, pts in enumerate(selected_clouds, start=1):
        pts = weighted_sample_points(
            pts,
            args.points_per_frame,
            rng,
            target_hist_profile,
            args.final_resample_mode,
            args.hist_power,
            args.hist_max_weight,
            args.hist_min_radius,
            args.hist_max_radius,
        )
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
        "target_source": str(Path(args.lumpi_measurement_dir)),
        "target_num_frames": args.target_num_frames,
        "target_frame_ids_first_last": [target_frame_ids[0], target_frame_ids[-1]],
        "target_frame_ids": target_frame_ids,
        "affine": not args.no_affine,
        "seed": args.seed,
        "num_frames": args.num_frames,
        "points_per_frame": args.points_per_frame,
        "selected_synthetic_frame_ids": selected,
        "selected_distance_min": float(distances[order[0]]),
        "selected_distance_median": float(np.median(distances[order[: args.num_frames]])),
        "selected_distance_max": float(distances[order[args.num_frames - 1]]),
        "target_mean": target_mu.tolist(),
        "target_cov_diag": np.diag(target_cov).tolist(),
        "source_selected_mean": src_mu.tolist(),
        "source_selected_cov_diag": np.diag(src_cov).tolist(),
        "post_crop_match_iters": args.post_crop_match_iters,
        "post_crop_stats_before_each_iter": post_crop_stats,
        "final_resample_mode": args.final_resample_mode,
        "hist_power": args.hist_power,
        "hist_max_weight": args.hist_max_weight,
        "hist_min_radius": args.hist_min_radius,
        "hist_max_radius": args.hist_max_radius,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir / 'submission.zip'}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
