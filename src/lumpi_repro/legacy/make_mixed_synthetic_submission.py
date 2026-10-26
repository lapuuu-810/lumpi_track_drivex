#!/usr/bin/env python3
"""Mix synthetic point clouds from existing submissions and rebuild a challenge zip."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-a", required=True, help="Directory containing synthetic/*.bin or the synthetic dir itself")
    parser.add_argument("--source-b", required=True, help="Directory containing synthetic/*.bin or the synthetic dir itself")
    parser.add_argument("--ratio-a", type=float, required=True)
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--points-per-frame", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--team-name", default="mixed_synthetic")
    return parser.parse_args()


def synthetic_dir(path: Path) -> Path:
    if (path / "synthetic").is_dir():
        return path / "synthetic"
    return path


def load_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 3 == 0:
        return raw.reshape(-1, 3)
    if raw.size % 4 == 0:
        return raw.reshape(-1, 4)[:, :3]
    raise ValueError(f"cannot infer point dimension for {path}")


def sample(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    replace = len(points) < n
    idx = rng.choice(len(points), size=n, replace=replace)
    return points[idx, :3].astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.ratio_a <= 1.0:
        raise ValueError("--ratio-a must be in [0, 1]")

    rng = np.random.default_rng(args.seed)
    source_a = synthetic_dir(Path(args.source_a))
    source_b = synthetic_dir(Path(args.source_b))
    output_dir = Path(args.output_dir)
    submission_dir = output_dir / "submission"
    out_synthetic = submission_dir / "synthetic"
    out_synthetic.mkdir(parents=True, exist_ok=True)

    files_a = sorted(source_a.glob("*.bin"))
    files_b = sorted(source_b.glob("*.bin"))
    if len(files_a) != 50 or len(files_b) != 50:
        raise ValueError(f"expected 50 bins in each source, got {len(files_a)} and {len(files_b)}")

    n_a = int(round(args.points_per_frame * args.ratio_a))
    n_b = args.points_per_frame - n_a
    for idx, (path_a, path_b) in enumerate(zip(files_a, files_b), start=1):
        pts_a = sample(load_bin(path_a), n_a, rng)
        pts_b = sample(load_bin(path_b), n_b, rng)
        pts = np.concatenate([pts_a, pts_b], axis=0)
        rng.shuffle(pts, axis=0)
        pts.astype(np.float32).tofile(out_synthetic / f"{idx:04d}.bin")

    (submission_dir / "predictions.json").write_bytes(Path(args.predictions_json).read_bytes())
    declaration_text = (
        "I confirm that I did not train on any frame listed in forbidden_frames.txt, "
        "and that all submitted detections were produced by a model trained only on synthetic data I generated.\n"
        f"Team: {args.team_name}\n"
    )
    (submission_dir / "declaration.txt").write_text(declaration_text, encoding="utf-8")

    with zipfile.ZipFile(output_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in sorted(out_synthetic.glob("*.bin")):
            zf.write(src, f"synthetic/{src.name}")
        zf.write(submission_dir / "predictions.json", "predictions.json")
        zf.write(submission_dir / "declaration.txt", "declaration.txt")

    manifest = {
        "source_a": str(source_a),
        "source_b": str(source_b),
        "ratio_a": args.ratio_a,
        "points_per_frame": args.points_per_frame,
        "seed": args.seed,
        "n_a": n_a,
        "n_b": n_b,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir / 'submission.zip'}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
