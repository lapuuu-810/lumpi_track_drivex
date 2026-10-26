#!/usr/bin/env python3
"""Final LUMPI synthetic-frame pipeline entry point.

`scripts/run_synthetic_repro.sh` delegates the synthetic workflow to this
module. The final frames reproduce the online submission through the exact
recipe:

  1. generate 16b (r_hist p0.75, 4 iters) and 16c (r_hist p0.5, 5 iters)
  2. mix 16c(75%) + 16b(25%) into the 19D base (seed 1234, 50k points/frame)
  3. apply the online "v4_frac0.5" affine calibration in float32

Every generated frame is byte-verified against the bundled online reference.
The bundled reference is only a validation target here; --use-bundled is an
explicit dry-run switch for environments without the raw sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_FIRST_SYNTHETIC_MD5 = "4710765bd4e597c2e039f22243c129b0"
FRAME_COUNT = 50

# Exact online affine calibration (realism_affine_v4_frac0.5), float32 values.
AFFINE_SCALE = np.array(
    [0.9949994087219238, 1.008866786956787, 1.0234358310699463], dtype=np.float32
)
AFFINE_SHIFT = np.array(
    [0.30948615074157715, 0.18277335166931152, 0.07691717147827148], dtype=np.float32
)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_xyz(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 3 == 0:
        return raw.reshape(-1, 3)
    return raw.reshape(-1, 4)[:, :3]


def check_output(synthetic_dir: Path, reference_dir: Path | None = None) -> None:
    frames = sorted(synthetic_dir.glob("*.bin"))
    if len(frames) != FRAME_COUNT:
        raise SystemExit(f"expected {FRAME_COUNT} synthetic frames, got {len(frames)}")
    first_md5 = file_md5(frames[0])
    print(f"synthetic frame1 md5: {first_md5}")
    if first_md5 != EXPECTED_FIRST_SYNTHETIC_MD5:
        raise SystemExit("synthetic frame md5 does not match the submitted set")
    if reference_dir is not None and reference_dir.is_dir():
        reference = {p.name: file_md5(p) for p in sorted(reference_dir.glob("*.bin"))}
        if len(reference) != FRAME_COUNT:
            raise SystemExit("reference synthetic set is incomplete")
        mismatched = [p.name for p in frames if file_md5(p) != reference.get(p.name)]
        if mismatched:
            raise SystemExit(
                f"{len(mismatched)} synthetic frames differ from the online submission: {mismatched[:5]}"
            )
        print("synthetic byte-compare vs online reference: 50/50 identical")
    print(f"synthetic output: {synthetic_dir} ({len(frames)} frames)")


def apply_online_affine(source_dir: Path, output_dir: Path) -> None:
    """Apply the online v4_frac0.5 affine calibration with float32 arithmetic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.glob("*.bin")):
        points = load_xyz(source)
        calibrated = np.ascontiguousarray(points * AFFINE_SCALE + AFFINE_SHIFT)
        calibrated.tofile(output_dir / source.name)


def run_from_raw_sources(
    project_root: Path,
    ucf_dt_root: Path,
    m5_dir: Path,
    starting_kit: Path,
    output_dir: Path,
) -> None:
    """Run 16b + 16c generation, the 19D mix, and the online affine calibration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids_json = starting_kit / "detection_test_frame_ids.json"
    dummy_predictions = output_dir / "dummy_predictions.json"
    frame_ids = json.loads(frame_ids_json.read_text(encoding="utf-8"))
    dummy_predictions.write_text(json.dumps({frame: [] for frame in frame_ids}), encoding="utf-8")
    env = {"PYTHONPATH": str(project_root)}

    def run_script(script_name: str, *args: str) -> None:
        command = [sys.executable, str(project_root / "src" / script_name), *args]
        subprocess.run(command, cwd=project_root, env=env, check=True)

    base = [
        "--starting-kit", str(starting_kit),
        "--predictions-json", str(dummy_predictions),
        "--points-per-frame", "50000",
        "--target-num-frames", "120",
        "--target-sample-points", "2500",
        "--candidate-sample-points", "2500",
        "--seed", "1234",
    ]
    print("[1/4] generate 16b (r_hist p0.75, 4 iterations)")
    run_script(
        "make_lumpi_realstat_submission.py",
        "--dataset-root", str(ucf_dt_root),
        "--lumpi-measurement-dir", str(m5_dir),
        "--output-dir", str(output_dir / "lumpi16b"),
        "--num-frames", "50",
        "--post-crop-match-iters", "4",
        "--final-resample-mode", "r_hist",
        "--hist-power", "0.75",
        "--hist-max-weight", "8.0",
        "--team-name", "repro_16b",
        *base,
    )
    print("[2/4] generate 16c (r_hist p0.5, 5 iterations)")
    run_script(
        "make_lumpi_realstat_submission.py",
        "--dataset-root", str(ucf_dt_root),
        "--lumpi-measurement-dir", str(m5_dir),
        "--output-dir", str(output_dir / "lumpi16c"),
        "--num-frames", "50",
        "--post-crop-match-iters", "5",
        "--final-resample-mode", "r_hist",
        "--hist-power", "0.5",
        "--hist-max-weight", "8.0",
        "--team-name", "repro_16c",
        *base,
    )
    print("[3/4] mix 16c(75%) + 16b(25%) -> 19D base")
    run_script(
        "make_mixed_synthetic_submission.py",
        "--source-a", str(output_dir / "lumpi16c" / "submission" / "synthetic"),
        "--source-b", str(output_dir / "lumpi16b" / "submission" / "synthetic"),
        "--ratio-a", "0.75",
        "--predictions-json", str(dummy_predictions),
        "--output-dir", str(output_dir / "lumpi19d"),
        "--points-per-frame", "50000",
        "--seed", "1234",
        "--team-name", "repro_19d",
    )
    generated = output_dir / "lumpi19d" / "submission" / "synthetic"
    if not generated.is_dir():
        generated = output_dir / "lumpi19d" / "synthetic"
    print("[4/4] apply online v4_frac0.5 affine calibration (float32)")
    apply_online_affine(generated, output_dir / "synthetic")
    check_output(output_dir / "synthetic", project_root / "outputs" / "final_package" / "synthetic")


def use_bundled_set(project_root: Path, output_dir: Path) -> None:
    """Dry-run validation only: copy the bundled canonical synthetic frames."""
    reference = project_root / "outputs" / "final_package" / "synthetic"
    if not reference.is_dir() or not any(reference.glob("*.bin")):
        raise SystemExit(f"missing bundled synthetic reference: {reference}")
    final_dir = output_dir / "synthetic"
    final_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(reference.glob("*.bin")):
        shutil.copyfile(source, final_dir / source.name)
    check_output(final_dir, reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--ucf-dt-root", type=Path, default=None)
    parser.add_argument("--m5-dir", type=Path, default=None)
    parser.add_argument("--starting-kit", type=Path, default=None)
    parser.add_argument("--use-bundled", action="store_true",
                        help="explicit dry-run: copy the bundled canonical frames instead of generating")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    starting_kit = (args.starting_kit or project_root / "competition" / "starting_kit").resolve()
    frame_ids_json = starting_kit / "detection_test_frame_ids.json"
    ucf_root = args.ucf_dt_root
    m5_dir = args.m5_dir
    have_raw = (
        ucf_root is not None and ucf_root.is_dir()
        and m5_dir is not None and m5_dir.is_dir()
        and frame_ids_json.is_file()
    )
    if args.use_bundled:
        use_bundled_set(project_root, output_dir)
    elif have_raw:
        run_from_raw_sources(project_root, ucf_root.resolve(), m5_dir.resolve(), starting_kit, output_dir)
    else:
        raise SystemExit(
            "raw UCF-DT/Measurement5 sources are required for synthetic generation; "
            "pass UCF_DT_ROOT/M5_DIR (or --use-bundled for dry-run validation only)"
        )


if __name__ == "__main__":
    main()
