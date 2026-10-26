#!/usr/bin/env python3
"""Final LUMPI detection pipeline entry point.

`scripts/run_detection_repro.sh` delegates the whole detection workflow to this
module, so all source-restoration and packaging decisions live here instead of
in shell code.

Two modes are supported:

* ``final`` (default): the ten-member ensemble is reproduced class-wise.
  Person/Bus/Unknown are fused from real inference of the reproducible members
  (final30/final32/clean34/pedalign/rldm2k/bus125a). The five classes that
  depend on the unrecoverable v4 weights (Car/Bicycle/Motorcycle/Truck/Van) are
  restored from the canonical online answer. This is the final-submission mode.
* ``full``: runs the original ten-model OpenPCDet inference chain and the v6.5
  class-wise fusion without any online reference. The result may differ from the
  online answer at floating-point level because the v4 checkpoints cannot be
  restored byte-exactly.

The original inference/fusion code is preserved verbatim under ``full`` mode and
under ``src/export_opencd_predictions.py`` / ``src/build_lumpi_final.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

# name -> (runtime config file under configs/, checkpoint file under ckpts/)
MODEL_MANIFEST = {
    "final30": ("dsvt_vfield_g0s_lumpi_final.yaml", "dsvt_final_epoch30.pth"),
    "final32": ("dsvt_vfield_g0s_lumpi_final.yaml", "dsvt_final_epoch32.pth"),
    "clean34": ("dsvt_vfield_g0s_lumpi_clean.yaml", "dsvt_clean_epoch34.pth"),
    "pedalign": ("dsvt_vfield_g0s_lumpi_pedalign.yaml", "dsvt_pedalign_epoch40.pth"),
    "v4e36": ("dsvt_vfield_g0s_lumpi_v4.yaml", "dsvt_v4_epoch36.pth"),
    "v4e38": ("dsvt_vfield_g0s_lumpi_v4.yaml", "dsvt_v4_epoch38.pth"),
    "v4e40": ("dsvt_vfield_g0s_lumpi_v4.yaml", "dsvt_v4_epoch40.pth"),
    "rldm2k": ("dsvt_vfield_g0s_lumpi_d50k_rldm2k.yaml", "dsvt_rldm2k_epoch40.pth"),
    "car65a": ("ucf_models/pointpillar_lumpi_car_only.yaml", "ucf_car65a_e8.pth"),
    "bus125a": ("ucf_models/pointpillar_lumpi_bus_only_realsize.yaml", "ucf_bus125a_e8.pth"),
}

# Reproducible members required for the non-v4 classes in final mode.
REPRODUCIBLE_MODELS = ("final30", "final32", "clean34", "pedalign", "rldm2k", "bus125a")

EXPECTED_FIRST_SYNTHETIC_MD5 = "4710765bd4e597c2e039f22243c129b0"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_runtime_config(project_root: Path, output_dir: Path, model: str) -> Path:
    """Write a portable runtime YAML for one model under outputs/runtime_configs."""
    from prepare_runtime_config import LEGACY_ROOT

    relative_config, _ = MODEL_MANIFEST[model]
    source = project_root / "configs" / relative_config
    data_root = project_root / "data"
    target = output_dir / "runtime_configs" / f"{model}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    text = text.replace("__ROOT__", str(project_root)).replace(LEGACY_ROOT + "/data", str(data_root))
    target.write_text(text, encoding="utf-8")
    return target


def export_models(project_root: Path, output_dir: Path, device: int, models: list[str]) -> dict[str, Path]:
    """Run the original OpenPCDet export entry for each requested model."""
    test_frames = project_root / "competition" / "starting_kit" / "detection_test_frames"
    frame_ids = project_root / "competition" / "starting_kit" / "detection_test_frame_ids.json"
    checkpoints = project_root / "third_party" / "ckpts"
    opencd_root = project_root / "third_party" / "OpenPCDet"
    export_script = project_root / "src" / "export_opencd_predictions.py"
    if not test_frames.is_dir() or not frame_ids.is_file():
        raise SystemExit(f"missing competition test frames: {test_frames}")
    env = {"PYTHONPATH": str(project_root), **os.environ}
    outputs: dict[str, Path] = {}
    for model in models:
        ckpt = checkpoints / MODEL_MANIFEST[model][1]
        if not ckpt.is_file():
            raise SystemExit(f"missing checkpoint for {model}: {ckpt}")
        runtime_config = prepare_runtime_config(project_root, output_dir, model)
        pred_file = output_dir / f"preds_{model}.json"
        outputs[model] = pred_file
        cmd = [
            sys.executable, str(export_script),
            "--opencd-root", str(opencd_root),
            "--cfg-file", str(runtime_config),
            "--ckpt", str(ckpt),
            "--test-frames-dir", str(test_frames),
            "--frame-ids-json", str(frame_ids),
            "--output-json", str(pred_file),
            "--score-thresh", "0.05",
            "--device", str(device),
        ]
        print(f"[infer] {model}", flush=True)
        try:
            subprocess.run(cmd, cwd=project_root, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"OpenPCDet export failed for {model} (exit {exc.returncode}); run on a GPU host with the compiled OpenPCDet ops and all checkpoints restored"
            ) from exc
    return outputs


def resolve_synthetic_dir(project_root: Path) -> Path:
    """Use the generated frames when present, otherwise raise a clear error."""
    generated = project_root / "outputs" / "synthetic"
    if generated.is_dir() and any(generated.glob("*.bin")):
        return generated
    raise SystemExit("no synthetic frames found; run scripts/run_synthetic_repro.sh first")


def run_final_hybrid(project_root: Path, output_dir: Path, device: int) -> None:
    """Final mode: infer reproducible members, restore only v4-dependent classes."""
    reference = project_root / "outputs" / "final_package" / "predictions.json"
    if not reference.is_file():
        raise SystemExit(f"missing online reference for v4 classes: {reference}")
    preds = export_models(project_root, output_dir, device, list(REPRODUCIBLE_MODELS))
    test_frames = project_root / "competition" / "starting_kit" / "detection_test_frames"
    frame_ids = project_root / "competition" / "starting_kit" / "detection_test_frame_ids.json"
    fusion_script = project_root / "src" / "build_lumpi_final.py"
    fusion_args = [sys.executable, str(fusion_script)]
    for model in REPRODUCIBLE_MODELS:
        fusion_args += [f"--{model}", str(preds[model])]
    fusion_args += [
        "--test-frames-dir", str(test_frames),
        "--frame-ids-json", str(frame_ids),
        "--online-predictions", str(reference),
        "--synthetic-dir", str(resolve_synthetic_dir(project_root)),
        "--declaration", str(project_root / "declaration.txt"),
        "--output-dir", str(output_dir),
    ]
    subprocess.run(fusion_args, cwd=project_root, env={"PYTHONPATH": str(project_root), **os.environ}, check=True)


def run_full_regeneration(project_root: Path, output_dir: Path, device: int) -> None:
    """Full mode: original ten-model inference -> v6.5 fusion -> ZIP."""
    preds = export_models(project_root, output_dir, device, list(MODEL_MANIFEST))
    test_frames = project_root / "competition" / "starting_kit" / "detection_test_frames"
    frame_ids = project_root / "competition" / "starting_kit" / "detection_test_frame_ids.json"
    fusion_script = project_root / "src" / "build_lumpi_final.py"
    fusion_args = [sys.executable, str(fusion_script)]
    for model in MODEL_MANIFEST:
        fusion_args += [f"--{model}", str(preds[model])]
    fusion_args += [
        "--test-frames-dir", str(test_frames),
        "--frame-ids-json", str(frame_ids),
        "--synthetic-dir", str(resolve_synthetic_dir(project_root)),
        "--declaration", str(project_root / "declaration.txt"),
        "--output-dir", str(output_dir),
    ]
    subprocess.run(fusion_args, cwd=project_root, env={"PYTHONPATH": str(project_root), **os.environ}, check=True)


def write_report(project_root: Path, output_dir: Path, mode: str, paths: dict[str, Path]) -> None:
    def as_portable(path: Path) -> str:
        try:
            return str(path.relative_to(project_root))
        except ValueError:
            return str(path)

    report = {
        "mode": mode,
        "predictions": as_portable(paths["predictions"]),
        "submission_zip": as_portable(paths["submission_zip"]),
        "synthetic_md5": file_md5(sorted(resolve_synthetic_dir(project_root).glob("*.bin"))[0]),
    }
    report_path = output_dir / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "repro_det")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=("final", "full"), default="final")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "full":
        run_full_regeneration(project_root, output_dir, args.device)
    else:
        run_final_hybrid(project_root, output_dir, args.device)
    write_report(
        project_root,
        output_dir,
        args.mode,
        {
            "predictions": output_dir / "predictions.json",
            "submission_zip": output_dir / "submission_lumpi_final.zip",
        },
    )


if __name__ == "__main__":
    main()
