"""Submission assembly and validation primitives."""

from __future__ import annotations

import zipfile
from pathlib import Path

from .constants import DECLARATION_TEMPLATE, SUBMISSION_FRAME_COUNT
from .io import write_json


def write_declaration(path: str | Path, team_name: str) -> None:
    Path(path).write_text(DECLARATION_TEMPLATE.format(team_name=team_name), encoding="utf-8")


def create_submission_zip(output_zip: str | Path, synthetic_dir: str | Path, predictions: str | Path, declaration: str | Path) -> None:
    """Build a challenge ZIP after checking its required synthetic-frame count."""
    synthetic_dir = Path(synthetic_dir)
    frames = sorted(synthetic_dir.glob("*.bin"))
    if len(frames) != SUBMISSION_FRAME_COUNT:
        raise ValueError(f"expected {SUBMISSION_FRAME_COUNT} synthetic bins, got {len(frames)}")
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for frame in frames:
            archive.write(frame, f"synthetic/{frame.name}")
        archive.write(predictions, "predictions.json")
        archive.write(declaration, "declaration.txt")
