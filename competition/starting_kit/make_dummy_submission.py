#!/usr/bin/env python3
"""
Produce a syntactically valid (but quality-poor) submission, useful only
for verifying that your packaging pipeline produces something the local
evaluator and the Codabench worker accept.

Output: dummy_submission.zip in the current directory.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
N_FRAMES = 50
N_POINTS = 1024  # well under the 125k cap


def main() -> None:
    ids_file = HERE / "detection_test_frame_ids.json"
    if not ids_file.exists():
        raise SystemExit(f"missing {ids_file} — re-download the starting kit")
    test_ids = json.loads(ids_file.read_text())

    rng = np.random.default_rng(0)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "synthetic").mkdir()
        for i in range(1, N_FRAMES + 1):
            pts = rng.normal(0, 10, size=(N_POINTS, 3)).astype(np.float32)
            pts.tofile(td / "synthetic" / f"{i:04d}.bin")

        # Empty predictions for every required frame ID -> mAP = 0.
        # Multi-class predictions take this shape:
        #   {"frame_id": [{"box": [...], "score": 0.97, "class": "Car"}, ...]}
        preds = {fid: [] for fid in test_ids}
        (td / "predictions.json").write_text(json.dumps(preds))

        decl = (
            "I confirm that I did not train on any frame listed in "
            "forbidden_frames.txt, and that all submitted detections were "
            "produced by a model trained only on synthetic data I generated.\n"
            "Team: dummy\nDate: 1970-01-01\n"
        )
        (td / "declaration.txt").write_text(decl)

        out = HERE / "dummy_submission.zip"
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(td.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(td))
        print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
