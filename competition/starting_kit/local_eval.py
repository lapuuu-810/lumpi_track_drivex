#!/usr/bin/env python3
"""
Run the server scoring program against a participant submission, locally.

Usage:
    python local_eval.py path/to/submission.zip [reference_data_dir]

If reference_data_dir is omitted, ./reference_data_dummy is used. To get a
meaningful realism score locally you should populate the reference dir's
realism_reference/ with 50 real frames you held out yourself (frames NOT
in forbidden_frames.txt).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).parent.resolve()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    zip_path = Path(sys.argv[1]).resolve()
    if not zip_path.exists():
        print(f"submission not found: {zip_path}", file=sys.stderr)
        return 2
    ref_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else (HERE / "reference_data_dummy")
    if not ref_dir.exists():
        print(f"reference data not found: {ref_dir}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        res = td / "input" / "res"
        ref = td / "input" / "ref"
        out = td / "output"
        res.mkdir(parents=True)
        ref.mkdir(parents=True)
        out.mkdir(parents=True)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(res)

        for item in ref_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, ref / item.name)
            else:
                shutil.copy2(item, ref / item.name)

        env = os.environ.copy()
        env["CODABENCH_REF"] = str(ref)
        env["CODABENCH_RES"] = str(res)
        env["CODABENCH_OUT"] = str(out)

        scoring_dir = HERE / "scoring_program"
        rc = subprocess.call(
            [sys.executable, str(scoring_dir / "score.py")],
            env=env,
            cwd=str(scoring_dir),
        )
        if rc != 0:
            return rc

        for f in ("scores.json", "detailed_results.html"):
            src = out / f
            if src.exists():
                shutil.copy2(src, HERE / f)
                print(f"wrote {HERE / f}")
        scores = (HERE / "scores.json").read_text()
        print("\n" + scores)
    return 0


if __name__ == "__main__":
    sys.exit(main())
