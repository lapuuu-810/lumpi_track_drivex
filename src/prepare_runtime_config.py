#!/usr/bin/env python3
"""Create a portable OpenPCDet config without mutating versioned YAML files."""

from __future__ import annotations

import argparse
from pathlib import Path


LEGACY_ROOT = "/data2/RS3D/LPP/urbantwin_lumpi"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--data-root", help="Defaults to <project-root>/data")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else project_root / "data"
    text = Path(args.config).read_text(encoding="utf-8")
    text = text.replace("__ROOT__", str(project_root)).replace(LEGACY_ROOT + "/data", str(data_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
