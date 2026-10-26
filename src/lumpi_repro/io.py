"""I/O helpers with validation for challenge data formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: str | Path) -> Any:
    """Read UTF-8 JSON from *path*."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any, *, indent: int | None = None) -> None:
    """Write UTF-8 JSON, creating the parent directory when needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=indent), encoding="utf-8")


def resolve_synthetic_dir(path: str | Path) -> Path:
    """Accept either a submission directory or its ``synthetic`` child."""
    candidate = Path(path)
    return candidate / "synthetic" if (candidate / "synthetic").is_dir() else candidate


def load_points(path: str | Path, *, bin_dim: int | None = None) -> np.ndarray:
    """Load an NPY point cloud or float32 BIN cloud and return contiguous XYZ."""
    source = Path(path)
    if source.suffix == ".npy":
        points = np.load(source)
    else:
        raw = np.fromfile(source, dtype=np.float32)
        if bin_dim is not None:
            if raw.size % bin_dim:
                raise ValueError(f"{source} cannot be reshaped to N x {bin_dim}")
            points = raw.reshape(-1, bin_dim)
        elif raw.size % 3 == 0:
            points = raw.reshape(-1, 3)
        elif raw.size % 4 == 0:
            points = raw.reshape(-1, 4)
        else:
            raise ValueError(f"cannot infer point dimension for {source}")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{source} must contain an N x 3 (or wider) point array")
    return np.ascontiguousarray(points[:, :3], dtype=np.float32)
