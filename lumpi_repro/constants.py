"""Project-wide constants."""

from __future__ import annotations

import numpy as np

CLASSES = ("Person", "Car", "Bicycle", "Motorcycle", "Bus", "Truck", "Van", "Unknown")
CLASS_SET = frozenset(CLASSES)
POINT_CLOUD_RANGE = np.array([-40.0, -40.0, -5.0, 40.0, 40.0, 5.0], dtype=np.float32)
SUBMISSION_FRAME_COUNT = 50
DEFAULT_POINTS_PER_FRAME = 50_000
DECLARATION_TEMPLATE = (
    "I confirm that I did not train on any frame listed in forbidden_frames.txt, "
    "and that all submitted detections were produced by a model trained only on "
    "synthetic data I generated.\nTeam: {team_name}\n"
)
