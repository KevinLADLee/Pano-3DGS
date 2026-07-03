from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_COLMAP = "/home/invs/repos/colmap_prebuild/bin/colmap"
DEFAULT_SAM3_REPO = Path(__file__).resolve().parents[2] / "third_party" / "sam3"
DEFAULT_FACES = ["front", "right", "back", "left", "top", "bottom"]

FACE_AXES = {
    "front": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
    "right": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    "back": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),
    "left": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
    "top": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
    "bottom": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
}

DYNAMIC_PROMPTS = [
    "person",
    "people",
    "camera",
    "tripod",
    "selfie stick",
    "phone",
]
