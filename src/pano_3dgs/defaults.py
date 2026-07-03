from __future__ import annotations

from pathlib import Path


DEFAULT_SAM3_REPO = Path(__file__).resolve().parents[2] / "third_party" / "sam3"

DYNAMIC_PROMPTS = [
    "person",
    "people",
    "camera",
    "tripod",
    "selfie stick",
    "phone",
]
