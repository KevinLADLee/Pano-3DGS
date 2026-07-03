from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAM3_MODEL = PROJECT_ROOT / "models" / "facebook" / "sam3"
DEFAULT_SAM3_REPO = PROJECT_ROOT / "third_party" / "sam3"

DYNAMIC_PROMPTS = [
    "person",
    "people",
    "camera",
    "tripod",
    "selfie stick",
    "phone",
]
