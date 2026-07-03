from __future__ import annotations

import argparse
import re
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_box(value: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise argparse.ArgumentTypeError("box coordinates must be normalized to [0,1]")
    return x0, y0, x1, y1


def parse_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return [part.strip() for part in value.split(",") if part.strip()]


def default_scene_name(video: Path) -> str:
    scene = re.sub(r"[^A-Za-z0-9_.-]+", "_", video.stem).strip("_.-")
    return scene or "scene"


def scene_run_dir(runs_dir: Path, scene: str, rate_hz: float, width: int) -> Path:
    rate = f"{rate_hz:g}hz".replace(".", "p")
    return runs_dir / f"{scene}_{rate}_{width}"
