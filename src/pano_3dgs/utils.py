from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil


_GIB = 1024**3


@dataclass(frozen=True)
class WorkerChoice:
    workers: int
    reason: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy_file(source: Path, destination: Path) -> str:
    ensure_dir(destination.parent)
    if destination.exists():
        try:
            if source.samefile(destination):
                return "existing"
        except OSError:
            pass
        source_stat = source.stat()
        destination_stat = destination.stat()
        if (
            source_stat.st_size == destination_stat.st_size
            and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
        ):
            return "existing"
    if sys.platform == "win32":
        shutil.copy2(source, destination)
        return "copied"
    if destination.exists():
        destination.unlink()
    os.link(source, destination)
    return "linked"


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


def choose_worker_count(
    requested_workers: int,
    item_count: int,
    *,
    estimated_per_worker_bytes: int,
    shared_memory_bytes: int = 0,
    max_auto_workers: int = 32,
    windows_max_auto_workers: int = 4,
    memory_fraction: float = 0.65,
    min_available_memory_bytes: int = _GIB,
) -> WorkerChoice:
    if item_count <= 0:
        return WorkerChoice(1, "no work items")

    if requested_workers > 0:
        workers = max(1, min(requested_workers, item_count))
        return WorkerChoice(workers, f"manual request={requested_workers}")

    logical_cpus = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    cpu_limit = max(1, logical_cpus - 1)
    platform_limit = windows_max_auto_workers if sys.platform == "win32" else max_auto_workers
    cpu_workers = max(1, min(cpu_limit, platform_limit, max_auto_workers, item_count))

    available_memory = psutil.virtual_memory().available
    usable_memory = max(0, int((available_memory - min_available_memory_bytes - shared_memory_bytes) * memory_fraction))
    if estimated_per_worker_bytes > 0:
        memory_workers = max(1, usable_memory // estimated_per_worker_bytes)
    else:
        memory_workers = cpu_workers
    memory_workers = max(1, min(memory_workers, item_count))

    workers = max(1, min(cpu_workers, memory_workers))
    reason = (
        f"auto cpu={logical_cpus}, cpu_limit={cpu_workers}, "
        f"available_mem={available_memory / _GIB:.1f}GiB, "
        f"estimated_worker_mem={estimated_per_worker_bytes / _GIB:.1f}GiB, "
        f"memory_limit={memory_workers}"
    )
    return WorkerChoice(workers, reason)
