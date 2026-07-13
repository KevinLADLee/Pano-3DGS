from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pano_3dgs.utils import ensure_dir


@dataclass
class Candidate:
    frame_index: int
    timestamp: float
    score: float
    frame: np.ndarray


def crop_roi(gray: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    height, width = gray.shape[:2]
    x0, y0, x1, y1 = roi
    return gray[
        int(round(y0 * height)) : int(round(y1 * height)),
        int(round(x0 * width)) : int(round(x1 * width)),
    ]


def sharpness_score(
    frame: np.ndarray,
    scale_width: int,
    roi: tuple[float, float, float, float],
) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale_width > 0 and gray.shape[1] > scale_width:
        scale = scale_width / float(gray.shape[1])
        gray = cv2.resize(
            gray,
            (scale_width, max(1, int(round(gray.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    lap = cv2.Laplacian(crop_roi(gray, roi), cv2.CV_64F)
    return float(lap.var())


def extract_sharpest(args: argparse.Namespace) -> None:
    out_dir = args.run / "frames"
    ensure_dir(out_dir)
    csv_path = out_dir / "sharpest_frames.csv"

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or args.fallback_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    chunk_size = args.chunk_size or max(1, int(round(args.window_seconds * fps)))
    end_frame = total_frames
    if args.max_windows is not None:
        end_frame = min(end_frame, args.max_windows * chunk_size)

    print(
        f"{args.video.name}: fps={fps:.3f}, frames={total_frames}, "
        f"chunk={chunk_size}, selected output={out_dir}",
        flush=True,
    )

    selected = 0
    best: Candidate | None = None
    current_window = 0
    fieldnames = [
        "video",
        "output",
        "sequence_index",
        "frame_index",
        "timestamp",
        "score",
        "window_index",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        frame_index = 0
        while frame_index < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            window_index = frame_index // chunk_size
            if window_index != current_window:
                selected = _write_candidate(
                    args.video,
                    out_dir,
                    writer,
                    selected,
                    best,
                    current_window,
                    args.frame_jpg_quality,
                )
                best = None
                current_window = window_index

            timestamp = frame_index / fps
            score = sharpness_score(frame, args.scale_width, args.roi)
            if best is None or score > best.score:
                best = Candidate(frame_index, timestamp, score, frame.copy())

            frame_index += 1
            if args.progress and frame_index % args.progress == 0:
                print(f"  processed={frame_index}, selected={selected}", flush=True)

        selected = _write_candidate(
            args.video,
            out_dir,
            writer,
            selected,
            best,
            current_window,
            args.frame_jpg_quality,
        )

    cap.release()
    print(f"done: selected={selected}, csv={csv_path}", flush=True)


def _write_candidate(
    video: Path,
    out_dir: Path,
    writer: csv.DictWriter,
    selected: int,
    candidate: Candidate | None,
    window_index: int,
    jpg_quality: int,
) -> int:
    if candidate is None:
        return selected
    selected += 1
    out_name = f"{video.stem}_{selected:06d}_t{candidate.timestamp:09.3f}_f{candidate.frame_index:07d}.jpg"
    out_path = out_dir / out_name
    cv2.imwrite(str(out_path), candidate.frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    writer.writerow(
        {
            "video": str(video),
            "output": str(out_path),
            "sequence_index": selected,
            "frame_index": candidate.frame_index,
            "timestamp": f"{candidate.timestamp:.6f}",
            "score": f"{candidate.score:.6f}",
            "window_index": window_index,
        }
    )
    return selected
