from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_COLMAP = "/home/invs/repos/colmap_prebuild/bin/colmap"
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


def load_dotenv(start: Path | None = None) -> None:
    """Load simple KEY=VALUE entries from the nearest .env file."""
    current = (start or Path.cwd()).resolve()
    candidates = [current] + list(current.parents)
    for directory in candidates:
        env_path = directory / ".env"
        if not env_path.exists():
            continue
        with env_path.open() as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
        return


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_path(name: str, default: str | Path | None = None) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value)
    if default is None:
        return None
    return Path(default)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_bool_or_none(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    if value.lower() in {"", "auto", "none"}:
        return None
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class Candidate:
    frame_index: int
    timestamp: float
    score: float
    frame: np.ndarray


@dataclass
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def run_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


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


def parse_faces(value: str) -> list[str]:
    if value == "all":
        return ["front", "right", "back", "left", "top", "bottom"]
    faces = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [face for face in faces if face not in FACE_AXES]
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown face(s): {', '.join(invalid)}")
    return faces


def parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def scene_run_dir(runs_dir: Path, scene: str, rate_hz: float, width: int) -> Path:
    rate = f"{rate_hz:g}hz".replace(".", "p")
    return runs_dir / f"{scene}_{rate}_{width}"


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
    fieldnames = ["video", "output", "sequence_index", "frame_index", "timestamp", "score", "window_index"]

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
                selected = _write_candidate(args.video, out_dir, writer, selected, best, current_window, args.frame_jpg_quality)
                best = None
                current_window = window_index

            timestamp = frame_index / fps
            score = sharpness_score(frame, args.scale_width, args.roi)
            if best is None or score > best.score:
                best = Candidate(frame_index, timestamp, score, frame.copy())

            frame_index += 1
            if args.progress and frame_index % args.progress == 0:
                print(f"  processed={frame_index}, selected={selected}", flush=True)

        selected = _write_candidate(args.video, out_dir, writer, selected, best, current_window, args.frame_jpg_quality)

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


def load_official_sam3(args: argparse.Namespace):
    import torch

    if args.sam3_repo:
        sys.path.insert(0, str(args.sam3_repo))
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise SystemExit(
            "Official SAM3 backend could not be imported. "
            "Install facebookresearch/sam3 dependencies first. "
            f"Original import error: {exc}"
        ) from exc

    checkpoint = Path(args.sam3_model)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "sam3.pt"
    if not checkpoint.exists():
        raise SystemExit(f"SAM3 checkpoint not found: {checkpoint}")

    device = args.device
    builder_device = "cuda" if device.startswith("cuda") else device
    if device.startswith("cuda:"):
        torch.cuda.set_device(int(device.split(":", 1)[1]))
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device=builder_device,
        eval_mode=True,
        enable_segmentation=True,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=args.score)
    return model, processor, torch


def mask_intersects_roi(mask: np.ndarray, rois: list[tuple[float, float, float, float]]) -> bool:
    if not rois:
        return True
    height, width = mask.shape
    for x0, y0, x1, y1 in rois:
        if np.any(mask[int(y0 * height) : int(np.ceil(y1 * height)), int(x0 * width) : int(np.ceil(x1 * width))]):
            return True
    return False


def run_sam3(args: argparse.Namespace) -> None:
    frames = args.run / "frames"
    out = args.run / "dynamic_masks"
    debug = args.run / "dynamic_mask_debug"
    ensure_dir(out)
    ensure_dir(debug)
    image_paths = sorted(frames.glob("*.jpg")) + sorted(frames.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"no frames found in {frames}")

    model, processor, torch = load_official_sam3(args)
    prompts = args.prompt or args.sam3_prompts or DYNAMIC_PROMPTS
    for idx, path in enumerate(image_paths, 1):
        keep = build_sam3_keep_mask(path, prompts, model, processor, torch, args)
        cv2.imwrite(str(out / f"{path.name}.png"), keep)
        write_debug_overlay(path, keep, debug / path.name)
        ignored = 1.0 - float((keep > 0).mean())
        print(f"[{idx}/{len(image_paths)}] {path.name}: ignored={ignored:.4f}", flush=True)


def build_sam3_keep_mask(
    image_path: Path,
    prompts: list[str],
    model,
    processor,
    torch,
    args: argparse.Namespace,
) -> np.ndarray:
    del model
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    ignored = np.zeros((height, width), np.uint8)
    min_area_px = args.min_area * width * height
    max_area_px = args.max_area * width * height
    autocast_dtype = {"auto": None, "float32": None, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    for prompt in prompts:
        if args.device.startswith("cuda") and autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                state = processor.set_image(image)
                state = processor.set_text_prompt(prompt=prompt, state=state)
        else:
            state = processor.set_image(image)
            state = processor.set_text_prompt(prompt=prompt, state=state)
        masks = state.get("masks", [])
        scores = state.get("scores", [])
        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
        if hasattr(scores, "detach"):
            scores = scores.detach().float().cpu().numpy()
        masks = np.asarray(masks)
        scores = np.asarray(scores)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        for mask, score in zip(masks, scores):
            if float(score) < args.score:
                continue
            mask = np.asarray(mask).astype(bool)
            area = int(mask.sum())
            if area < min_area_px or area > max_area_px:
                continue
            if not mask_intersects_roi(mask, args.sam3_roi):
                continue
            ignored[mask] = 255

    if args.dilate > 0 and np.any(ignored):
        ignored = cv2.dilate(ignored, np.ones((args.dilate, args.dilate), np.uint8))
    keep = np.full((height, width), 255, np.uint8)
    keep[ignored > 0] = 0
    return keep


def write_debug_overlay(image_path: Path, keep_mask: np.ndarray, out_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    overlay = image.copy()
    overlay[keep_mask == 0] = (0, 0, 255)
    cv2.imwrite(str(out_path), cv2.addWeighted(image, 0.65, overlay, 0.35, 0))


def make_colmap_masks(args: argparse.Namespace) -> None:
    frames = args.run / "frames"
    out = args.run / "colmap_masks"
    dynamic = args.dynamic_mask_dir or args.run / "dynamic_masks"
    ensure_dir(out)
    paths = sorted(frames.glob("*.jpg")) + sorted(frames.glob("*.png"))
    if not paths:
        raise SystemExit(f"no frames found in {frames}")

    progress = max(1, args.mask_progress)
    for idx, path in enumerate(paths, 1):
        out_path = out / f"{path.name}.png"
        dyn_path = dynamic / f"{path.name}.png"
        has_dynamic = dynamic.exists() and dyn_path.exists()
        use_heuristics = (not has_dynamic) if args.mask_heuristics is None else args.mask_heuristics

        if not use_heuristics and has_dynamic:
            shutil.copyfile(dyn_path, out_path)
            action = "copied dynamic mask"
            if idx == 1 or idx == len(paths) or idx % progress == 0:
                print(f"[{idx}/{len(paths)}] {path.name}: {action}", flush=True)
            continue

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"cannot read {path}")
        height, width = image.shape[:2]
        mask = np.full((height, width), 255, np.uint8)

        if use_heuristics:
            if args.sky_mask:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                bright = hsv[..., 2] > 175
                low_sat = hsv[..., 1] < 70
                top_half = np.arange(height)[:, None] < int(0.48 * height)
                mask[bright & low_sat & top_half] = 0
            if args.zenith_mask > 0:
                mask[: int(args.zenith_mask * height), :] = 0
            if args.nadir_mask > 0:
                mask[int((1.0 - args.nadir_mask) * height) :, :] = 0

        if has_dynamic:
            dyn = cv2.imread(str(dyn_path), cv2.IMREAD_GRAYSCALE)
            if dyn is None:
                raise SystemExit(f"cannot read dynamic mask: {dyn_path}")
            if dyn.shape != mask.shape:
                raise SystemExit(f"mask shape mismatch: {dyn_path}")
            mask[dyn == 0] = 0

        cv2.imwrite(str(out_path), mask)
        if has_dynamic and use_heuristics:
            action = "wrote dynamic+heuristic mask"
        elif use_heuristics:
            action = "wrote heuristic mask"
        else:
            action = "wrote white mask"
        if idx == 1 or idx == len(paths) or idx % progress == 0:
            print(f"[{idx}/{len(paths)}] {path.name}: {action}", flush=True)
    print(f"done: wrote {len(paths)} masks to {out}", flush=True)


def run_colmap(args: argparse.Namespace) -> None:
    colmap = str(args.colmap)
    run = args.run
    image_dir = run / "frames"
    mask_dir = run / "colmap_masks"
    workspace = run / "colmap_cli_shared"
    db = workspace / "database.db"
    sparse = workspace / "sparse"
    sparse_txt = workspace / "sparse_txt"
    if getattr(args, "clean_colmap", False) and workspace.exists():
        print(f"cleaning COLMAP workspace: {workspace}", flush=True)
        shutil.rmtree(workspace)
    ensure_dir(workspace)
    ensure_dir(sparse)
    ensure_dir(sparse_txt)

    run_cmd(
        [
            colmap,
            "feature_extractor",
            "--database_path",
            str(db),
            "--image_path",
            str(image_dir),
            "--ImageReader.camera_model",
            "EQUIRECTANGULAR",
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_params",
            f"{args.equirect_width},{args.equirect_height}",
            "--ImageReader.mask_path",
            str(mask_dir),
            "--FeatureExtraction.use_gpu",
            "1",
            "--FeatureExtraction.gpu_index",
            str(args.gpu_index),
            "--FeatureExtraction.num_threads",
            str(args.threads),
            "--SiftExtraction.max_num_features",
            str(args.max_features),
        ]
    )
    run_cmd(
        [
            colmap,
            "sequential_matcher",
            "--database_path",
            str(db),
            "--FeatureMatching.use_gpu",
            "1",
            "--FeatureMatching.gpu_index",
            str(args.gpu_index),
            "--FeatureMatching.guided_matching",
            "1",
            "--FeatureMatching.num_threads",
            str(args.threads),
            "--SequentialMatching.overlap",
            str(args.overlap),
            "--SequentialMatching.quadratic_overlap",
            "1",
        ]
    )
    run_cmd(
        [
            colmap,
            "mapper",
            "--database_path",
            str(db),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse),
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
            "--Mapper.ba_use_gpu",
            "1",
            "--Mapper.ba_gpu_index",
            str(args.gpu_index),
        ]
    )
    model_dir = sparse / "0"
    if not model_dir.exists():
        model_dir = sorted(p for p in sparse.iterdir() if p.is_dir())[0]
    run_cmd([colmap, "model_analyzer", "--path", str(model_dir)])
    run_cmd([colmap, "model_converter", "--input_path", str(model_dir), "--output_path", str(sparse_txt), "--output_type", "TXT"])


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * z * x + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * z * x - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def rotmat_to_qvec(rotmat: np.ndarray) -> np.ndarray:
    m = rotmat
    k = np.array(
        [
            [m[0, 0] - m[1, 1] - m[2, 2], 0, 0, 0],
            [m[1, 0] + m[0, 1], m[1, 1] - m[0, 0] - m[2, 2], 0, 0],
            [m[2, 0] + m[0, 2], m[2, 1] + m[1, 2], m[2, 2] - m[0, 0] - m[1, 1], 0],
            [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], m[0, 0] + m[1, 1] + m[2, 2]],
        ],
        dtype=np.float64,
    )
    k /= 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def read_equirect_size(cameras_txt: Path) -> tuple[int, int]:
    with cameras_txt.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            if elems[1] != "EQUIRECTANGULAR":
                raise SystemExit(f"expected EQUIRECTANGULAR camera, got {line}")
            return int(elems[2]), int(elems[3])
    raise SystemExit(f"no cameras found in {cameras_txt}")


def read_images_txt(path: Path) -> list[ColmapImage]:
    images: list[ColmapImage] = []
    with path.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            images.append(
                ColmapImage(
                    image_id=int(elems[0]),
                    qvec=np.array([float(v) for v in elems[1:5]], dtype=np.float64),
                    tvec=np.array([float(v) for v in elems[5:8]], dtype=np.float64),
                    camera_id=int(elems[8]),
                    name=elems[9],
                )
            )
            f.readline()
    return images


def build_face_map(face: str, face_size: int, fov: float, src_width: int, src_height: int) -> tuple[np.ndarray, np.ndarray]:
    focal = (face_size / 2.0) / math.tan(math.radians(fov) / 2.0)
    center = face_size / 2.0
    xs = (np.arange(face_size, dtype=np.float32) + 0.5 - center) / focal
    ys = (np.arange(face_size, dtype=np.float32) + 0.5 - center) / focal
    x_grid, y_grid = np.meshgrid(xs, ys)
    dirs_face = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)
    equi_from_face = FACE_AXES[face].astype(np.float32).T
    dirs_equi = dirs_face @ equi_from_face.T
    dx, dy, dz = dirs_equi[..., 0], dirs_equi[..., 1], dirs_equi[..., 2]
    theta = np.arctan2(dx, dz)
    phi = np.arctan2(-dy, np.sqrt(dx * dx + dz * dz))
    map_x = (theta / (2.0 * math.pi) + 0.5) * src_width
    map_y = (0.5 - phi / math.pi) * src_height
    return map_x.astype(np.float32), map_y.astype(np.float32)


def write_cubemap_camera(path: Path, face_size: int, fov: float) -> None:
    focal = (face_size / 2.0) / math.tan(math.radians(fov) / 2.0)
    center = face_size / 2.0
    path.write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 PINHOLE {face_size} {face_size} {focal:.17g} {focal:.17g} {center:.17g} {center:.17g}\n"
    )


def write_cubemap_images(path: Path, records: list[tuple[int, np.ndarray, np.ndarray, str]]) -> None:
    with path.open("w") as out:
        out.write("# Image list with two lines of data per image:\n")
        out.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        out.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        out.write(f"# Number of images: {len(records)}, mean observations per image: 0\n")
        for image_id, qvec, tvec, name in records:
            out.write(
                f"{image_id} "
                + " ".join(f"{v:.17g}" for v in qvec)
                + " "
                + " ".join(f"{v:.17g}" for v in tvec)
                + f" 1 {name}\n\n"
            )


def copy_points3d_for_splatting(src: Path, dst: Path) -> None:
    with src.open() as fin, dst.open("w") as out:
        out.write("# 3D point list with one line of data per point:\n")
        out.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for line in fin:
            if not line.strip() or line.startswith("#"):
                continue
            out.write(" ".join(line.split()[:8]) + "\n")


def write_empty_rigs_frames(out_sparse: Path) -> None:
    (out_sparse / "rigs.txt").write_text(
        "# Rig calib list with one line of data per calib:\n"
        "#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[] as "
        "(SENSOR_TYPE, SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])\n"
        "# Number of rigs: 0\n"
    )
    (out_sparse / "frames.txt").write_text(
        "# Frame list with one line of data per frame:\n"
        "#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], "
        "NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)\n"
        "# Number of frames: 0\n"
    )


def remap_image(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, is_mask: bool) -> np.ndarray:
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def mask_output_names(face_name: str, mode: str) -> list[str]:
    if mode == "colmap":
        return [f"{face_name}.png"]
    if mode == "stem":
        return [f"{Path(face_name).stem}.png"]
    return [f"{face_name}.png", f"{Path(face_name).stem}.png"]


def write_image(path: Path, image: np.ndarray, params: list[int] | None = None) -> None:
    if not cv2.imwrite(str(path), image, params or []):
        raise RuntimeError(f"failed to write image: {path}")


def convert_cubemap_image(
    idx: int,
    image: ColmapImage,
    *,
    faces: list[str],
    image_ext: str,
    cubemap_jpg_quality: int,
    mask_name_mode: str,
    maps: dict[str, tuple[np.ndarray, np.ndarray]],
    source_images: Path,
    source_masks: Path,
    out_images: Path,
    out_masks: Path,
) -> list[tuple[int, np.ndarray, np.ndarray, str]]:
    src = cv2.imread(str(source_images / image.name), cv2.IMREAD_COLOR)
    if src is None:
        raise RuntimeError(f"cannot read image: {source_images / image.name}")
    mask = None
    mask_path = source_masks / f"{image.name}.png"
    if source_masks.exists() and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"cannot read mask: {mask_path}")

    records: list[tuple[int, np.ndarray, np.ndarray, str]] = []
    source_rot = qvec_to_rotmat(image.qvec)
    stem = Path(image.name).stem
    base_id = (idx - 1) * len(faces) + 1

    for face_idx, face in enumerate(faces):
        face_name = f"{stem}_{face}.{image_ext}"
        map_x, map_y = maps[face]
        face_image = remap_image(src, map_x, map_y, is_mask=False)
        if image_ext == "jpg":
            write_image(out_images / face_name, face_image, [cv2.IMWRITE_JPEG_QUALITY, cubemap_jpg_quality])
        else:
            write_image(out_images / face_name, face_image)

        if mask is not None:
            face_mask = remap_image(mask, map_x, map_y, is_mask=True)
            for name in mask_output_names(face_name, mask_name_mode):
                write_image(out_masks / name, face_mask)

        face_from_equi = FACE_AXES[face]
        records.append((base_id + face_idx, rotmat_to_qvec(face_from_equi @ source_rot), face_from_equi @ image.tvec, face_name))

    return records


def convert_cubemap(args: argparse.Namespace) -> None:
    run = args.run
    sparse_txt = run / "colmap_cli_shared" / "sparse_txt"
    source_images = run / "frames"
    source_masks = run / "colmap_masks"
    out = run / f"cubemap_{args.face_size}_{len(args.faces)}faces"
    out_images = out / "images"
    out_masks = out / "masks"
    out_sparse_txt = out / "sparse_txt"
    out_sparse_bin = out / "sparse" / "0"
    ensure_dir(out_images)
    ensure_dir(out_masks)
    ensure_dir(out_sparse_txt)
    ensure_dir(out_sparse_bin)

    src_width, src_height = read_equirect_size(sparse_txt / "cameras.txt")
    images = read_images_txt(sparse_txt / "images.txt")
    maps = {face: build_face_map(face, args.face_size, args.fov, src_width, src_height) for face in args.faces}

    workers = args.cubemap_workers if args.cubemap_workers > 0 else args.threads
    workers = max(1, min(workers, len(images) or 1))
    records_by_image: dict[int, list[tuple[int, np.ndarray, np.ndarray, str]]] = {}
    print(f"cubemap workers: {workers}", flush=True)

    if workers == 1:
        for idx, image in enumerate(images, 1):
            records_by_image[idx] = convert_cubemap_image(
                idx,
                image,
                faces=args.faces,
                image_ext=args.image_ext,
                cubemap_jpg_quality=args.cubemap_jpg_quality,
                mask_name_mode=args.mask_name_mode,
                maps=maps,
                source_images=source_images,
                source_masks=source_masks,
                out_images=out_images,
                out_masks=out_masks,
            )
            print(f"[{idx}/{len(images)}] {image.name}: wrote {len(args.faces)} faces", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    convert_cubemap_image,
                    idx,
                    image,
                    faces=args.faces,
                    image_ext=args.image_ext,
                    cubemap_jpg_quality=args.cubemap_jpg_quality,
                    mask_name_mode=args.mask_name_mode,
                    maps=maps,
                    source_images=source_images,
                    source_masks=source_masks,
                    out_images=out_images,
                    out_masks=out_masks,
                ): (idx, image.name)
                for idx, image in enumerate(images, 1)
            }
            done = 0
            for future in as_completed(futures):
                idx, image_name = futures[future]
                records_by_image[idx] = future.result()
                done += 1
                print(f"[{done}/{len(images)}] {image_name}: wrote {len(args.faces)} faces", flush=True)

    records = [record for idx in sorted(records_by_image) for record in records_by_image[idx]]

    write_cubemap_camera(out_sparse_txt / "cameras.txt", args.face_size, args.fov)
    write_cubemap_images(out_sparse_txt / "images.txt", records)
    copy_points3d_for_splatting(sparse_txt / "points3D.txt", out_sparse_txt / "points3D.txt")
    write_empty_rigs_frames(out_sparse_txt)
    run_cmd([str(args.colmap), "model_converter", "--input_path", str(out_sparse_txt), "--output_path", str(out_sparse_bin), "--output_type", "BIN"])
    print(f"done: {out}", flush=True)


def convert_sparse_models_to_text(colmap: Path, root: Path, binary_dir_name: str, text_dir_name: str) -> None:
    binary_root = root / binary_dir_name
    text_root = root / text_dir_name
    if not binary_root.exists():
        return
    for model_dir in sorted(p for p in binary_root.iterdir() if p.is_dir()):
        out_dir = text_root / model_dir.name
        ensure_dir(out_dir)
        run_cmd(
            [
                str(colmap),
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(out_dir),
                "--output_type",
                "TXT",
            ]
        )


def run_panorama_sfm_workflow(args: argparse.Namespace) -> None:
    from pano_3dgs.panorama_sfm import run_panorama_sfm

    output_path = run_panorama_sfm(args)
    convert_sparse_models_to_text(args.colmap, output_path, "sparse", "sparse_txt")
    convert_sparse_models_to_text(args.colmap, output_path, "sparse_equirectangular", "sparse_equirectangular_txt")


def run_all(args: argparse.Namespace) -> None:
    args.run = scene_run_dir(args.runs_dir, args.scene, args.rate_hz, args.equirect_width)
    extract_args = argparse.Namespace(**vars(args))
    extract_args.window_seconds = 1.0 / args.rate_hz
    extract_args.chunk_size = None
    extract_sharpest(extract_args)

    if args.dynamic_mask_dir:
        target = args.run / "dynamic_masks"
        ensure_dir(target)
        for path in sorted(args.dynamic_mask_dir.glob("*.png")):
            dst = target / path.name
            if not dst.exists():
                os.link(path, dst)
    elif args.skip_sam3:
        print("SAM3 skipped by --skip-sam3", flush=True)
    elif args.sam3_model:
        run_sam3(args)
    else:
        raise SystemExit("SAM3 is enabled by default, but --sam3-model is not set. Use --skip-sam3 to continue without SAM3 masks.")

    make_colmap_masks(args)
    run_colmap(args)
    convert_cubemap(args)


def add_common_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, required=True)


def add_colmap_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--colmap", type=Path, default=env_path("PANO3DGS_COLMAP", DEFAULT_COLMAP))
    parser.add_argument("--gpu-index", default=env_str("PANO3DGS_GPU_INDEX", "0"))
    parser.add_argument("--threads", type=int, default=env_int("PANO3DGS_THREADS", 8))
    parser.add_argument("--clean-colmap", action=argparse.BooleanOptionalAction, default=env_bool("PANO3DGS_CLEAN_COLMAP", False))
    parser.add_argument("--max-features", type=int, default=env_int("PANO3DGS_MAX_FEATURES", 12000))
    parser.add_argument("--overlap", type=int, default=env_int("PANO3DGS_OVERLAP", 25))
    parser.add_argument("--equirect-width", type=int, default=env_int("PANO3DGS_EQUIRECT_WIDTH", 7680))
    parser.add_argument("--equirect-height", type=int, default=env_int("PANO3DGS_EQUIRECT_HEIGHT", 3840))


def add_extract_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=env_float("PANO3DGS_WINDOW_SECONDS", 0.5))
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--scale-width", type=int, default=env_int("PANO3DGS_SCALE_WIDTH", 1920))
    parser.add_argument("--roi", type=parse_box, default=(0.0, 0.08, 1.0, 0.92))
    parser.add_argument("--fallback-fps", type=float, default=env_float("PANO3DGS_FALLBACK_FPS", 30.0))
    parser.add_argument("--frame-jpg-quality", type=int, default=env_int("PANO3DGS_FRAME_JPG_QUALITY", 98))
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--progress", type=int, default=env_int("PANO3DGS_PROGRESS", 300))


def add_mask_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dynamic-mask-dir", type=Path, default=env_path("PANO3DGS_DYNAMIC_MASK_DIR"))
    parser.add_argument("--mask-heuristics", action=argparse.BooleanOptionalAction, default=env_bool_or_none("PANO3DGS_MASK_HEURISTICS"))
    parser.add_argument("--mask-progress", type=int, default=env_int("PANO3DGS_MASK_PROGRESS", 50))
    parser.add_argument("--sky-mask", action=argparse.BooleanOptionalAction, default=env_bool("PANO3DGS_SKY_MASK", True))
    parser.add_argument("--zenith-mask", type=float, default=env_float("PANO3DGS_ZENITH_MASK", 0.04))
    parser.add_argument("--nadir-mask", type=float, default=env_float("PANO3DGS_NADIR_MASK", 0.04))


def add_sam3_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sam3-model", type=Path, default=env_path("PANO3DGS_SAM3_MODEL"))
    parser.add_argument("--sam3-repo", type=Path, default=env_path("PANO3DGS_SAM3_REPO", "/tmp/sam3-official"))
    parser.add_argument("--device", default=env_str("PANO3DGS_DEVICE", "cuda:0"))
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default=env_str("PANO3DGS_DTYPE", "bfloat16"))
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--sam3-prompts", type=parse_list, default=parse_list(env_str("PANO3DGS_SAM3_PROMPTS", "")))
    parser.add_argument("--sam3-roi", type=parse_box, action="append", default=[])
    parser.add_argument("--score", type=float, default=env_float("PANO3DGS_SAM3_SCORE", 0.35))
    parser.add_argument("--min-area", type=float, default=env_float("PANO3DGS_SAM3_MIN_AREA", 0.00005))
    parser.add_argument("--max-area", type=float, default=env_float("PANO3DGS_SAM3_MAX_AREA", 0.80))
    parser.add_argument("--dilate", type=int, default=env_int("PANO3DGS_DILATE", 7))


def add_cubemap_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--face-size", type=int, default=env_int("PANO3DGS_FACE_SIZE", 2048))
    parser.add_argument("--faces", type=parse_faces, default=parse_faces(env_str("PANO3DGS_FACES", ",".join(DEFAULT_FACES))))
    parser.add_argument("--fov", type=float, default=env_float("PANO3DGS_FOV", 90.0))
    parser.add_argument("--image-ext", choices=["jpg", "png"], default=env_str("PANO3DGS_IMAGE_EXT", "jpg"))
    parser.add_argument("--cubemap-jpg-quality", type=int, default=env_int("PANO3DGS_CUBEMAP_JPG_QUALITY", 95))
    parser.add_argument("--cubemap-workers", type=int, default=env_int("PANO3DGS_CUBEMAP_WORKERS", 0))
    parser.add_argument("--mask-name-mode", choices=["colmap", "stem", "both"], default=env_str("PANO3DGS_MASK_NAME_MODE", "colmap"))


def add_panorama_sfm_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pycolmap-path", type=Path, default=env_path("PANO3DGS_PYCOLMAP_PATH"))
    parser.add_argument(
        "--require-pycolmap-cuda",
        action=argparse.BooleanOptionalAction,
        default=env_bool("PANO3DGS_REQUIRE_PYCOLMAP_CUDA", True),
    )
    parser.add_argument("--panorama-sfm-output", type=Path, default=env_path("PANO3DGS_PANORAMA_SFM_OUTPUT"))
    parser.add_argument(
        "--pano-render-type",
        choices=["perspective_overlapping", "perspective_non_overlapping"],
        default=env_str("PANO3DGS_PANO_RENDER_TYPE", "perspective_overlapping"),
    )
    parser.add_argument(
        "--panorama-virtual-camera-model",
        choices=["pinhole", "simple_pinhole"],
        default=env_str("PANO3DGS_PANORAMA_VIRTUAL_CAMERA_MODEL", "pinhole"),
    )
    parser.add_argument(
        "--panorama-matcher",
        choices=["sequential", "exhaustive", "vocabtree", "spatial"],
        default=env_str("PANO3DGS_PANORAMA_MATCHER", "sequential"),
    )
    parser.add_argument(
        "--panorama-mapper",
        choices=["incremental", "global"],
        default=env_str("PANO3DGS_PANORAMA_MAPPER", "incremental"),
    )
    parser.add_argument(
        "--panorama-ba-backend",
        choices=["ceres", "caspar"],
        default=env_str("PANO3DGS_PANORAMA_BA_BACKEND", "ceres"),
    )
    parser.add_argument(
        "--panorama-loop-detection",
        action=argparse.BooleanOptionalAction,
        default=env_bool("PANO3DGS_PANORAMA_LOOP_DETECTION", False),
    )
    parser.add_argument("--panorama-vocab-tree-path", type=Path, default=env_path("PANO3DGS_PANORAMA_VOCAB_TREE_PATH"))
    parser.add_argument("--panorama-workers", type=int, default=env_int("PANO3DGS_PANORAMA_WORKERS", 0))
    parser.add_argument(
        "--panorama-use-input-masks",
        action=argparse.BooleanOptionalAction,
        default=env_bool("PANO3DGS_PANORAMA_USE_INPUT_MASKS", True),
    )
    parser.add_argument(
        "--rerender-perspective",
        action=argparse.BooleanOptionalAction,
        default=env_bool("PANO3DGS_RERENDER_PERSPECTIVE", False),
    )
    parser.add_argument(
        "--rerun-panorama-features",
        action=argparse.BooleanOptionalAction,
        default=env_bool("PANO3DGS_RERUN_PANORAMA_FEATURES", False),
    )
    parser.add_argument(
        "--rerun-panorama-matching",
        action=argparse.BooleanOptionalAction,
        default=env_bool("PANO3DGS_RERUN_PANORAMA_MATCHING", False),
    )
    parser.add_argument(
        "--clean-panorama-sfm",
        action=argparse.BooleanOptionalAction,
        default=env_bool("PANO3DGS_CLEAN_PANORAMA_SFM", False),
    )


def build_parser() -> argparse.ArgumentParser:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="pano-3dgs")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract")
    add_common_run(p)
    add_extract_options(p)
    p.set_defaults(func=extract_sharpest)

    p = sub.add_parser("sam3")
    add_common_run(p)
    add_sam3_options(p)
    p.set_defaults(func=run_sam3)

    p = sub.add_parser("masks")
    add_common_run(p)
    add_mask_options(p)
    p.set_defaults(func=make_colmap_masks)

    p = sub.add_parser("colmap")
    add_common_run(p)
    add_colmap_options(p)
    p.set_defaults(func=run_colmap)

    p = sub.add_parser("cubemap")
    add_common_run(p)
    add_colmap_options(p)
    add_cubemap_options(p)
    p.set_defaults(func=convert_cubemap)

    p = sub.add_parser("panorama-sfm")
    add_common_run(p)
    add_colmap_options(p)
    add_panorama_sfm_options(p)
    p.set_defaults(func=run_panorama_sfm_workflow)

    p = sub.add_parser("run")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--runs-dir", type=Path, default=env_path("PANO3DGS_RUNS_DIR", "runs"))
    p.add_argument("--rate-hz", type=float, default=env_float("PANO3DGS_RATE_HZ", 2.0))
    add_extract_options_no_video(p)
    add_mask_options(p)
    add_sam3_options(p)
    p.add_argument("--skip-sam3", action="store_true", default=env_bool("PANO3DGS_SKIP_SAM3", False))
    add_colmap_options(p)
    add_cubemap_options(p)
    p.set_defaults(func=run_all)

    return parser


def add_extract_options_no_video(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scale-width", type=int, default=env_int("PANO3DGS_SCALE_WIDTH", 1920))
    parser.add_argument("--roi", type=parse_box, default=(0.0, 0.08, 1.0, 0.92))
    parser.add_argument("--fallback-fps", type=float, default=env_float("PANO3DGS_FALLBACK_FPS", 30.0))
    parser.add_argument("--frame-jpg-quality", type=int, default=env_int("PANO3DGS_FRAME_JPG_QUALITY", 98))
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--progress", type=int, default=env_int("PANO3DGS_PROGRESS", 300))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
