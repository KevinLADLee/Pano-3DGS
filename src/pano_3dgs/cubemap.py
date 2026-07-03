from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pano_3dgs.defaults import FACE_AXES
from pano_3dgs.utils import ensure_dir, run_cmd


@dataclass
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str

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
