from __future__ import annotations

import argparse
import collections
import math
import os
import sqlite3
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from PIL import ExifTags, Image, UnidentifiedImageError


@dataclass
class PanoRenderOptions:
    num_steps_yaw: int
    pitches_deg: Sequence[float]
    hfov_deg: float
    vfov_deg: float


PANO_RENDER_OPTIONS: dict[str, PanoRenderOptions] = {
    "perspective_overlapping": PanoRenderOptions(
        num_steps_yaw=4,
        pitches_deg=(-35.0, 0.0, 35.0),
        hfov_deg=90.0,
        vfov_deg=90.0,
    ),
    "perspective_non_overlapping": PanoRenderOptions(
        num_steps_yaw=4,
        pitches_deg=(0.0,),
        hfov_deg=90.0,
        vfov_deg=90.0,
    ),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def import_pycolmap(pycolmap_path: Path | None = None, require_cuda: bool = True):
    if pycolmap_path:
        sys.path.insert(0, str(pycolmap_path))
    try:
        import pycolmap
    except ImportError as exc:
        raise SystemExit(
            "Official panorama SfM workflow requires CUDA-enabled PyCOLMAP. "
            "Install the CUDA 12 wheel with `uv sync` / `uv pip install pycolmap-cuda12`, "
            "or build PyCOLMAP from your CUDA COLMAP source with `uv pip install /home/invs/repos/colmap`."
        ) from exc
    if require_cuda:
        if not getattr(pycolmap, "has_cuda", False):
            raise SystemExit(
                "Imported PyCOLMAP does not report CUDA support. "
                "Use `pycolmap-cuda12` on Linux/CUDA 12, or build PyCOLMAP from the CUDA-enabled COLMAP source."
            )
        if hasattr(pycolmap, "get_num_cuda_devices") and pycolmap.get_num_cuda_devices() < 1:
            raise SystemExit("PyCOLMAP has CUDA support, but no CUDA device is currently visible.")
    print(
        f"pycolmap={getattr(pycolmap, '__version__', 'unknown')} cuda={getattr(pycolmap, 'has_cuda', False)}",
        flush=True,
    )
    return pycolmap


def rotation_x(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rotation_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def virtual_camera_model_id(pycolmap, name: str):
    if name == "pinhole":
        return pycolmap.CameraModelId.PINHOLE
    if name == "simple_pinhole":
        return pycolmap.CameraModelId.SIMPLE_PINHOLE
    raise SystemExit(f"unknown panorama virtual camera model: {name}")


def create_virtual_camera(
    pycolmap,
    pano_width: int,
    pano_height: int,
    hfov_deg: float,
    vfov_deg: float,
    camera_model: str,
):
    image_width = int(pano_width * hfov_deg / 360)
    image_height = int(pano_height * vfov_deg / 180)
    focal = image_width / (2 * math.tan(math.radians(hfov_deg) / 2))
    camera = pycolmap.Camera.create_from_model_id(
        camera_id=0,
        model=virtual_camera_model_id(pycolmap, camera_model),
        focal_length=focal,
        width=image_width,
        height=image_height,
    )
    camera.has_prior_focal_length = True
    return camera


def database_camera_model_ids(database_path: Path) -> set[int]:
    with sqlite3.connect(database_path) as db:
        rows = db.execute("SELECT DISTINCT model FROM cameras").fetchall()
    return {int(row[0]) for row in rows}


def camera_model_id_name(pycolmap, model_id: int) -> str:
    try:
        return pycolmap.CameraModelId(model_id).name
    except ValueError:
        return f"UNKNOWN({model_id})"


def get_virtual_camera_rays(camera) -> np.ndarray:
    size = (camera.width, camera.height)
    x, y = np.indices(size).astype(np.float32)
    xy = np.column_stack([x.ravel(), y.ravel()])
    xy += 0.5
    xy_norm = camera.cam_from_img(image_points=xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def spherical_img_from_cam(image_size: tuple[int, int], rays_in_cam: np.ndarray) -> np.ndarray:
    if image_size[0] != image_size[1] * 2:
        raise ValueError("Only 360 degree panoramas with width=2*height are supported.")
    rays = rays_in_cam.T
    yaw = np.arctan2(rays[0], rays[2])
    pitch = -np.arctan2(rays[1], np.linalg.norm(rays[[0, 2]], axis=0))
    u = (1 + yaw / math.pi) / 2
    v = (1 - pitch * 2 / math.pi) / 2
    return np.stack([u, v], axis=-1) * image_size


def get_virtual_rotations(num_steps_yaw: int, pitches_deg: Sequence[float]) -> list[np.ndarray]:
    rotations = []
    yaws = np.linspace(0, 360, num_steps_yaw, endpoint=False)
    for pitch_deg in pitches_deg:
        yaw_offset = (360 / num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            rotations.append(rotation_x(-pitch_deg) @ rotation_y(-yaw_deg))
    return rotations


def create_pano_rig_config(pycolmap, cams_from_pano_rotation: Sequence[np.ndarray], ref_idx: int = 0):
    rig_cameras = []
    zero_translation = np.zeros((3, 1), dtype=np.float64)
    for idx, cam_from_pano_rotation in enumerate(cams_from_pano_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = cam_from_pano_rotation @ cams_from_pano_rotation[ref_idx].T
            cam_from_rig = pycolmap.Rigid3d(pycolmap.Rotation3d(cam_from_ref_rotation), zero_translation)
        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=idx == ref_idx,
                image_prefix=f"pano_camera{idx}/",
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)


class PanoProcessor:
    def __init__(
        self,
        pycolmap,
        pano_image_dir: Path,
        output_image_dir: Path,
        output_mask_dir: Path,
        source_mask_dir: Path | None,
        render_options: PanoRenderOptions,
        camera_model: str,
        rerender: bool,
    ):
        self.pycolmap = pycolmap
        self.pano_image_dir = pano_image_dir
        self.output_image_dir = output_image_dir
        self.output_mask_dir = output_mask_dir
        self.source_mask_dir = source_mask_dir
        self.render_options = render_options
        self.camera_model = camera_model
        self.rerender = rerender
        self.cams_from_pano_rotation = get_virtual_rotations(render_options.num_steps_yaw, render_options.pitches_deg)
        self.rig_config = create_pano_rig_config(pycolmap, self.cams_from_pano_rotation)
        self.cam_centers_in_pano = np.einsum("nij,i->nj", self.cams_from_pano_rotation, [0, 0, 1])
        self._lock = Lock()
        self._camera = None
        self._pano_size: tuple[int, int] | None = None
        self._rays_in_cam: np.ndarray | None = None

    def process(self, pano_name: str) -> None:
        if not self.rerender and self._render_outputs_exist(pano_name):
            return

        pano_path = self.pano_image_dir / pano_name
        try:
            pano_pil_image = Image.open(pano_path)
        except UnidentifiedImageError:
            print(f"skipping unreadable image: {pano_path}", flush=True)
            return

        pano_exif = pano_pil_image.getexif()
        gpsonly_exif = Image.Exif()
        gpsonly_exif[ExifTags.IFD.GPSInfo] = pano_exif.get_ifd(ExifTags.IFD.GPSInfo)

        pano_image = np.asarray(pano_pil_image)
        pano_height, pano_width, *_ = pano_image.shape
        if pano_width != pano_height * 2:
            raise ValueError(f"Only 360 degree panoramas are supported: {pano_path}")

        self.ensure_camera(pano_width, pano_height)

        source_mask = self._read_source_mask(pano_name, pano_width, pano_height)
        assert self._camera is not None
        assert self._pano_size is not None
        assert self._rays_in_cam is not None

        for cam_idx, cam_from_pano_rotation in enumerate(self.cams_from_pano_rotation):
            rays_in_pano = self._rays_in_cam @ cam_from_pano_rotation
            xy_in_pano = spherical_img_from_cam(self._pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(self._camera.width, self._camera.height, 2).astype(np.float32)
            xy_in_pano -= 0.5
            x_coords, y_coords = np.moveaxis(xy_in_pano, [0, 1, 2], [2, 1, 0])
            image = cv2.remap(pano_image, x_coords, y_coords, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

            closest_camera = np.argmax(rays_in_pano @ self.cam_centers_in_pano.T, axis=-1)
            mask = (
                ((closest_camera == cam_idx) * 255)
                .astype(np.uint8)
                .reshape(self._camera.width, self._camera.height)
                .transpose()
            )
            if source_mask is not None:
                rendered_source_mask = cv2.remap(
                    source_mask,
                    x_coords,
                    y_coords,
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_WRAP,
                )
                mask[rendered_source_mask == 0] = 0

            image_name = self.rig_config.cameras[cam_idx].image_prefix + pano_name
            mask_name = f"{image_name}.png"

            image_path = self.output_image_dir / image_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(image_path, exif=gpsonly_exif)

            mask_path = self.output_mask_dir / mask_name
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            if not self.pycolmap.Bitmap.from_array(mask).write(mask_path):
                raise RuntimeError(f"Cannot write {mask_path}")

    def ensure_camera(self, pano_width: int, pano_height: int) -> None:
        with self._lock:
            if self._camera is None:
                self._camera = create_virtual_camera(
                    self.pycolmap,
                    pano_width=pano_width,
                    pano_height=pano_height,
                    hfov_deg=self.render_options.hfov_deg,
                    vfov_deg=self.render_options.vfov_deg,
                    camera_model=self.camera_model,
                )
                for rig_camera in self.rig_config.cameras:
                    rig_camera.camera = self._camera
                self._pano_size = (pano_width, pano_height)
                self._rays_in_cam = get_virtual_camera_rays(self._camera)
            elif (pano_width, pano_height) != self._pano_size:
                raise ValueError("Panoramas of different sizes are not supported.")

    def ensure_camera_from_pano(self, pano_name: str) -> None:
        if self._camera is not None:
            return
        pano_path = self.pano_image_dir / pano_name
        with Image.open(pano_path) as pano_pil_image:
            pano_width, pano_height = pano_pil_image.size
        if pano_width != pano_height * 2:
            raise ValueError(f"Only 360 degree panoramas are supported: {pano_path}")
        self.ensure_camera(pano_width, pano_height)

    def _render_outputs_exist(self, pano_name: str) -> bool:
        for rig_camera in self.rig_config.cameras:
            image_name = rig_camera.image_prefix + pano_name
            if not (self.output_image_dir / image_name).exists():
                return False
            if not (self.output_mask_dir / f"{image_name}.png").exists():
                return False
        return True

    def _read_source_mask(self, pano_name: str, pano_width: int, pano_height: int) -> np.ndarray | None:
        if self.source_mask_dir is None:
            return None
        mask_path = self.source_mask_dir / f"{pano_name}.png"
        if not mask_path.exists():
            return None
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"cannot read mask: {mask_path}")
        if mask.shape != (pano_height, pano_width):
            raise RuntimeError(f"mask shape mismatch: {mask_path}")
        return mask

    def split_image_name(self, image_name: str) -> tuple[int, str]:
        for cam_idx, rig_camera in enumerate(self.rig_config.cameras):
            prefix = rig_camera.image_prefix
            if image_name.startswith(prefix):
                return cam_idx, image_name[len(prefix) :]
        raise ValueError(f"Unknown virtual camera for image {image_name!r}.")

    def convert_to_equirectangular(self, reconstruction):
        if self._camera is None or self._pano_size is None:
            raise RuntimeError("No panorama was rendered yet.")
        pycolmap = self.pycolmap
        pano_width, pano_height = self._pano_size

        equirect = pycolmap.Reconstruction()
        equirect_camera = pycolmap.Camera.create_from_model_id(
            camera_id=1,
            model=pycolmap.CameraModelId.EQUIRECTANGULAR,
            focal_length=0.0,
            width=pano_width,
            height=pano_height,
        )
        equirect.add_camera_with_trivial_rig(equirect_camera)

        pano_from_ref = pycolmap.Rigid3d(
            pycolmap.Rotation3d(self.cams_from_pano_rotation[0]),
            np.zeros((3, 1), dtype=np.float64),
        ).inverse()

        images_by_frame = collections.defaultdict(list)
        for image in reconstruction.images.values():
            if image.has_pose:
                images_by_frame[image.frame_id].append(image)

        old_to_new_point2d: dict[int, dict[int, tuple[int, str]]] = {}
        pano_to_image_id: dict[str, int] = {}
        frame_images = sorted(images_by_frame.values(), key=lambda images: self.split_image_name(images[0].name)[1])

        for image_id, images in enumerate(frame_images, start=1):
            pano_name = self.split_image_name(images[0].name)[1]
            pano_to_image_id[pano_name] = image_id
            frame = images[0].frame
            rig_from_world = frame.rig_from_world
            pano_from_world = pano_from_ref * rig_from_world

            keypoints = []
            for image in images:
                cam_idx = self.split_image_name(image.name)[0]
                num_points2d = len(image.points2D)
                if num_points2d == 0:
                    old_to_new_point2d[image.image_id] = {}
                    continue
                xy = np.array([point2d.xy for point2d in image.points2D])
                rays_in_cam = np.asarray(self._camera.cam_ray_from_img(image_points=xy))
                rays_in_cam /= np.linalg.norm(rays_in_cam, axis=-1, keepdims=True)
                rays_in_pano = rays_in_cam @ self.cams_from_pano_rotation[cam_idx]
                xy_in_pano = spherical_img_from_cam(self._pano_size, rays_in_pano)

                base_idx = len(keypoints)
                keypoints.extend(xy_in_pano)
                old_to_new_point2d[image.image_id] = {
                    point2d_idx: (base_idx + point2d_idx, pano_name) for point2d_idx in range(num_points2d)
                }

            equirect.add_image_with_trivial_frame(
                pycolmap.Image(
                    name=pano_name,
                    keypoints=keypoints,
                    camera_id=equirect_camera.camera_id,
                    image_id=image_id,
                ),
                pano_from_world,
            )

        for point3d_id, point3d in reconstruction.points3D.items():
            track = pycolmap.Track()
            for element in point3d.track.elements:
                new_point2d_idx, pano_name = old_to_new_point2d[element.image_id][element.point2D_idx]
                track.add_element(pano_to_image_id[pano_name], new_point2d_idx)
            equirect.add_point3D_with_id(
                point3d_id,
                pycolmap.Point3D(xyz=point3d.xyz, color=point3d.color, track=track),
            )

        return equirect


def render_perspective_images(
    pycolmap,
    pano_image_names: Sequence[str],
    pano_image_dir: Path,
    output_image_dir: Path,
    output_mask_dir: Path,
    source_mask_dir: Path | None,
    render_options: PanoRenderOptions,
    camera_model: str,
    max_workers: int,
    rerender: bool,
) -> PanoProcessor:
    processor = PanoProcessor(
        pycolmap,
        pano_image_dir,
        output_image_dir,
        output_mask_dir,
        source_mask_dir,
        render_options,
        camera_model,
        rerender,
    )
    workers = max_workers if max_workers > 0 else min(32, max(1, (os.cpu_count() or 2) - 1))
    workers = max(1, min(workers, len(pano_image_names) or 1))
    print(f"rendering {len(pano_image_names)} panoramas with {workers} workers", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as thread_pool:
        futures = [thread_pool.submit(processor.process, pano_name) for pano_name in pano_image_names]
        for future in as_completed(futures):
            future.result()
            done += 1
            print(f"[{done}/{len(pano_image_names)}] rendered perspective rig images", flush=True)
    if processor._camera is None and pano_image_names:
        processor.ensure_camera_from_pano(pano_image_names[0])
    return processor


def pycolmap_device(pycolmap, require_cuda: bool):
    return pycolmap.Device.cuda if require_cuda else pycolmap.Device.auto


def pycolmap_ba_backend(pycolmap, name: str):
    if name == "ceres":
        return pycolmap.BundleAdjustmentBackend.CERES
    if name == "caspar":
        if not hasattr(pycolmap, "CasparBundleAdjustmentOptions"):
            raise SystemExit("This PyCOLMAP build does not expose Caspar bundle adjustment options.")
        return pycolmap.BundleAdjustmentBackend.CASPAR
    raise SystemExit(f"unknown bundle adjustment backend: {name}")


def check_incremental_options(opts, backend_name: str) -> None:
    try:
        ok = opts.check()
    except Exception as exc:
        if backend_name == "caspar":
            raise SystemExit(
                "Caspar was requested, but this PyCOLMAP build was not compiled with CASPAR_ENABLED. "
                "Build COLMAP/PyCOLMAP from source with -DCASPAR_ENABLED=ON, or use --panorama-ba-backend ceres."
            ) from exc
        raise
    if ok is False and backend_name == "caspar":
        raise SystemExit(
            "Caspar was requested, but this PyCOLMAP build was not compiled with CASPAR_ENABLED. "
            "Build COLMAP/PyCOLMAP from source with -DCASPAR_ENABLED=ON, or use --panorama-ba-backend ceres."
        )


def run_matcher(pycolmap, args: argparse.Namespace, database_path: Path, matching_options, device) -> None:
    if args.panorama_matcher == "sequential":
        pairing_options = pycolmap.SequentialPairingOptions()
        pairing_options.overlap = args.overlap
        pairing_options.quadratic_overlap = True
        pairing_options.expand_rig_images = True
        pairing_options.loop_detection = args.panorama_loop_detection
        pairing_options.num_threads = args.threads
        if args.panorama_vocab_tree_path:
            pairing_options.vocab_tree_path = str(args.panorama_vocab_tree_path)
        elif args.panorama_loop_detection:
            raise SystemExit(
                "Sequential loop detection needs a local --panorama-vocab-tree-path. "
                "Disable it with --no-panorama-loop-detection to avoid COLMAP downloading the tree."
            )
        pycolmap.match_sequential(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
    elif args.panorama_matcher == "exhaustive":
        pycolmap.match_exhaustive(database_path, matching_options=matching_options, device=device)
    elif args.panorama_matcher == "vocabtree":
        if not args.panorama_vocab_tree_path:
            raise SystemExit("vocabtree matcher needs --panorama-vocab-tree-path to avoid network download.")
        pairing_options = pycolmap.VocabTreePairingOptions()
        pairing_options.vocab_tree_path = str(args.panorama_vocab_tree_path)
        pairing_options.num_threads = args.threads
        pycolmap.match_vocabtree(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
    elif args.panorama_matcher == "spatial":
        pycolmap.match_spatial(database_path, matching_options=matching_options, device=device)
    else:
        raise SystemExit(f"unknown panorama matcher: {args.panorama_matcher}")


def run_panorama_sfm(args: argparse.Namespace) -> Path:
    pycolmap = import_pycolmap(args.pycolmap_path, args.require_pycolmap_cuda)
    device = pycolmap_device(pycolmap, args.require_pycolmap_cuda)
    pycolmap.set_random_seed(0)

    output_path = args.panorama_sfm_output or (args.run / "panorama_sfm")
    if args.clean_panorama_sfm and output_path.exists():
        import shutil

        print(f"cleaning panorama SfM workspace: {output_path}", flush=True)
        shutil.rmtree(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    database_path = output_path / "database.db"
    rerun_features = args.rerun_panorama_features or args.rerender_perspective
    rerun_matching = args.rerun_panorama_matching or rerun_features
    reuse_database = database_path.exists() and not rerun_features and not rerun_matching
    if database_path.exists() and (rerun_features or rerun_matching):
        print(f"removing panorama SfM database for rerun: {database_path}", flush=True)
        database_path.unlink()

    image_dir = output_path / "images"
    mask_dir = output_path / "masks"
    rec_path = output_path / "sparse"
    image_dir.mkdir(exist_ok=True, parents=True)
    mask_dir.mkdir(exist_ok=True, parents=True)
    rec_path.mkdir(exist_ok=True, parents=True)

    pano_image_dir = args.run / "frames"
    source_mask_dir = args.run / "colmap_masks" if args.panorama_use_input_masks else None
    pano_image_names = sorted(
        p.relative_to(pano_image_dir).as_posix()
        for p in pano_image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not pano_image_names:
        raise SystemExit(f"no panorama frames found in {pano_image_dir}")
    print(f"found {len(pano_image_names)} panorama frames in {pano_image_dir}", flush=True)

    processor = render_perspective_images(
        pycolmap,
        pano_image_names,
        pano_image_dir,
        image_dir,
        mask_dir,
        source_mask_dir if source_mask_dir and source_mask_dir.exists() else None,
        PANO_RENDER_OPTIONS[args.pano_render_type],
        args.panorama_virtual_camera_model,
        args.panorama_workers,
        args.rerender_perspective,
    )
    rig_config = processor.rig_config
    rendered_camera = rig_config.cameras[0].camera
    expected_camera_model_id = int(rendered_camera.model)
    if reuse_database:
        try:
            existing_camera_model_ids = database_camera_model_ids(database_path)
        except sqlite3.Error as exc:
            print(
                f"removing invalid panorama SfM database: {database_path} ({exc})",
                flush=True,
            )
            database_path.unlink()
            reuse_database = False
        else:
            existing_names = ", ".join(
                camera_model_id_name(pycolmap, model_id)
                for model_id in sorted(existing_camera_model_ids)
            )
            if existing_camera_model_ids != {expected_camera_model_id}:
                print(
                    "removing panorama SfM database because camera model changed: "
                    f"{existing_names or 'none'} -> {rendered_camera.model_name}",
                    flush=True,
                )
                database_path.unlink()
                reuse_database = False

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.num_threads = args.threads
    extraction_options.gpu_index = str(args.gpu_index)
    extraction_options.sift.max_num_features = args.max_features

    if reuse_database:
        print(f"reusing existing feature/match database: {database_path}", flush=True)
    else:
        print("extracting features with perspective rig camera", flush=True)
        pycolmap.extract_features(
            database_path,
            image_dir,
            reader_options=pycolmap.ImageReaderOptions(
                mask_path=mask_dir,
                camera_model=rendered_camera.model_name,
                camera_params=rendered_camera.params_to_string(),
            ),
            extraction_options=extraction_options,
            camera_mode=pycolmap.CameraMode.PER_FOLDER,
            device=device,
        )

        with pycolmap.Database.open(database_path) as db:
            pycolmap.apply_rig_config([rig_config], db)

        matching_options = pycolmap.FeatureMatchingOptions()
        matching_options.num_threads = args.threads
        matching_options.gpu_index = str(args.gpu_index)
        matching_options.guided_matching = True
        matching_options.rig_verification = True
        matching_options.skip_image_pairs_in_same_frame = True
        print(f"matching features with {args.panorama_matcher}", flush=True)
        run_matcher(pycolmap, args, database_path, matching_options, device)

    if args.panorama_mapper == "incremental":
        ba_backend = pycolmap_ba_backend(pycolmap, args.panorama_ba_backend)
        if args.panorama_ba_backend == "caspar" and args.panorama_virtual_camera_model == "simple_pinhole":
            raise SystemExit(
                "Caspar does not support SIMPLE_PINHOLE in this COLMAP build. "
                "Use --panorama-virtual-camera-model pinhole for Caspar, "
                "or --panorama-ba-backend ceres for SIMPLE_PINHOLE."
            )
        opts = pycolmap.IncrementalPipelineOptions(
            ba_refine_sensor_from_rig=False,
            ba_refine_focal_length=False,
            ba_refine_principal_point=False,
            ba_refine_extra_params=False,
        )
        opts.num_threads = args.threads
        opts.ba_use_gpu = args.require_pycolmap_cuda
        opts.ba_gpu_index = str(args.gpu_index)
        opts.ba_local_backend = ba_backend
        opts.ba_global_backend = ba_backend
        check_incremental_options(opts, args.panorama_ba_backend)
        print(f"incremental mapper BA backend: {args.panorama_ba_backend}", flush=True)
        recs = pycolmap.incremental_mapping(database_path, image_dir, rec_path, opts)
    elif args.panorama_mapper == "global":
        if args.panorama_ba_backend == "caspar":
            raise SystemExit("Caspar is only supported for incremental mapper; use --panorama-mapper incremental.")
        opts = pycolmap.GlobalPipelineOptions(mapper=pycolmap.GlobalMapperOptions(refine_sensor_from_rig=False))
        opts.mapper.bundle_adjustment.refine_focal_length = False
        opts.mapper.bundle_adjustment.refine_principal_point = False
        opts.mapper.bundle_adjustment.refine_extra_params = False
        recs = pycolmap.global_mapping(database_path, image_dir, rec_path, opts)
    else:
        raise SystemExit(f"unknown panorama mapper: {args.panorama_mapper}")

    for idx, rec in recs.items():
        print(f"#{idx} {rec.summary()}", flush=True)

    equirect_rec_path = output_path / "sparse_equirectangular"
    print("converting perspective rig reconstruction back to equirectangular", flush=True)
    for idx, rec in recs.items():
        equirect_rec = processor.convert_to_equirectangular(rec)
        dst = equirect_rec_path / str(idx)
        dst.mkdir(exist_ok=True, parents=True)
        equirect_rec.write(dst)
        print(f"equirect #{idx} {equirect_rec.summary()}", flush=True)

    print(f"done: {output_path}", flush=True)
    return output_path
