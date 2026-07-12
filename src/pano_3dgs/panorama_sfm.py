from __future__ import annotations

import argparse
import collections
import json
import math
import shutil
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from PIL import ExifTags, Image, UnidentifiedImageError

from pano_3dgs.pycolmap_io import write_reconstruction_pair
from pano_3dgs.sfm_utils import (
    configure_feature_extraction_options,
    configure_feature_matching_options,
    configure_incremental_ba,
    import_pycolmap,
    prepare_feature_database,
    pycolmap_device,
    run_matcher,
)
from pano_3dgs.utils import (
    IMAGE_EXTENSIONS,
    choose_worker_count,
    prune_generated_files,
    require_complete_mask_set,
)


@dataclass
class PanoRenderOptions:
    num_steps_yaw: int
    pitches_deg: Sequence[float]
    hfov_deg: float
    vfov_deg: float


@dataclass(frozen=True)
class RenderMap:
    x_coords: np.ndarray
    y_coords: np.ndarray
    base_mask: np.ndarray


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


def estimate_pano_render_memory(
    pano_width: int,
    pano_height: int,
    render_options: PanoRenderOptions,
    *,
    has_source_mask: bool,
) -> tuple[int, int]:
    pano_pixels = pano_width * pano_height
    virtual_width = int(pano_width * render_options.hfov_deg / 360)
    virtual_height = int(pano_height * render_options.vfov_deg / 180)
    virtual_pixels = virtual_width * virtual_height
    virtual_cameras = render_options.num_steps_yaw * len(render_options.pitches_deg)

    pano_image = pano_pixels * 3
    source_mask = pano_pixels if has_source_mask else 0
    rendered_image = virtual_pixels * 3
    rendered_mask = virtual_pixels
    rendered_source_mask = virtual_pixels if has_source_mask else 0

    shared_rays = virtual_pixels * 3 * 8
    shared_remap_coords = virtual_pixels * 2 * 4 * virtual_cameras
    shared_base_masks = virtual_pixels * virtual_cameras
    shared_memory = shared_rays + shared_remap_coords + shared_base_masks
    per_worker = (
        pano_image + source_mask + rendered_image + rendered_mask + rendered_source_mask
    )
    safety_factor = 3.0 if sys.platform == "win32" else 2.0
    return int(per_worker * safety_factor), shared_memory


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


def get_virtual_camera_rays(camera) -> np.ndarray:
    size = (camera.width, camera.height)
    x, y = np.indices(size).astype(np.float32)
    xy = np.column_stack([x.ravel(), y.ravel()])
    xy += 0.5
    xy_norm = camera.cam_from_img(image_points=xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def spherical_img_from_cam(
    image_size: tuple[int, int], rays_in_cam: np.ndarray
) -> np.ndarray:
    if image_size[0] != image_size[1] * 2:
        raise ValueError("Only 360 degree panoramas with width=2*height are supported.")
    rays = rays_in_cam.T
    yaw = np.arctan2(rays[0], rays[2])
    pitch = -np.arctan2(rays[1], np.linalg.norm(rays[[0, 2]], axis=0))
    u = (1 + yaw / math.pi) / 2
    v = (1 - pitch * 2 / math.pi) / 2
    return np.stack([u, v], axis=-1) * image_size


def get_virtual_rotations(
    num_steps_yaw: int, pitches_deg: Sequence[float]
) -> list[np.ndarray]:
    rotations = []
    yaws = np.linspace(0, 360, num_steps_yaw, endpoint=False)
    for pitch_deg in pitches_deg:
        yaw_offset = (360 / num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            rotations.append(rotation_x(-pitch_deg) @ rotation_y(-yaw_deg))
    return rotations


def virtual_image_names(
    pano_image_names: Sequence[str],
    render_options: PanoRenderOptions,
) -> set[str]:
    num_cameras = render_options.num_steps_yaw * len(render_options.pitches_deg)
    return {
        f"pano_camera{camera_index}/{pano_name}"
        for pano_name in pano_image_names
        for camera_index in range(num_cameras)
    }


def build_render_configuration(
    *,
    pano_width: int,
    pano_height: int,
    render_options: PanoRenderOptions,
    camera_model: str,
    use_source_masks: bool,
) -> dict[str, object]:
    return {
        "version": 1,
        "pano_width": pano_width,
        "pano_height": pano_height,
        "num_steps_yaw": render_options.num_steps_yaw,
        "pitches_deg": list(render_options.pitches_deg),
        "hfov_deg": render_options.hfov_deg,
        "vfov_deg": render_options.vfov_deg,
        "camera_model": camera_model,
        "use_source_masks": use_source_masks,
    }


def render_configuration_changed(
    output_image_dir: Path,
    configuration: dict[str, object],
) -> bool:
    if not output_image_dir.exists() or not any(
        path.is_file() for path in output_image_dir.rglob("*")
    ):
        return False
    manifest_path = output_image_dir.parent / "render_config.json"
    try:
        stored_configuration = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return stored_configuration != configuration


def write_render_configuration(
    output_image_dir: Path,
    configuration: dict[str, object],
) -> None:
    manifest_path = output_image_dir.parent / "render_config.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)


def create_pano_rig_config(
    pycolmap, cams_from_pano_rotation: Sequence[np.ndarray], ref_idx: int = 0
):
    rig_cameras = []
    zero_translation = np.zeros((3, 1), dtype=np.float64)
    for idx, cam_from_pano_rotation in enumerate(cams_from_pano_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = (
                cam_from_pano_rotation @ cams_from_pano_rotation[ref_idx].T
            )
            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_rotation), zero_translation
            )
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
        self.cams_from_pano_rotation = get_virtual_rotations(
            render_options.num_steps_yaw, render_options.pitches_deg
        )
        self.rig_config = create_pano_rig_config(pycolmap, self.cams_from_pano_rotation)
        self.cam_centers_in_pano = np.einsum(
            "nij,i->nj", self.cams_from_pano_rotation, [0, 0, 1]
        )
        self._lock = Lock()
        self._camera = None
        self._pano_size: tuple[int, int] | None = None
        self._rays_in_cam: np.ndarray | None = None
        self._render_maps: list[RenderMap] | None = None

    @property
    def camera(self):
        if self._camera is None:
            raise RuntimeError("no panorama has been initialized")
        return self._camera

    def process(self, pano_name: str) -> None:
        if not self.rerender and self._render_outputs_exist(pano_name):
            return

        pano_path = self.pano_image_dir / pano_name
        try:
            with Image.open(pano_path) as pano_pil_image:
                pano_exif = pano_pil_image.getexif()
                pano_image = np.asarray(pano_pil_image).copy()
        except UnidentifiedImageError as exc:
            raise RuntimeError(f"cannot read panorama image: {pano_path}") from exc

        gpsonly_exif = Image.Exif()
        gpsonly_exif[ExifTags.IFD.GPSInfo] = pano_exif.get_ifd(ExifTags.IFD.GPSInfo)

        pano_height, pano_width, *_ = pano_image.shape
        if pano_width != pano_height * 2:
            raise ValueError(f"Only 360 degree panoramas are supported: {pano_path}")

        self.ensure_camera(pano_width, pano_height)

        source_mask = self._read_source_mask(pano_name, pano_width, pano_height)
        if self._render_maps is None:
            raise RuntimeError("perspective render maps were not initialized")

        for cam_idx, render_map in enumerate(self._render_maps):
            image = cv2.remap(
                pano_image,
                render_map.x_coords,
                render_map.y_coords,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
            mask = render_map.base_mask.copy()
            if source_mask is not None:
                rendered_source_mask = cv2.remap(
                    source_mask,
                    render_map.x_coords,
                    render_map.y_coords,
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
            if not cv2.imwrite(str(mask_path), mask):
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
                self._render_maps = self._build_render_maps()
            elif (pano_width, pano_height) != self._pano_size:
                raise ValueError("Panoramas of different sizes are not supported.")

    def _build_render_maps(self) -> list[RenderMap]:
        if self._camera is None or self._pano_size is None or self._rays_in_cam is None:
            raise RuntimeError("cannot build render maps before camera initialization")

        render_maps = []
        score_chunk_size = 262_144
        for cam_idx, cam_from_pano_rotation in enumerate(self.cams_from_pano_rotation):
            rays_in_pano = self._rays_in_cam @ cam_from_pano_rotation
            xy_in_pano = spherical_img_from_cam(self._pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(
                self._camera.width, self._camera.height, 2
            ).astype(np.float32)
            xy_in_pano -= 0.5
            x_coords, y_coords = np.moveaxis(xy_in_pano, [0, 1, 2], [2, 1, 0])

            closest_camera = np.empty(rays_in_pano.shape[0], dtype=np.uint8)
            for start in range(0, rays_in_pano.shape[0], score_chunk_size):
                end = min(start + score_chunk_size, rays_in_pano.shape[0])
                closest_camera[start:end] = np.argmax(
                    rays_in_pano[start:end] @ self.cam_centers_in_pano.T,
                    axis=-1,
                )
            base_mask = (
                ((closest_camera == cam_idx) * 255)
                .astype(np.uint8)
                .reshape(self._camera.width, self._camera.height)
                .transpose()
            )
            render_maps.append(
                RenderMap(x_coords.copy(), y_coords.copy(), base_mask.copy())
            )
        return render_maps

    def _render_outputs_exist(self, pano_name: str) -> bool:
        for rig_camera in self.rig_config.cameras:
            image_name = rig_camera.image_prefix + pano_name
            if not (self.output_image_dir / image_name).exists():
                return False
            if not (self.output_mask_dir / f"{image_name}.png").exists():
                return False
        return True

    def _read_source_mask(
        self, pano_name: str, pano_width: int, pano_height: int
    ) -> np.ndarray | None:
        if self.source_mask_dir is None:
            return None
        mask_path = self.source_mask_dir / f"{pano_name}.png"
        if not mask_path.exists():
            raise RuntimeError(f"source mask is missing: {mask_path}")
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
        frame_images = sorted(
            images_by_frame.values(),
            key=lambda images: self.split_image_name(images[0].name)[1],
        )

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
                    point2d_idx: (base_idx + point2d_idx, pano_name)
                    for point2d_idx in range(num_points2d)
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
                new_point2d_idx, pano_name = old_to_new_point2d[element.image_id][
                    element.point2D_idx
                ]
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
    if not pano_image_names:
        raise ValueError("at least one panorama is required for perspective rendering")
    with Image.open(pano_image_dir / pano_image_names[0]) as pano_pil_image:
        pano_width, pano_height = pano_pil_image.size
    configuration = build_render_configuration(
        pano_width=pano_width,
        pano_height=pano_height,
        render_options=render_options,
        camera_model=camera_model,
        use_source_masks=source_mask_dir is not None,
    )
    configuration_changed = render_configuration_changed(
        output_image_dir, configuration
    )
    effective_rerender = rerender or configuration_changed
    if configuration_changed:
        print(
            "perspective render configuration changed; rerendering outputs", flush=True
        )
    processor = PanoProcessor(
        pycolmap,
        pano_image_dir,
        output_image_dir,
        output_mask_dir,
        source_mask_dir,
        render_options,
        camera_model,
        effective_rerender,
    )
    estimated_worker_memory, shared_memory = estimate_pano_render_memory(
        pano_width,
        pano_height,
        render_options,
        has_source_mask=source_mask_dir is not None,
    )
    worker_choice = choose_worker_count(
        max_workers,
        len(pano_image_names),
        estimated_per_worker_bytes=estimated_worker_memory,
        shared_memory_bytes=shared_memory,
    )
    workers = worker_choice.workers
    print(
        f"rendering {len(pano_image_names)} panoramas with {workers} workers ({worker_choice.reason})",
        flush=True,
    )
    print("precomputing perspective remap maps", flush=True)
    processor.ensure_camera(pano_width, pano_height)
    done = 0
    previous_cv_threads = cv2.getNumThreads()
    if workers > 1:
        cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=workers) as thread_pool:
            futures = [
                thread_pool.submit(processor.process, pano_name)
                for pano_name in pano_image_names
            ]
            for future in as_completed(futures):
                future.result()
                done += 1
                print(
                    f"[{done}/{len(pano_image_names)}] rendered perspective rig images",
                    flush=True,
                )
    finally:
        if workers > 1:
            cv2.setNumThreads(previous_cv_threads)
    write_render_configuration(output_image_dir, configuration)
    return processor


def run_panorama_sfm(args: argparse.Namespace) -> Path:
    pycolmap = import_pycolmap(args.pycolmap_path, args.require_pycolmap_cuda)
    device = pycolmap_device(pycolmap, args.require_pycolmap_cuda)
    pycolmap.set_random_seed(0)
    extraction_options = configure_feature_extraction_options(pycolmap, args)
    matching_options = configure_feature_matching_options(pycolmap, args)

    output_path = args.panorama_sfm_output or (args.run / "panorama_sfm")
    if args.clean_panorama_sfm and output_path.exists():
        print(f"cleaning panorama SfM workspace: {output_path}", flush=True)
        shutil.rmtree(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    database_path = output_path / "database.db"
    image_dir = output_path / "images"
    mask_dir = output_path / "masks"
    rec_path = output_path / "sparse"
    image_dir.mkdir(exist_ok=True, parents=True)
    mask_dir.mkdir(exist_ok=True, parents=True)
    rec_path.mkdir(exist_ok=True, parents=True)

    pano_image_dir = args.run / "frames"
    pano_image_names = sorted(
        p.relative_to(pano_image_dir).as_posix()
        for p in pano_image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not pano_image_names:
        raise SystemExit(f"no panorama frames found in {pano_image_dir}")
    print(
        f"found {len(pano_image_names)} panorama frames in {pano_image_dir}", flush=True
    )
    source_mask_dir = (
        require_complete_mask_set(pano_image_names, args.run / "colmap_masks")
        if args.panorama_use_input_masks
        else None
    )

    with Image.open(pano_image_dir / pano_image_names[0]) as pano_pil_image:
        pano_width, pano_height = pano_pil_image.size
    render_options = PANO_RENDER_OPTIONS[args.pano_render_type]
    expected_camera = create_virtual_camera(
        pycolmap,
        pano_width=pano_width,
        pano_height=pano_height,
        hfov_deg=render_options.hfov_deg,
        vfov_deg=render_options.vfov_deg,
        camera_model=args.panorama_virtual_camera_model,
    )
    render_configuration = build_render_configuration(
        pano_width=pano_width,
        pano_height=pano_height,
        render_options=render_options,
        camera_model=args.panorama_virtual_camera_model,
        use_source_masks=source_mask_dir is not None,
    )
    render_outputs_changed = render_configuration_changed(
        image_dir,
        render_configuration,
    )
    expected_image_names = virtual_image_names(pano_image_names, render_options)
    removed_images = prune_generated_files(image_dir, expected_image_names)
    removed_masks = prune_generated_files(
        mask_dir,
        {f"{image_name}.png" for image_name in expected_image_names},
    )
    if removed_images or removed_masks:
        print(
            f"removed stale perspective outputs: images={removed_images} masks={removed_masks}",
            flush=True,
        )
    database_plan = prepare_feature_database(
        pycolmap,
        database_path,
        expected_camera_model_id=int(expected_camera.model),
        expected_image_names=expected_image_names,
        rerun_features=(
            args.rerun_panorama_features
            or args.rerender_perspective
            or render_outputs_changed
        ),
        rerun_matching=args.rerun_panorama_matching,
    )

    processor = render_perspective_images(
        pycolmap,
        pano_image_names,
        pano_image_dir,
        image_dir,
        mask_dir,
        source_mask_dir,
        render_options,
        args.panorama_virtual_camera_model,
        args.panorama_workers,
        args.rerender_perspective,
    )
    rig_config = processor.rig_config
    rendered_camera = rig_config.cameras[0].camera

    if database_plan.extract_features:
        print(
            f"extracting {args.feature_type} features with perspective rig camera",
            flush=True,
        )
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
    else:
        print(f"reusing perspective feature database: {database_path}", flush=True)

    if database_plan.match_features:
        print(f"matching features with {args.panorama_matcher}", flush=True)
        run_matcher(
            pycolmap,
            database_path,
            matching_options,
            device,
            matcher=args.panorama_matcher,
            overlap=args.overlap,
            loop_detection=args.panorama_loop_detection,
            vocab_tree_path=args.panorama_vocab_tree_path,
            num_threads=args.threads,
        )
    else:
        print(f"reusing perspective feature matches: {database_path}", flush=True)

    if args.panorama_mapper == "incremental":
        if (
            args.panorama_ba_backend == "caspar"
            and args.panorama_virtual_camera_model == "simple_pinhole"
        ):
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
        configure_incremental_ba(
            pycolmap,
            opts,
            backend_name=args.panorama_ba_backend,
            num_threads=args.threads,
            require_cuda=args.require_pycolmap_cuda,
            gpu_index=args.gpu_index,
        )
        print(f"incremental mapper BA backend: {args.panorama_ba_backend}", flush=True)
        recs = pycolmap.incremental_mapping(database_path, image_dir, rec_path, opts)
    elif args.panorama_mapper == "global":
        if args.panorama_ba_backend == "caspar":
            raise SystemExit(
                "Caspar is only supported for incremental mapper; use --panorama-mapper incremental."
            )
        opts = pycolmap.GlobalPipelineOptions(
            mapper=pycolmap.GlobalMapperOptions(refine_sensor_from_rig=False)
        )
        opts.mapper.bundle_adjustment.refine_focal_length = False
        opts.mapper.bundle_adjustment.refine_principal_point = False
        opts.mapper.bundle_adjustment.refine_extra_params = False
        recs = pycolmap.global_mapping(database_path, image_dir, rec_path, opts)
    else:
        raise SystemExit(f"unknown panorama mapper: {args.panorama_mapper}")

    for idx, rec in recs.items():
        print(f"#{idx} {rec.summary()}", flush=True)
        write_reconstruction_pair(
            rec,
            rec_path / str(idx),
            output_path / "sparse_txt" / str(idx),
        )

    equirect_rec_path = output_path / "sparse_equirectangular"
    print(
        "converting perspective rig reconstruction back to equirectangular", flush=True
    )
    for idx, rec in recs.items():
        equirect_rec = processor.convert_to_equirectangular(rec)
        dst = equirect_rec_path / str(idx)
        write_reconstruction_pair(
            equirect_rec,
            dst,
            output_path / "sparse_equirectangular_txt" / str(idx),
        )
        print(f"equirect #{idx} {equirect_rec.summary()}", flush=True)

    print(f"done: {output_path}", flush=True)
    return output_path
