from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pano_3dgs.panorama_sfm import (
    PANO_RENDER_OPTIONS,
    render_perspective_images,
    virtual_image_names,
)
from pano_3dgs.pycolmap_io import write_reconstruction_pair
from pano_3dgs.sfm_utils import import_pycolmap
from pano_3dgs.utils import prune_generated_files, require_complete_mask_set


@dataclass
class VirtualImage:
    image_id: int
    name: str
    cam_from_world: object
    keypoints: list[np.ndarray]


def resolve_input_sparse(args: argparse.Namespace) -> Path:
    if args.input_sparse:
        path = args.input_sparse
    else:
        path = args.run / "equirect_sfm" / "sparse" / "0"
        if not path.exists():
            sparse_root = args.run / "equirect_sfm" / "sparse"
            candidates = (
                sorted(p for p in sparse_root.iterdir() if p.is_dir())
                if sparse_root.exists()
                else []
            )
            if candidates:
                path = candidates[0]
    if not path.exists():
        raise SystemExit(
            f"equirectangular sparse reconstruction not found: {path}. "
            "Run equirect-sfm first, or pass --input-sparse."
        )
    return path


def load_equirect_reconstruction(pycolmap, sparse_path: Path):
    rec = pycolmap.Reconstruction(str(sparse_path))
    if not rec.images:
        raise SystemExit(f"input reconstruction has no images: {sparse_path}")
    camera_model_names = {camera.model_name for camera in rec.cameras.values()}
    if camera_model_names != {"EQUIRECTANGULAR"}:
        names = ", ".join(sorted(camera_model_names))
        raise SystemExit(f"expected EQUIRECTANGULAR input reconstruction, got: {names}")
    return rec


def registered_equirect_images(rec) -> list:
    images = [image for image in rec.images.values() if image.has_pose]
    if not images:
        raise SystemExit("input reconstruction has no registered images")
    return sorted(images, key=lambda image: image.name)


def image_cam_from_world(image):
    cam_from_world = image.cam_from_world
    return cam_from_world() if callable(cam_from_world) else cam_from_world


def build_virtual_images(
    pycolmap,
    equirect_images: list,
    processor,
) -> tuple[dict[tuple[int, int], VirtualImage], object]:
    camera = processor.camera
    camera.camera_id = 1
    virtual_images: dict[tuple[int, int], VirtualImage] = {}
    zero_translation = np.zeros((3, 1), dtype=np.float64)
    next_image_id = 1
    for equirect_image in equirect_images:
        pano_from_world = image_cam_from_world(equirect_image)
        for cam_idx, cam_from_pano_rotation in enumerate(
            processor.cams_from_pano_rotation
        ):
            cam_from_pano = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_pano_rotation),
                zero_translation,
            )
            cam_from_world = cam_from_pano * pano_from_world
            image_name = f"pano_camera{cam_idx}/{equirect_image.name}"
            virtual_images[(equirect_image.image_id, cam_idx)] = VirtualImage(
                image_id=next_image_id,
                name=image_name,
                cam_from_world=cam_from_world,
                keypoints=[],
            )
            next_image_id += 1
    return virtual_images, camera


def project_observations_to_pinhole(
    rec, processor, virtual_images: dict[tuple[int, int], VirtualImage], camera
):
    equirect_images = rec.images
    projected_tracks: dict[int, list[tuple[int, int]]] = {}
    camera_width = camera.width
    camera_height = camera.height

    for point3d_id, point3d in rec.points3D.items():
        observations: list[tuple[int, int]] = []
        for element in point3d.track.elements:
            equirect_image = equirect_images[element.image_id]
            if not equirect_image.has_pose:
                continue
            xyz_in_pano = image_cam_from_world(equirect_image) * point3d.xyz
            norm = np.linalg.norm(xyz_in_pano)
            if norm <= 0:
                continue
            ray_in_pano = xyz_in_pano / norm
            cam_idx = int(np.argmax(ray_in_pano @ processor.cam_centers_in_pano.T))
            virtual_image = virtual_images.get((equirect_image.image_id, cam_idx))
            if virtual_image is None:
                continue
            xyz_in_cam = virtual_image.cam_from_world * point3d.xyz
            if xyz_in_cam[2] <= 0:
                continue
            xy = np.asarray(camera.img_from_cam(np.asarray([xyz_in_cam])))[0]
            if not (0 <= xy[0] < camera_width and 0 <= xy[1] < camera_height):
                continue
            point2d_idx = len(virtual_image.keypoints)
            virtual_image.keypoints.append(xy)
            observations.append((virtual_image.image_id, point2d_idx))
        if observations:
            projected_tracks[point3d_id] = observations
    return projected_tracks


def build_pinhole_reconstruction(
    pycolmap, equirect_rec, processor, min_track_length: int
):
    equirect_images = registered_equirect_images(equirect_rec)
    virtual_images, camera = build_virtual_images(pycolmap, equirect_images, processor)
    projected_tracks = project_observations_to_pinhole(
        equirect_rec, processor, virtual_images, camera
    )

    pinhole_rec = pycolmap.Reconstruction()
    pinhole_rec.add_camera_with_trivial_rig(camera)

    for virtual_image in sorted(
        virtual_images.values(), key=lambda image: image.image_id
    ):
        pinhole_rec.add_image_with_trivial_frame(
            pycolmap.Image(
                name=virtual_image.name,
                keypoints=virtual_image.keypoints,
                camera_id=camera.camera_id,
                image_id=virtual_image.image_id,
            ),
            virtual_image.cam_from_world,
        )

    kept_points = 0
    for point3d_id, observations in projected_tracks.items():
        if len(observations) < min_track_length:
            continue
        track = pycolmap.Track()
        for image_id, point2d_idx in observations:
            track.add_element(image_id, point2d_idx)
        point3d = equirect_rec.points3D[point3d_id]
        pinhole_rec.add_point3D_with_id(
            point3d_id,
            pycolmap.Point3D(xyz=point3d.xyz, color=point3d.color, track=track),
        )
        kept_points += 1

    print(
        f"projected pinhole observations for {kept_points}/{len(equirect_rec.points3D)} points "
        f"across {len(virtual_images)} virtual images",
        flush=True,
    )
    return pinhole_rec


def export_pinhole_3dgs(args: argparse.Namespace) -> Path:
    pycolmap = import_pycolmap(args.pycolmap_path, require_cuda=False)
    input_sparse = resolve_input_sparse(args)
    equirect_rec = load_equirect_reconstruction(pycolmap, input_sparse)

    output_path = args.output or (args.run / "pinhole_3dgs")
    if args.clean and output_path.exists():
        print(f"cleaning PINHOLE 3DGS export: {output_path}", flush=True)
        shutil.rmtree(output_path)
    image_dir = output_path / "images"
    mask_dir = output_path / "masks"
    sparse_dir = output_path / "sparse"
    image_dir.mkdir(exist_ok=True, parents=True)
    mask_dir.mkdir(exist_ok=True, parents=True)
    sparse_dir.mkdir(exist_ok=True, parents=True)

    pano_image_dir = args.run / "frames"
    pano_image_names = [
        image.name for image in registered_equirect_images(equirect_rec)
    ]
    missing_images = [
        name for name in pano_image_names if not (pano_image_dir / name).is_file()
    ]
    if missing_images:
        preview = ", ".join(missing_images[:5])
        suffix = (
            f" (and {len(missing_images) - 5} more)" if len(missing_images) > 5 else ""
        )
        raise SystemExit(
            f"missing {len(missing_images)} registered panorama images: {preview}{suffix}"
        )
    source_mask_dir = (
        require_complete_mask_set(pano_image_names, args.run / "colmap_masks")
        if args.use_input_masks
        else None
    )
    render_options = PANO_RENDER_OPTIONS[args.render_type]
    expected_image_names = virtual_image_names(pano_image_names, render_options)
    removed_images = prune_generated_files(image_dir, expected_image_names)
    removed_masks = prune_generated_files(
        mask_dir,
        {f"{image_name}.png" for image_name in expected_image_names},
    )
    if removed_images or removed_masks:
        print(
            f"removed stale PINHOLE outputs: images={removed_images} masks={removed_masks}",
            flush=True,
        )

    print(
        f"rendering PINHOLE images from {len(pano_image_names)} panoramas", flush=True
    )
    processor = render_perspective_images(
        pycolmap,
        pano_image_names,
        pano_image_dir,
        image_dir,
        mask_dir,
        source_mask_dir,
        render_options,
        "pinhole",
        args.workers,
        args.rerender,
    )

    pinhole_rec = build_pinhole_reconstruction(
        pycolmap,
        equirect_rec,
        processor,
        args.min_track_length,
    )
    write_reconstruction_pair(
        pinhole_rec, sparse_dir / "0", output_path / "sparse_txt" / "0"
    )
    print(f"done: {output_path}", flush=True)
    return output_path
