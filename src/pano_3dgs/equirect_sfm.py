from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from pano_3dgs.panorama_sfm import (
    IMAGE_EXTENSIONS,
    camera_model_id_name,
    check_incremental_options,
    configure_feature_extraction_options,
    configure_feature_matching_options,
    database_camera_model_ids,
    import_pycolmap,
    pycolmap_ba_backend,
    pycolmap_device,
    remove_database_files,
    run_matcher,
)
from pano_3dgs.pycolmap_io import write_reconstruction_pair
from pano_3dgs.utils import link_or_copy_file


def run_equirect_sfm_workflow(args: argparse.Namespace) -> None:
    run_equirect_sfm(args)


def prepare_equirect_inputs(
    pano_image_names: list[str],
    source_image_dir: Path,
    source_mask_dir: Path | None,
    output_image_dir: Path,
    output_mask_dir: Path,
) -> None:
    copied_images = 0
    copied_masks = 0
    for image_name in pano_image_names:
        if link_or_copy_file(source_image_dir / image_name, output_image_dir / image_name) != "existing":
            copied_images += 1
        if source_mask_dir is None:
            continue
        source_mask = source_mask_dir / f"{image_name}.png"
        if source_mask.exists():
            if link_or_copy_file(source_mask, output_mask_dir / f"{image_name}.png") != "existing":
                copied_masks += 1
    print(f"prepared equirect images={len(pano_image_names)} copied_images={copied_images} copied_masks={copied_masks}", flush=True)


def create_equirect_camera(pycolmap, image_path: Path):
    with Image.open(image_path) as image:
        width, height = image.size
    if width != height * 2:
        raise SystemExit(f"expected a 2:1 equirectangular image, got {width}x{height}: {image_path}")
    camera = pycolmap.Camera.create_from_model_id(
        camera_id=0,
        model=pycolmap.CameraModelId.EQUIRECTANGULAR,
        focal_length=0.0,
        width=width,
        height=height,
    )
    return camera


def equirect_matcher_namespace(args: argparse.Namespace) -> argparse.Namespace:
    matcher_args = argparse.Namespace(**vars(args))
    matcher_args.panorama_matcher = args.equirect_matcher
    matcher_args.panorama_loop_detection = args.equirect_loop_detection
    matcher_args.panorama_vocab_tree_path = args.equirect_vocab_tree_path
    return matcher_args


def run_equirect_mapping(pycolmap, args: argparse.Namespace, database_path: Path, image_dir: Path, rec_path: Path):
    if args.equirect_mapper == "incremental":
        ba_backend = pycolmap_ba_backend(pycolmap, args.equirect_ba_backend)
        opts = pycolmap.IncrementalPipelineOptions(
            ba_refine_focal_length=False,
            ba_refine_principal_point=False,
            ba_refine_extra_params=False,
        )
        opts.num_threads = args.threads
        opts.ba_use_gpu = args.require_pycolmap_cuda
        opts.ba_gpu_index = str(args.gpu_index)
        opts.ba_local_backend = ba_backend
        opts.ba_global_backend = ba_backend
        check_incremental_options(opts, args.equirect_ba_backend)
        print(f"incremental mapper BA backend: {args.equirect_ba_backend}", flush=True)
        return pycolmap.incremental_mapping(database_path, image_dir, rec_path, opts)

    if args.equirect_mapper == "global":
        if args.equirect_ba_backend == "caspar":
            raise SystemExit("Caspar is only supported for incremental mapper; use --equirect-mapper incremental.")
        opts = pycolmap.GlobalPipelineOptions()
        opts.mapper.bundle_adjustment.refine_focal_length = False
        opts.mapper.bundle_adjustment.refine_principal_point = False
        opts.mapper.bundle_adjustment.refine_extra_params = False
        return pycolmap.global_mapping(database_path, image_dir, rec_path, opts)

    raise SystemExit(f"unknown equirect mapper: {args.equirect_mapper}")


def run_equirect_sfm(args: argparse.Namespace) -> Path:
    pycolmap = import_pycolmap(args.pycolmap_path, args.require_pycolmap_cuda)
    device = pycolmap_device(pycolmap, args.require_pycolmap_cuda)
    pycolmap.set_random_seed(0)
    extraction_options = configure_feature_extraction_options(pycolmap, args)
    matching_options = configure_feature_matching_options(pycolmap, args, rig_verification=False)

    output_path = args.equirect_sfm_output or (args.run / "equirect_sfm")
    if args.clean_equirect_sfm and output_path.exists():
        print(f"cleaning equirect SfM workspace: {output_path}", flush=True)
        shutil.rmtree(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    database_path = output_path / "database.db"
    rerun_features = args.rerun_equirect_features
    rerun_matching = args.rerun_equirect_matching or rerun_features
    reuse_database = database_path.exists() and not rerun_features and not rerun_matching
    if database_path.exists() and (rerun_features or rerun_matching):
        remove_database_files(database_path, "for equirect rerun")

    source_image_dir = args.run / "frames"
    source_mask_dir = args.run / "colmap_masks" if args.equirect_use_input_masks else None
    image_dir = output_path / "images"
    mask_dir = output_path / "masks"
    rec_path = output_path / "sparse"
    image_dir.mkdir(exist_ok=True, parents=True)
    mask_dir.mkdir(exist_ok=True, parents=True)
    rec_path.mkdir(exist_ok=True, parents=True)

    pano_image_names = sorted(
        p.relative_to(source_image_dir).as_posix()
        for p in source_image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not pano_image_names:
        raise SystemExit(f"no panorama frames found in {source_image_dir}")
    print(f"found {len(pano_image_names)} equirect frames in {source_image_dir}", flush=True)
    prepare_equirect_inputs(
        pano_image_names,
        source_image_dir,
        source_mask_dir if source_mask_dir and source_mask_dir.exists() else None,
        image_dir,
        mask_dir,
    )

    camera = create_equirect_camera(pycolmap, image_dir / pano_image_names[0])
    expected_camera_model_id = int(camera.model)
    if reuse_database:
        try:
            existing_camera_model_ids = database_camera_model_ids(database_path)
        except Exception as exc:
            remove_database_files(database_path, f"because it is invalid ({exc})")
            reuse_database = False
        else:
            if existing_camera_model_ids != {expected_camera_model_id}:
                existing_names = ", ".join(
                    camera_model_id_name(pycolmap, model_id)
                    for model_id in sorted(existing_camera_model_ids)
                )
                remove_database_files(
                    database_path,
                    f"because camera model changed: {existing_names or 'none'} -> {camera.model_name}",
                )
                reuse_database = False

    if reuse_database:
        print(f"reusing existing equirect feature/match database: {database_path}", flush=True)
    else:
        print(f"extracting {args.feature_type} features with EQUIRECTANGULAR camera", flush=True)
        pycolmap.extract_features(
            database_path,
            image_dir,
            reader_options=pycolmap.ImageReaderOptions(
                mask_path=mask_dir if mask_dir.exists() else "",
                camera_model=camera.model_name,
                camera_params=camera.params_to_string(),
            ),
            extraction_options=extraction_options,
            camera_mode=pycolmap.CameraMode.SINGLE,
            device=device,
        )

        print(f"matching equirect features with {args.equirect_matcher}", flush=True)
        run_matcher(
            pycolmap,
            equirect_matcher_namespace(args),
            database_path,
            matching_options,
            device,
        )

    recs = run_equirect_mapping(pycolmap, args, database_path, image_dir, rec_path)
    for idx, rec in recs.items():
        print(f"#{idx} {rec.summary()}", flush=True)
        write_reconstruction_pair(
            rec,
            rec_path / str(idx),
            output_path / "sparse_txt" / str(idx),
        )

    print(f"done: {output_path}", flush=True)
    return output_path
