from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

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
    link_or_copy_file,
    prune_generated_files,
    require_complete_mask_set,
)


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
        if (
            link_or_copy_file(
                source_image_dir / image_name, output_image_dir / image_name
            )
            != "existing"
        ):
            copied_images += 1
        if source_mask_dir is None:
            continue
        source_mask = source_mask_dir / f"{image_name}.png"
        if (
            link_or_copy_file(source_mask, output_mask_dir / f"{image_name}.png")
            != "existing"
        ):
            copied_masks += 1
    print(
        f"prepared equirect images={len(pano_image_names)} "
        f"copied_images={copied_images} copied_masks={copied_masks}",
        flush=True,
    )


def create_equirect_camera(pycolmap, image_path: Path):
    with Image.open(image_path) as image:
        width, height = image.size
    if width != height * 2:
        raise SystemExit(
            f"expected a 2:1 equirectangular image, got {width}x{height}: {image_path}"
        )
    camera = pycolmap.Camera.create_from_model_id(
        camera_id=0,
        model=pycolmap.CameraModelId.EQUIRECTANGULAR,
        focal_length=0.0,
        width=width,
        height=height,
    )
    return camera


def run_equirect_mapping(
    pycolmap,
    args: argparse.Namespace,
    database_path: Path,
    image_dir: Path,
    rec_path: Path,
):
    if args.equirect_mapper == "incremental":
        opts = pycolmap.IncrementalPipelineOptions(
            ba_refine_focal_length=False,
            ba_refine_principal_point=False,
            ba_refine_extra_params=False,
        )
        configure_incremental_ba(
            pycolmap,
            opts,
            backend_name=args.equirect_ba_backend,
            num_threads=args.threads,
            require_cuda=args.require_pycolmap_cuda,
            gpu_index=args.gpu_index,
        )
        print(f"incremental mapper BA backend: {args.equirect_ba_backend}", flush=True)
        return pycolmap.incremental_mapping(database_path, image_dir, rec_path, opts)

    if args.equirect_mapper == "global":
        if args.equirect_ba_backend == "caspar":
            raise SystemExit(
                "Caspar is only supported for incremental mapper; use --equirect-mapper incremental."
            )
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
    matching_options = configure_feature_matching_options(
        pycolmap, args, rig_verification=False
    )

    output_path = args.equirect_sfm_output or (args.run / "equirect_sfm")
    if args.clean_equirect_sfm and output_path.exists():
        print(f"cleaning equirect SfM workspace: {output_path}", flush=True)
        shutil.rmtree(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    database_path = output_path / "database.db"
    source_image_dir = args.run / "frames"
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
    print(
        f"found {len(pano_image_names)} equirect frames in {source_image_dir}",
        flush=True,
    )
    source_mask_dir = (
        require_complete_mask_set(pano_image_names, args.run / "colmap_masks")
        if args.equirect_use_input_masks
        else None
    )
    removed_images = prune_generated_files(image_dir, pano_image_names)
    expected_masks = (
        {f"{image_name}.png" for image_name in pano_image_names}
        if source_mask_dir is not None
        else set()
    )
    removed_masks = prune_generated_files(mask_dir, expected_masks)
    if removed_images or removed_masks:
        print(
            f"removed stale equirect inputs: images={removed_images} masks={removed_masks}",
            flush=True,
        )
    prepare_equirect_inputs(
        pano_image_names,
        source_image_dir,
        source_mask_dir,
        image_dir,
        mask_dir,
    )

    camera = create_equirect_camera(pycolmap, image_dir / pano_image_names[0])
    database_plan = prepare_feature_database(
        pycolmap,
        database_path,
        expected_camera_model_id=int(camera.model),
        expected_image_names=pano_image_names,
        rerun_features=args.rerun_equirect_features,
        rerun_matching=args.rerun_equirect_matching,
    )

    if database_plan.extract_features:
        print(
            f"extracting {args.feature_type} features with EQUIRECTANGULAR camera",
            flush=True,
        )
        pycolmap.extract_features(
            database_path,
            image_dir,
            reader_options=pycolmap.ImageReaderOptions(
                mask_path=mask_dir if source_mask_dir is not None else "",
                camera_model=camera.model_name,
                camera_params=camera.params_to_string(),
            ),
            extraction_options=extraction_options,
            camera_mode=pycolmap.CameraMode.SINGLE,
            device=device,
        )
    else:
        print(f"reusing equirect feature database: {database_path}", flush=True)

    if database_plan.match_features:
        print(f"matching equirect features with {args.equirect_matcher}", flush=True)
        run_matcher(
            pycolmap,
            database_path,
            matching_options,
            device,
            matcher=args.equirect_matcher,
            overlap=args.overlap,
            loop_detection=args.equirect_loop_detection,
            vocab_tree_path=args.equirect_vocab_tree_path,
            num_threads=args.threads,
        )
    else:
        print(f"reusing equirect feature matches: {database_path}", flush=True)

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
