from __future__ import annotations

import argparse
import os

from pano_3dgs.extract import extract_sharpest
from pano_3dgs.panorama_sfm import run_panorama_sfm
from pano_3dgs.sam3_masks import make_colmap_masks, run_sam3
from pano_3dgs.utils import default_scene_name, ensure_dir, scene_run_dir


def run_panorama_sfm_workflow(args: argparse.Namespace) -> None:
    run_panorama_sfm(args)


def run_all(args: argparse.Namespace) -> None:
    if args.scene is None:
        args.scene = default_scene_name(args.video)
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
    run_panorama_sfm(args)
