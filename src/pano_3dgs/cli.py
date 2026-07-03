from __future__ import annotations

import argparse
from pathlib import Path

from pano_3dgs.config import Settings, find_cli_config, load_settings
from pano_3dgs.extract import extract_sharpest
from pano_3dgs.sam3_masks import make_colmap_masks, run_sam3
from pano_3dgs.utils import parse_box, parse_list
from pano_3dgs.workflow import run_all, run_panorama_sfm_workflow


def add_config_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML config file; CLI flags override it. Defaults to nearest pano3dgs.toml when present.",
    )


def add_common_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, required=True)


def add_sfm_options(parser: argparse.ArgumentParser, settings: Settings) -> None:
    parser.add_argument("--gpu-index", default=settings.gpu_index)
    parser.add_argument("--threads", type=int, default=settings.threads)
    parser.add_argument("--max-features", type=int, default=settings.max_features)
    parser.add_argument("--overlap", type=int, default=settings.overlap)


def add_extract_options(parser: argparse.ArgumentParser, settings: Settings) -> None:
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=settings.window_seconds)
    parser.add_argument("--chunk-size", type=int)
    add_extract_options_no_video(parser, settings)


def add_extract_options_no_video(parser: argparse.ArgumentParser, settings: Settings) -> None:
    parser.add_argument("--scale-width", type=int, default=settings.scale_width)
    parser.add_argument("--roi", type=parse_box, default=settings.roi)
    parser.add_argument("--fallback-fps", type=float, default=settings.fallback_fps)
    parser.add_argument("--frame-jpg-quality", type=int, default=settings.frame_jpg_quality)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--progress", type=int, default=settings.progress)


def add_mask_options(parser: argparse.ArgumentParser, settings: Settings) -> None:
    parser.add_argument("--dynamic-mask-dir", type=Path, default=settings.dynamic_mask_dir)
    parser.add_argument("--mask-heuristics", action=argparse.BooleanOptionalAction, default=settings.mask_heuristics)
    parser.add_argument("--mask-progress", type=int, default=settings.mask_progress)
    parser.add_argument("--sky-mask", action=argparse.BooleanOptionalAction, default=settings.sky_mask)
    parser.add_argument("--zenith-mask", type=float, default=settings.zenith_mask)
    parser.add_argument("--nadir-mask", type=float, default=settings.nadir_mask)


def add_sam3_options(parser: argparse.ArgumentParser, settings: Settings) -> None:
    parser.add_argument("--sam3-model", type=Path, default=settings.sam3_model)
    parser.add_argument("--sam3-repo", type=Path, default=settings.sam3_repo)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default=settings.dtype)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--sam3-prompts", type=parse_list, default=settings.resolved_sam3_prompts)
    parser.add_argument("--sam3-roi", type=parse_box, action="append", default=[])
    parser.add_argument("--score", type=float, default=settings.score)
    parser.add_argument("--min-area", type=float, default=settings.min_area)
    parser.add_argument("--max-area", type=float, default=settings.max_area)
    parser.add_argument("--dilate", type=int, default=settings.dilate)


def add_pycolmap_path_option(parser: argparse.ArgumentParser, settings: Settings) -> None:
    parser.add_argument("--pycolmap-path", type=Path, default=settings.pycolmap_path)


def add_pycolmap_options(parser: argparse.ArgumentParser, settings: Settings, *, cuda_default: bool) -> None:
    add_pycolmap_path_option(parser, settings)
    parser.add_argument(
        "--require-pycolmap-cuda",
        action=argparse.BooleanOptionalAction,
        default=settings.require_pycolmap_cuda if cuda_default else False,
    )


def add_panorama_sfm_options(parser: argparse.ArgumentParser, settings: Settings) -> None:
    add_pycolmap_options(parser, settings, cuda_default=True)
    parser.add_argument("--panorama-sfm-output", type=Path, default=settings.panorama_sfm_output)
    parser.add_argument(
        "--pano-render-type",
        choices=["perspective_overlapping", "perspective_non_overlapping"],
        default=settings.pano_render_type,
    )
    parser.add_argument(
        "--panorama-virtual-camera-model",
        choices=["pinhole", "simple_pinhole"],
        default=settings.panorama_virtual_camera_model,
    )
    parser.add_argument(
        "--panorama-matcher",
        choices=["sequential", "exhaustive", "vocabtree", "spatial"],
        default=settings.panorama_matcher,
    )
    parser.add_argument(
        "--panorama-mapper",
        choices=["incremental", "global"],
        default=settings.panorama_mapper,
    )
    parser.add_argument(
        "--panorama-ba-backend",
        choices=["ceres", "caspar"],
        default=settings.panorama_ba_backend,
    )
    parser.add_argument(
        "--panorama-loop-detection",
        action=argparse.BooleanOptionalAction,
        default=settings.panorama_loop_detection,
    )
    parser.add_argument("--panorama-vocab-tree-path", type=Path, default=settings.panorama_vocab_tree_path)
    parser.add_argument("--panorama-workers", type=int, default=settings.panorama_workers)
    parser.add_argument(
        "--panorama-use-input-masks",
        action=argparse.BooleanOptionalAction,
        default=settings.panorama_use_input_masks,
    )
    parser.add_argument(
        "--rerender-perspective",
        action=argparse.BooleanOptionalAction,
        default=settings.rerender_perspective,
    )
    parser.add_argument(
        "--rerun-panorama-features",
        action=argparse.BooleanOptionalAction,
        default=settings.rerun_panorama_features,
    )
    parser.add_argument(
        "--rerun-panorama-matching",
        action=argparse.BooleanOptionalAction,
        default=settings.rerun_panorama_matching,
    )
    parser.add_argument(
        "--clean-panorama-sfm",
        action=argparse.BooleanOptionalAction,
        default=settings.clean_panorama_sfm,
    )


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pano-3dgs")
    add_config_option(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract")
    add_config_option(p)
    add_common_run(p)
    add_extract_options(p, settings)
    p.set_defaults(func=extract_sharpest)

    p = sub.add_parser("sam3")
    add_config_option(p)
    add_common_run(p)
    add_sam3_options(p, settings)
    add_mask_options(p, settings)
    p.add_argument(
        "--colmap-masks",
        dest="sam3_colmap_masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also write merged COLMAP masks after SAM3 dynamic masks",
    )
    p.set_defaults(func=run_sam3, sam3_colmap_masks=True)

    p = sub.add_parser("masks")
    add_config_option(p)
    add_common_run(p)
    add_mask_options(p, settings)
    p.set_defaults(func=make_colmap_masks)

    p = sub.add_parser("panorama-sfm")
    add_config_option(p)
    add_common_run(p)
    add_sfm_options(p, settings)
    add_panorama_sfm_options(p, settings)
    p.set_defaults(func=run_panorama_sfm_workflow)

    p = sub.add_parser("run")
    add_config_option(p)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--scene")
    p.add_argument("--runs-dir", type=Path, default=settings.runs_dir)
    p.add_argument("--rate-hz", type=float, default=settings.rate_hz)
    p.add_argument("--equirect-width", type=int, default=settings.equirect_width)
    p.add_argument("--equirect-height", type=int, default=settings.equirect_height)
    add_extract_options_no_video(p, settings)
    add_mask_options(p, settings)
    add_sam3_options(p, settings)
    p.add_argument("--skip-sam3", action="store_true", default=settings.skip_sam3)
    add_sfm_options(p, settings)
    add_panorama_sfm_options(p, settings)
    p.set_defaults(func=run_all)

    return parser


def main(argv: list[str] | None = None) -> None:
    settings = load_settings(find_cli_config(argv))
    parser = build_parser(settings)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
