from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pano_3dgs.utils import ensure_dir, run_cmd


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
