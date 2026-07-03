from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pano_3dgs.defaults import DEFAULT_SAM3_REPO, DYNAMIC_PROMPTS
from pano_3dgs.utils import ensure_dir


def load_official_sam3(args: argparse.Namespace):
    import torch

    sam3_repo = DEFAULT_SAM3_REPO.resolve()
    if not (sam3_repo / "sam3").is_dir():
        raise SystemExit(f"SAM3 submodule package not found: {sam3_repo}")
    sys.path.insert(0, str(sam3_repo))
    existing = sys.modules.get("sam3")
    if existing is not None:
        existing_origin = Path(getattr(existing, "__file__", "")).resolve()
        if not existing_origin.is_relative_to(sam3_repo):
            for name in list(sys.modules):
                if name == "sam3" or name.startswith("sam3."):
                    del sys.modules[name]
    try:
        import sam3
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise SystemExit(
            "Official SAM3 backend could not be imported. "
            "Install facebookresearch/sam3 dependencies first. "
            f"Original import error: {exc}"
        ) from exc
    sam3_origin = Path(sam3.__file__).resolve()
    if not sam3_origin.is_relative_to(sam3_repo):
        raise SystemExit(f"SAM3 imported from {sam3_origin}, expected submodule under {sam3_repo}")

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
    if getattr(args, "sam3_colmap_masks", False):
        make_colmap_masks(args)

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
