# pano-3dgs

[English](README.md) | [简体中文](README.zh-CN.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![uv](https://img.shields.io/badge/package%20manager-uv-6f42c1)](#)
[![PyCOLMAP](https://img.shields.io/badge/SfM-PyCOLMAP-2f6f6f)](#)
[![SAM3](https://img.shields.io/badge/masks-SAM3-111827)](#)

Convert pre-stitched 360 equirectangular videos into COLMAP datasets for
3D Gaussian Splatting tools. The default workflow reconstructs directly with
COLMAP `EQUIRECTANGULAR` cameras, then exports a perspective `PINHOLE` dataset
for importers that require standard pinhole images.

**Tags:** `360-video` `equirectangular` `3dgs` `colmap` `pycolmap` `sam3`
`caspar-ba` `equirect-sfm` `panorama-sfm`

```text
360 MP4
  -> sharp panorama frames
  -> SAM3 dynamic masks
  -> EQUIRECTANGULAR PyCOLMAP SfM
  -> exported PINHOLE COLMAP dataset for 3DGS import
```

The recommended path is fully PyCOLMAP-based. The older COLMAP binary workflow
has been removed from the CLI.

See [workflow_summary.md](docs/workflow_summary.md) for the Chinese workflow
summary, backend choices, output layout, and limitations.

## Features

- Extracts sharp frames from 360 videos with fixed-rate temporal windows.
- Uses official SAM3 from the `third_party/sam3` submodule for dynamic masks.
- Runs PyCOLMAP feature extraction, matching, and mapping on `EQUIRECTANGULAR`
  panoramas.
- Exports the equirectangular reconstruction to an overlapping perspective
  `PINHOLE` dataset for standard 3DGS importers.
- Still supports the older perspective-rig `panorama-sfm` workflow.
- Supports Caspar bundle adjustment when the installed PyCOLMAP wheel exposes it.
- Writes both binary and text COLMAP models.
- Uses TOML config with CLI overrides.

## Requirements

- Linux or Windows with Python 3.10+.
- `uv` for dependency and CLI execution.
- CUDA-capable PyCOLMAP for the recommended workflow.
- NVIDIA runtime packages compatible with the installed Torch / PyCOLMAP wheels.
- Optional but recommended: SAM3 weights for dynamic object masking.

Torch is pinned for SAM3 on Linux x86_64 and Windows, using the PyTorch cu128
wheel index by default:

```text
torch==2.10.0
torchvision==0.25.0
index = https://download.pytorch.org/whl/cu128
```

## Quick Start

```bash
cd /path/to/pano-3dgs
git submodule update --init --recursive
uv sync
cp pano3dgs.example.toml pano3dgs.toml
```

On Windows PowerShell:

```powershell
cd D:\path\to\pano-3dgs
git submodule update --init --recursive
uv sync
Copy-Item pano3dgs.example.toml pano3dgs.toml
```

`uv sync` installs a PyCOLMAP wheel that matches your Python version, platform,
and CUDA runtime from the
[COLMAP Build v4.1.0 release](https://github.com/lyehe/build_gpu_colmap/releases/tag/v4.1.0).
On Windows this uses the `cuda.cudss` `win_amd64` wheel. On Linux x86_64 this
uses the `cu128.bundled.cudss` `manylinux_2_35_x86_64` wheel.

To inspect or reinstall the wheel selected for the current interpreter:

```bash
uv run pano-3dgs install-pycolmap
```

To only print the selected wheel URL:

```bash
uv run pano-3dgs install-pycolmap --print-only
```

The same release also provides COLMAP 4.1.0 archives and matching `pycolmap`
wheels. You can still install a wheel manually:

```bash
uv pip install /path/to/pycolmap-*.whl
```

On Windows:

```powershell
uv pip install C:\path\to\pycolmap-*.whl
```

Download SAM3 weights. The official model page is
[Hugging Face facebook/sam3](https://huggingface.co/facebook/sam3). For
[ModelScope facebook/sam3](https://www.modelscope.cn/models/facebook/sam3/summary),
use:

```bash
scripts/download_sam3_modelscope.sh
```

On Windows:

```powershell
.\scripts\download_sam3_modelscope.ps1
```

The script downloads to the default local weight directory:
`models/facebook/sam3`.

Run the full workflow:

```bash
uv run pano-3dgs run --video /path/to/video.mp4
```

By default, `/path/to/my_scene.mp4` writes to:

```text
runs/my_scene_2hz_7680/
```

## Configuration

Edit `pano3dgs.toml` for local paths and defaults. The CLI automatically finds
the nearest `pano3dgs.toml` or `pano-3dgs.toml`.

Priority:

```text
CLI flags > TOML config > built-in defaults
```

Important defaults:

```toml
[sam3]
model = "models/facebook/sam3"
prompts = ["sky"]

[pycolmap]
require_cuda = true

[sfm]
feature_type = "sift"
feature_matcher = "auto"
max_features = 12000

[equirect_sfm]
matcher = "sequential"
mapper = "incremental"
ba_backend = "caspar"

[pinhole_3dgs]
enabled = true
render_type = "perspective_overlapping"
workers = 0  # auto-select from CPU threads and available memory

[panorama_sfm]
render_type = "perspective_overlapping"
virtual_camera_model = "pinhole"
matcher = "sequential"
mapper = "incremental"
ba_backend = "caspar"
workers = 0
```

Use `--config path/to/file.toml` to select a specific config file.

To test direct equirectangular SfM with ALIKED + LightGlue, provide the model
files explicitly:

```bash
uv run pano-3dgs equirect-sfm \
  --run "$RUN" \
  --feature-type aliked_n16rot \
  --feature-matcher aliked_lightglue \
  --aliked-model-path models/colmap/aliked-n16rot.onnx \
  --aliked-matcher-model-path models/colmap/aliked-lightglue.onnx \
  --max-features 8192 \
  --rerun-equirect-features
```

## Commands

### One Command

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4
```

By default this runs `extract -> sam3/masks -> equirect-sfm ->
export-pinhole-3dgs`. Use `--sfm-workflow panorama` to run the older
perspective-rig SfM route instead.

Useful overrides:

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4 \
  --scene custom_scene \
  --rate-hz 3 \
  --gpu-index 1
```

Reuse existing dynamic masks:

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4 \
  --dynamic-mask-dir /path/to/dynamic_masks
```

Skip SAM3:

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4 \
  --skip-sam3
```

### Step by Step

```bash
VIDEO=/path/to/video.mp4
RUN=runs/$(basename "$VIDEO" .mp4)_2hz_7680
```

Extract sharp panorama frames:

```bash
uv run pano-3dgs extract \
  --video "$VIDEO" \
  --run "$RUN"
```

Run SAM3 and write merged COLMAP masks:

```bash
uv run pano-3dgs sam3 \
  --run "$RUN"
```

Run direct equirectangular SfM:

```bash
uv run pano-3dgs equirect-sfm \
  --run "$RUN" \
  --clean-equirect-sfm
```

Export a standard perspective `PINHOLE` dataset for 3DGS import:

```bash
uv run pano-3dgs export-pinhole-3dgs \
  --run "$RUN" \
  --clean-pinhole-3dgs
```

Run the older perspective panorama SfM route:

```bash
uv run pano-3dgs panorama-sfm \
  --run "$RUN" \
  --clean-panorama-sfm
```

With explicit Caspar settings:

```bash
uv run pano-3dgs panorama-sfm \
  --run "$RUN" \
  --clean-panorama-sfm \
  --pano-render-type perspective_overlapping \
  --panorama-virtual-camera-model pinhole \
  --panorama-ba-backend caspar \
  --panorama-mapper incremental \
  --panorama-matcher sequential
```

## Output

Default run layout:

```text
runs/<scene>_2hz_7680/
  frames/
  dynamic_masks/
  dynamic_mask_debug/
  colmap_masks/
  equirect_sfm/
    images/
    masks/
    database.db
    sparse/0/
    sparse_txt/0/
  pinhole_3dgs/
    images/
    masks/
    sparse/0/
    sparse_txt/0/
  panorama_sfm/
    images/
    masks/
    database.db
    sparse/0/
    sparse_txt/0/
    sparse_equirectangular/0/
    sparse_equirectangular_txt/0/
```

Use these paths for standard 3DGS import:

```text
pinhole_3dgs/images/
pinhole_3dgs/sparse/0/
```

`equirect_sfm/sparse/0` keeps the direct panorama reconstruction. Use it only
with tools that support COLMAP's `EQUIRECTANGULAR` camera model.

`panorama_sfm/` is written only when running `panorama-sfm` directly or when
`run --sfm-workflow panorama` is selected.

## Masks

All masks use COLMAP convention:

```text
white = keep
black = ignore
```

`sam3` writes `dynamic_masks/` and, by default, also writes merged
`colmap_masks/`. The standalone `masks` command is also available:

```bash
uv run pano-3dgs masks --run "$RUN"
```

When an SfM or PINHOLE export command has input masks enabled, every selected
panorama must have a corresponding file in `colmap_masks/`. Missing masks are
reported as an error instead of silently running those frames without a mask.

`[masks].heuristics = "auto"` means: use SAM3 masks when present, and use
heuristic sky / zenith / nadir masks only for frames without dynamic masks.

## PyCOLMAP Notes

Caspar only applies to the incremental mapper. Keep:

```toml
[equirect_sfm]
mapper = "incremental"

[panorama_sfm]
mapper = "incremental"
virtual_camera_model = "pinhole"
```

The public `pycolmap-cuda12` wheel may expose the `CASPAR` enum without being
compiled with Caspar support. The CLI checks this early and exits with a clear
error.

Loop detection is off by default to avoid runtime vocabulary-tree downloads. If
you need it:

```bash
uv run pano-3dgs panorama-sfm \
  --run "$RUN" \
  --panorama-loop-detection \
  --panorama-vocab-tree-path /path/to/vocab_tree_faiss_flickr100K_words256K.bin
```

## Rebuild Controls

Existing perspective images, masks, features, and matches are reused when
possible. Database reuse validates both the camera model and the complete image
set. Perspective outputs also store `render_config.json`; changing the render
geometry or input-mask mode triggers an automatic rerender.

```bash
uv run pano-3dgs equirect-sfm --run "$RUN" --rerun-equirect-features
uv run pano-3dgs equirect-sfm --run "$RUN" --rerun-equirect-matching
uv run pano-3dgs equirect-sfm --run "$RUN" --clean-equirect-sfm
uv run pano-3dgs export-pinhole-3dgs --run "$RUN" --rerender-pinhole
uv run pano-3dgs export-pinhole-3dgs --run "$RUN" --clean-pinhole-3dgs
uv run pano-3dgs panorama-sfm --run "$RUN" --rerender-perspective
uv run pano-3dgs panorama-sfm --run "$RUN" --rerun-panorama-features
uv run pano-3dgs panorama-sfm --run "$RUN" --rerun-panorama-matching
uv run pano-3dgs panorama-sfm --run "$RUN" --clean-panorama-sfm
```

`--rerun-*-matching` preserves existing features and clears only match tables.
Use `--rerun-*-features` after changing the feature model or feature masks, and
`--rerun-*-matching` after changing matcher settings. Use `--clean-*` only when
the complete generated workspace should be rebuilt.

## Troubleshooting

If `uv` cannot write to its cache directory in a restricted environment:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
```

On Windows PowerShell:

```powershell
$env:UV_CACHE_DIR = "$env:TEMP\uv-cache"
```

If `import torch` fails with `libcudnn.so.9`, install NVIDIA runtime packages
matching the pinned Torch build.

If PyCOLMAP reports no CUDA device, check driver visibility and whether the
installed wheel is actually CUDA-enabled.
