# pano-3dgs

`pano-3dgs` converts pre-stitched 360 equirectangular video into a standard
PINHOLE COLMAP dataset suitable for common 3DGS tools such as LichtFeld Studio.

Pipeline:

```text
360 video
  -> sharp equirectangular frames + SAM3 masks
  -> COLMAP masks
  -> COLMAP EQUIRECTANGULAR SfM
  -> cubemap PINHOLE COLMAP dataset
  -> standard 3DGS / LichtFeld Studio
```

## Setup

Use `uv` from this project directory:

```bash
cd /home/invs/research/pano-3dgs
uv sync
```

Create local defaults:

```bash
cp .env.example .env
```

Edit `.env` for your machine. Common keys:

```bash
PANO3DGS_COLMAP=/home/invs/repos/colmap/build_cuda/src/colmap/exe/colmap
PANO3DGS_RUNS_DIR=runs
PANO3DGS_GPU_INDEX=0
PANO3DGS_SAM3_MODEL=/home/invs/research/ava360_3dgs/models/facebook_sam3
PANO3DGS_SAM3_REPO=/tmp/sam3-official
PANO3DGS_DEVICE=cuda:0
PANO3DGS_SAM3_PROMPTS=person,people,camera,tripod,selfie stick,phone
PANO3DGS_SKIP_SAM3=false
PANO3DGS_MASK_HEURISTICS=auto
PANO3DGS_MASK_PROGRESS=50
PANO3DGS_CLEAN_COLMAP=false
PANO3DGS_FACE_SIZE=2048
PANO3DGS_FACES=all
PANO3DGS_CUBEMAP_WORKERS=0
```

The CLI automatically loads the nearest `.env` from the current directory or a
parent directory. Command-line flags override `.env` values.

Optional SAM3 runtime needs the official SAM3 checkout at the path referenced by
`pyproject.toml`. Dependencies are installed by `uv sync`:

```bash
git clone https://github.com/facebookresearch/sam3.git /tmp/sam3-official
uv sync
```

## One-Command Flow

From video to final cubemap dataset, with SAM3 enabled by default from `.env`:

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4 \
  --scene xianjin_cofe
```

If SAM3 dynamic masks already exist, reuse them:

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4 \
  --scene xianjin_cofe \
  --dynamic-mask-dir /path/to/dynamic_masks
```

If you intentionally want to skip SAM3:

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4 \
  --scene xianjin_cofe \
  --skip-sam3
```

You can still override any `.env` default:

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4 \
  --scene xianjin_cofe \
  --rate-hz 3 \
  --face-size 1536 \
  --gpu-index 1
```

Default output:

```text
runs/<scene>_2hz_7680/
  frames/
  dynamic_masks/
  dynamic_mask_debug/
  colmap_masks/
  colmap_cli_shared/
  cubemap_2048_6faces/
    images/
    masks/
    sparse/0/
    sparse_txt/
```

## Step Commands

Extract sharp frames:

```bash
uv run pano-3dgs extract \
  --video videos/xianjin_cofe.mp4 \
  --run runs/xianjin_cofe_2hz_7680
```

Run SAM3 dynamic masking:

```bash
uv run pano-3dgs sam3 \
  --run runs/xianjin_cofe_2hz_7680
```

Build final COLMAP masks:

```bash
uv run pano-3dgs masks \
  --run runs/xianjin_cofe_2hz_7680
```

By default this uses SAM3 masks when available and only falls back to heuristic
sky / zenith / nadir masks when no SAM3 mask exists for a frame.

Run COLMAP equirectangular SfM:

```bash
uv run pano-3dgs colmap \
  --run runs/xianjin_cofe_2hz_7680
```

For a clean rebuild after changing masks:

```bash
uv run pano-3dgs colmap \
  --run runs/xianjin_cofe_2hz_7680 \
  --clean-colmap
```

Convert to cubemap PINHOLE dataset:

```bash
uv run pano-3dgs cubemap \
  --run runs/xianjin_cofe_2hz_7680
```

## Environment Variables

`.env` keys currently supported:

```text
PANO3DGS_COLMAP
PANO3DGS_RUNS_DIR
PANO3DGS_RATE_HZ
PANO3DGS_EQUIRECT_WIDTH
PANO3DGS_EQUIRECT_HEIGHT
PANO3DGS_SCALE_WIDTH
PANO3DGS_FRAME_JPG_QUALITY
PANO3DGS_FALLBACK_FPS
PANO3DGS_PROGRESS
PANO3DGS_GPU_INDEX
PANO3DGS_THREADS
PANO3DGS_CLEAN_COLMAP
PANO3DGS_MAX_FEATURES
PANO3DGS_OVERLAP
PANO3DGS_SAM3_MODEL
PANO3DGS_SAM3_REPO
PANO3DGS_DEVICE
PANO3DGS_DTYPE
PANO3DGS_SAM3_PROMPTS
PANO3DGS_SKIP_SAM3
PANO3DGS_SAM3_SCORE
PANO3DGS_SAM3_MIN_AREA
PANO3DGS_SAM3_MAX_AREA
PANO3DGS_DILATE
PANO3DGS_DYNAMIC_MASK_DIR
PANO3DGS_MASK_HEURISTICS
PANO3DGS_MASK_PROGRESS
PANO3DGS_SKY_MASK
PANO3DGS_ZENITH_MASK
PANO3DGS_NADIR_MASK
PANO3DGS_FACE_SIZE
PANO3DGS_FACES
PANO3DGS_FOV
PANO3DGS_IMAGE_EXT
PANO3DGS_CUBEMAP_JPG_QUALITY
PANO3DGS_CUBEMAP_WORKERS
PANO3DGS_MASK_NAME_MODE
```

`PANO3DGS_CUBEMAP_WORKERS=0` lets the cubemap converter use `PANO3DGS_THREADS`.
Set it to `1` for serial conversion or to a fixed worker count such as `8`.

`PANO3DGS_MASK_HEURISTICS=auto` means: use SAM3 / dynamic masks when present,
and use heuristic masks only for frames without a dynamic mask. Set it to
`true` to always merge heuristics, or `false` to never use heuristics.

## Mask Convention

All masks use COLMAP convention:

```text
white = keep
black = ignore
```

Cubemap masks are written as `image.jpg.png` by default. Set:

```bash
PANO3DGS_MASK_NAME_MODE=stem
```

to write `image.png`, or:

```bash
PANO3DGS_MASK_NAME_MODE=both
```

to write both naming styles.

## Troubleshooting

If `uv` reports that its cache directory is not writable in a restricted
environment, set a writable cache directory for that shell only:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
```

This is not required on a normal user shell with a writable home directory.
