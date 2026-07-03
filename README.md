# pano-3dgs

`pano-3dgs` converts pre-stitched 360 equirectangular video into a standard
PINHOLE COLMAP dataset suitable for common 3DGS tools such as LichtFeld Studio.

Pipeline:

```text
360 video
  -> sharp equirectangular frames + SAM3 masks
  -> COLMAP masks
  -> perspective PINHOLE panorama SfM
  -> standard 3DGS / LichtFeld Studio
```

The older equirectangular-COLMAP-to-cubemap path is still available through
`colmap` and `cubemap`, but the recommended workflow for Caspar and common 3DGS
tools is `panorama-sfm` with perspective virtual `PINHOLE` cameras.

See [docs/workflow_summary.md](docs/workflow_summary.md) for the current
workflow, backend choices, output layout, and known limitations.

## Setup

Use `uv` from this project directory:

```bash
cd /home/invs/research/pano-3dgs
uv sync
```

Create local defaults with TOML:

```bash
cp pano3dgs.example.toml pano3dgs.toml
```

Edit `pano3dgs.toml` for your machine. The CLI automatically loads the nearest
`pano3dgs.toml` or `pano-3dgs.toml`; you can also pass `--config path.toml`.
Priority is: command-line flags, TOML config, built-in defaults.

Optional SAM3 runtime uses the official SAM3 checkout vendored as a git
submodule. Initialize submodules before syncing dependencies:

```bash
git submodule update --init --recursive
uv sync
```

### Prebuilt COLMAP / PyCOLMAP

For the Caspar panorama workflow, use the matching prebuilt COLMAP repo and its
PyCOLMAP wheel. The local prebuild expected by the defaults is:

```text
/home/invs/repos/colmap_prebuild/
  bin/colmap
  lib/
  pycolmap-4.1.0+cu128.bundled.cudss-cp311-cp311-manylinux_2_35_x86_64.whl
```

If you start from the release archive, unpack it to that path:

```bash
mkdir -p /home/invs/repos
unzip COLMAP-4.1.0-ubuntu-22.04-CUDA-cuDSS-Caspar.zip -d /home/invs/repos/colmap_prebuild
```

Then point the CLI at the prebuilt binary in `pano3dgs.toml`:

```toml
[paths]
colmap = "/home/invs/repos/colmap_prebuild/bin/colmap"
```

Install the bundled PyCOLMAP wheel into this project's uv environment. PyCOLMAP
is intentionally not pinned in `pyproject.toml`, because the CUDA/Caspar wheel is
machine-specific. On the current Linux/Python 3.11 setup:

```bash
uv pip install /home/invs/repos/colmap_prebuild/pycolmap-4.1.0+cu128.bundled.cudss-cp311-cp311-manylinux_2_35_x86_64.whl
```

Verify both pieces before running SfM:

```bash
/home/invs/repos/colmap_prebuild/bin/colmap version
uv run python - <<'PY'
import pycolmap
print(pycolmap.__version__, "cuda=", pycolmap.has_cuda)
print("cuda devices=", pycolmap.get_num_cuda_devices())
print("caspar=", hasattr(pycolmap, "CasparBundleAdjustmentOptions"))
PY
```

The prebuild used here is COLMAP 4.1.0 for Ubuntu 22.04 with CUDA 12.8, cuDSS,
and Caspar enabled. Caspar currently works with the rendered `PINHOLE` virtual
cameras used by `panorama-sfm`; do not switch it to `SIMPLE_PINHOLE` when using
`--panorama-ba-backend caspar`.

Torch is pinned to a CUDA 12 compatible pair (`torch==2.10.0`,
`torchvision==0.25.0`) for SAM3.

## One-Command Flow

From video to final cubemap dataset, with SAM3 configured in `pano3dgs.toml`:

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4
```

By default, the run directory is named from the video filename stem, for example
`/path/to/xianjin_cofe.mp4` writes to `runs/xianjin_cofe_2hz_7680`.

If SAM3 dynamic masks already exist, reuse them:

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4 \
  --dynamic-mask-dir /path/to/dynamic_masks
```

If you intentionally want to skip SAM3:

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4 \
  --skip-sam3
```

You can still override TOML defaults for a single command:

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4 \
  --scene custom_scene_name \
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

Run the recommended perspective panorama SfM workflow:

```bash
uv run pano-3dgs panorama-sfm \
  --run runs/xianjin_cofe_2hz_7680 \
  --clean-panorama-sfm \
  --pano-render-type perspective_overlapping \
  --panorama-virtual-camera-model pinhole
```

To try COLMAP's experimental Caspar GPU bundle-adjustment backend for the
incremental mapper, use the prebuilt PyCOLMAP wheel above and keep the virtual
camera model as `pinhole`:

```bash
uv run pano-3dgs panorama-sfm \
  --run runs/xianjin_cofe_2hz_7680 \
  --clean-panorama-sfm \
  --pano-render-type perspective_overlapping \
  --panorama-virtual-camera-model pinhole \
  --panorama-ba-backend caspar \
  --panorama-mapper incremental \
  --panorama-matcher sequential
```

Caspar is only used for `--panorama-mapper incremental`. It supports this
workflow because the rendered virtual cameras are `PINHOLE` and the rig sensor
poses are fixed. The public `pycolmap-cuda12` wheel may expose the `CASPAR`
enum while still being compiled without Caspar support; in that case the command
will fail early and ask you to build PyCOLMAP from source with Caspar enabled.

By default, sequential matching does not run vocabulary-tree loop detection.
This avoids COLMAP/PyCOLMAP trying to download
`vocab_tree_faiss_flickr100K_words256K.bin` at runtime. If loop detection is
needed, download the vocabulary tree yourself and pass it explicitly:

```bash
uv run pano-3dgs panorama-sfm \
  --run runs/xianjin_cofe_2hz_7680 \
  --panorama-loop-detection \
  --panorama-vocab-tree-path /path/to/vocab_tree_faiss_flickr100K_words256K.bin
```

This renders each equirectangular frame into the COLMAP example's default
`perspective_overlapping` rig: 4 yaw steps x 3 pitch angles = 12 overlapping
virtual PINHOLE cameras. `colmap_masks/` are projected into those views and
intersected with COLMAP's per-virtual-camera masks. The 3DGS-friendly output is:

```text
runs/<scene>_2hz_7680/panorama_sfm/
  images/
  masks/
  database.db
  sparse/0/
  sparse_txt/0/
  sparse_equirectangular/0/
  sparse_equirectangular_txt/0/
```

Use `panorama_sfm/images` with `panorama_sfm/sparse/0` for PINHOLE 3DGS import.
The `sparse_equirectangular` model maps the reconstruction back to the original
panorama frames and is mainly useful for inspection or tools that support
COLMAP's `EQUIRECTANGULAR` camera model.

Existing perspective images and masks are reused by default. To force rendering
again after changing masks or render settings, use:

```bash
uv run pano-3dgs panorama-sfm \
  --run runs/xianjin_cofe_2hz_7680 \
  --rerender-perspective
```

The feature/match database is also reused by default. This means you can switch
mapper settings, for example trying Caspar, without re-rendering images or
re-running feature extraction / matching:

```bash
uv run pano-3dgs panorama-sfm \
  --run runs/xianjin_cofe_2hz_7680 \
  --panorama-ba-backend caspar
```

Use `--rerun-panorama-features` or `--rerun-panorama-matching` when you change
image, mask, feature, matching, or rig settings. `--clean-panorama-sfm` removes
the whole panorama SfM workspace and recomputes everything.

For a new video with the current perspective + Caspar + PINHOLE workflow, run
the frame/mask steps first and then run `panorama-sfm`:

```bash
RUN=runs/xianjin_cofe_2hz_7680
VIDEO=/home/invs/datasets/avata360_data/xianjin_cofe.mp4

uv run pano-3dgs extract \
  --run "$RUN" \
  --video "$VIDEO" \
  --window-seconds 0.5

uv run pano-3dgs sam3 \
  --run "$RUN"

uv run pano-3dgs panorama-sfm \
  --run "$RUN" \
  --clean-panorama-sfm \
  --pano-render-type perspective_overlapping \
  --panorama-virtual-camera-model pinhole \
  --panorama-ba-backend caspar \
  --panorama-mapper incremental \
  --panorama-matcher sequential
```

With the recommended values in `pano3dgs.toml`, the final command can be:

```bash
uv run pano-3dgs panorama-sfm --run "$RUN" --clean-panorama-sfm
```

## Configuration

The local configuration file is `pano3dgs.toml`. See
[`pano3dgs.example.toml`](pano3dgs.example.toml) for all supported sections.
The CLI automatically searches the current directory and parent directories for
`pano3dgs.toml` or `pano-3dgs.toml`.

Use `--config path/to/file.toml` to run with a specific config file. Command-line
flags override TOML values for that invocation only.

`[cubemap].workers = 0` lets the cubemap converter use `[colmap].threads`. Set
it to `1` for serial conversion or to a fixed worker count such as `8`.

`[masks].heuristics = "auto"` means: use SAM3 / dynamic masks when present, and
use heuristic masks only for frames without a dynamic mask. Set it to `true` to
always merge heuristics, or `false` to never use heuristics.

## Mask Convention

All masks use COLMAP convention:

```text
white = keep
black = ignore
```

Cubemap masks are written as `image.jpg.png` by default. Set:

```toml
[cubemap]
mask_name_mode = "stem"
```

to write `image.png`, or:

```toml
[cubemap]
mask_name_mode = "both"
```

to write both naming styles.

## Troubleshooting

If `uv` reports that its cache directory is not writable in a restricted
environment, set a writable cache directory for that shell only:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
```

This is not required on a normal user shell with a writable home directory.
