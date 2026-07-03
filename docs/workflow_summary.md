# pano-3dgs Workflow Summary

This document summarizes the current pipeline, tool choices, backend choices,
and the assumptions behind them.

## Goal

`pano-3dgs` converts a pre-stitched 360 equirectangular MP4 into a COLMAP
dataset that can be imported by standard 3D Gaussian Splatting tools.

The preferred output for current 3DGS tooling is not an equirectangular COLMAP
model. The preferred output is a perspective `PINHOLE` model rendered from the
360 frames, because common 3DGS importers and the Caspar bundle-adjustment path
work reliably with ordinary `PINHOLE` cameras.

## Recommended Workflow

For a new video, the current recommended workflow is:

```bash
VIDEO=/path/to/video.mp4
RUN=runs/$(basename "$VIDEO" .mp4)_2hz_7680

uv run pano-3dgs extract \
  --video "$VIDEO" \
  --run "$RUN" \
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

The `sam3` command now writes both `dynamic_masks/` and merged `colmap_masks/`
by default, so a separate `masks` command is normally unnecessary.

The `run` command is still available, and it now derives the run name from the
MP4 filename if `--scene` is omitted. However, `run` currently finishes with the
older equirectangular COLMAP plus cubemap conversion path. For the current
perspective + Caspar workflow, prefer the explicit `extract -> sam3 ->
panorama-sfm` sequence above.

## Output Layout

A typical run directory looks like this:

```text
runs/<video_stem>_2hz_7680/
  frames/
    sharpest_frames.csv
    *.jpg
  dynamic_masks/
    *.jpg.png
  dynamic_mask_debug/
    *.jpg
  colmap_masks/
    *.jpg.png
  panorama_sfm/
    images/
      pano_camera0/
      ...
      pano_camera11/
    masks/
    database.db
    sparse/0/
    sparse_txt/0/
    sparse_equirectangular/0/
    sparse_equirectangular_txt/0/
```

Use `panorama_sfm/images/` with `panorama_sfm/sparse/0/` for standard 3DGS
import. The `sparse_equirectangular` model is a convenience conversion back to
the original panorama frames for inspection or tooling that explicitly supports
COLMAP's `EQUIRECTANGULAR` camera model.

## Step Details

### 1. Frame Extraction

`extract` opens the MP4 with OpenCV, splits it into fixed time windows, scores
every frame in each window, and writes the sharpest frame per window.

Current default behavior:

- `--window-seconds 0.5`, equivalent to 2 Hz for normal video.
- Sharpness is Laplacian variance on grayscale.
- The sharpness score is computed after optional downscale to
  `--scale-width 1920`.
- The default ROI is `0.0,0.08,1.0,0.92`, ignoring the extreme zenith and nadir
  bands when scoring sharpness.
- Output filenames include sequence index, timestamp, and original frame index.
- `frames/sharpest_frames.csv` records the selected frame metadata and score.

This extraction logic is intentionally simple: it guarantees temporal coverage
and avoids selecting blurry frames, but it does not yet reason about camera pose,
scene overlap, optical flow, or feature baseline.

For reconstruction, this is a reasonable first-stage sampler when the input
video is a smooth walk-through or handheld scan. It does not guarantee that
selected frames are spatially well spaced. If the camera moves very slowly, 2 Hz
can still produce many near-duplicate views. If the camera moves quickly or
turns sharply, 2 Hz may skip useful coverage.

Practical rate choices:

- Normal walking capture: `--window-seconds 0.5` or `run --rate-hz 2`.
- Very slow capture or long static pauses: use 1 Hz or 0.5 Hz.
- Fast motion, turns, clutter, or narrow spaces: use 3 Hz or denser, then prune
  if needed.

A stronger future version should use a two-stage sampler: first select sharp
candidate frames densely, then remove frames that are too visually close using
feature overlap, optical flow, or estimated pose baseline while preserving loop
and coverage points.

### 2. SAM3 Dynamic Masking

`sam3` uses the official SAM3 implementation vendored as a git submodule at
`third_party/sam3`. This avoids relying on a temporary external checkout.

The project dependency points to the local submodule:

```toml
[tool.uv.sources]
sam3 = { path = "third_party/sam3" }
```

SAM3 loads the checkpoint from `PANO3DGS_SAM3_MODEL` or `--sam3-model`.
The expected local model directory can contain `sam3.pt`.

The default dynamic prompts target objects that should not contribute stable
3D structure, such as people, camera rigs, tripods, selfie sticks, and phones.
Local `.env` can extend these prompts. For example, adding `sky` can help when
the sky produces unstable or unwanted features, but it may also remove useful
far-background constraints if the prompt over-segments.

SAM3 writes masks using COLMAP convention:

```text
white = keep
black = ignore
```

It also writes red debug overlays to `dynamic_mask_debug/`.

### 3. COLMAP Mask Merge

`colmap_masks/` are the masks passed to feature extraction.

The mask merge step uses dynamic masks when present. With
`PANO3DGS_MASK_HEURISTICS=auto`, heuristics are used only for frames that do not
have a dynamic SAM3 mask. If heuristics are forced on, they can also mask bright
sky-like top regions plus fixed zenith and nadir bands.

The current default is conservative: prefer SAM3 masks, and avoid adding
heuristic masks unless needed. This keeps more image content available for SfM.

### 4. Perspective Panorama SfM

`panorama-sfm` renders each equirectangular frame into a virtual perspective
camera rig, then runs PyCOLMAP on the perspective images.

The recommended render type is:

```text
perspective_overlapping
```

This renders 12 virtual cameras per panorama:

```text
4 yaw steps x 3 pitch angles = 12 views
```

Each virtual camera has a 90 degree horizontal and vertical field of view.
The recommended camera model is:

```text
PINHOLE
```

`PINHOLE` is preferred over `SIMPLE_PINHOLE` for the current Caspar build,
because this COLMAP/Caspar combination does not support `SIMPLE_PINHOLE` in the
Caspar bundle-adjustment path.

Input panorama masks from `colmap_masks/` are projected into the virtual views
and intersected with the virtual camera masks generated by the panorama renderer.

Feature extraction uses PyCOLMAP with:

- CUDA required by default.
- SIFT `max_num_features` from `--max-features`.
- `CameraMode.PER_FOLDER`, so each virtual camera folder maps to its own camera.
- Rig configuration applied after feature extraction.

Matching defaults to sequential matching:

- `--panorama-matcher sequential`
- `--overlap 25`
- `quadratic_overlap = True`
- `expand_rig_images = True`
- `rig_verification = True`
- `skip_image_pairs_in_same_frame = True`

Loop detection is off by default to avoid PyCOLMAP/COLMAP trying to download a
vocabulary tree at runtime. If loop detection is needed, provide a local
vocabulary tree with `--panorama-vocab-tree-path`.

### 5. Mapper and Bundle Adjustment

The recommended mapper/backend pair is:

```text
--panorama-mapper incremental
--panorama-ba-backend caspar
```

Caspar is used only with the incremental mapper. The global mapper path falls
back to standard Ceres bundle adjustment.

The incremental mapping options intentionally keep the rig and camera intrinsics
fixed:

- `ba_refine_sensor_from_rig = False`
- `ba_refine_focal_length = False`
- `ba_refine_principal_point = False`
- `ba_refine_extra_params = False`

This matches the construction of the virtual panorama rig: every virtual camera
has known intrinsics and known relative pose inside the rig. The unknown is the
pose of each source panorama frame in the world.

Caspar requires the local COLMAP/PyCOLMAP build to expose and support Caspar.
The public wheel can expose enum names without being compiled with real Caspar
support, so the CLI checks this early and reports a clear error.

## Tool and Backend Choices

### `uv`

`uv` manages the Python environment and runs the CLI. It keeps the project
reproducible while still allowing local machine-specific packages, such as the
prebuilt PyCOLMAP wheel, to be installed into the environment.

### Official SAM3 Submodule

SAM3 is vendored as `third_party/sam3` and referenced as a local dependency.
This is more reliable than importing from `/tmp` or another ad hoc checkout.
Fresh clones should run:

```bash
git submodule update --init --recursive
uv sync
```

### CUDA Torch

Torch and torchvision are pinned to CUDA-compatible versions for the Linux x86_64
environment:

```text
torch==2.10.0
torchvision==0.25.0
```

The working environment also needs matching NVIDIA runtime packages, including
cuDNN 9, so that `import torch` can find libraries such as `libcudnn.so.9`.

### COLMAP CLI

The COLMAP binary defaults to:

```text
/home/invs/repos/colmap_prebuild/bin/colmap
```

This is still used for legacy commands and for converting sparse binary models
to text after `panorama-sfm`.

### PyCOLMAP

`panorama-sfm` requires CUDA-enabled PyCOLMAP. In this environment, the intended
wheel comes from the local COLMAP prebuild:

```text
/home/invs/repos/colmap_prebuild/pycolmap-4.1.0+cu128.bundled.cudss-cp311-cp311-manylinux_2_35_x86_64.whl
```

PyCOLMAP is intentionally not pinned in `pyproject.toml`, because the correct
wheel is machine- and CUDA-build-specific.

## Legacy Path

The older commands remain available:

```bash
uv run pano-3dgs colmap --run "$RUN"
uv run pano-3dgs cubemap --run "$RUN"
uv run pano-3dgs run --video "$VIDEO"
```

This path runs equirectangular COLMAP first and then converts to a cubemap
dataset. It is useful for compatibility and experiments, but it is no longer the
recommended route for the current Caspar + standard 3DGS workflow.

## Known Limitations

- Frame extraction is not yet geometry-aware. It may keep frames that are too
  close together or miss coverage during fast motion.
- SAM3 masks depend on prompt quality. Over-broad prompts can remove stable
  structure; under-broad prompts can leave dynamic objects for SfM.
- `panorama-sfm` assumes true 360 panoramas with width equal to twice height.
- Caspar support depends on the actual local COLMAP/PyCOLMAP build, not only on
  Python enum availability.
- A plain `uv sync` can remove machine-specific PyCOLMAP if it was installed
  manually, because PyCOLMAP is not a project dependency.

## Current Recommendation

For production-style reconstruction tests, use:

```text
extract 2 Hz sharp frames
-> SAM3 dynamic masks
-> merged COLMAP masks
-> perspective_overlapping panorama-sfm
-> PINHOLE virtual cameras
-> sequential matching
-> incremental mapper
-> Caspar BA backend when available
```

This gives the current best alignment with common 3DGS importers while keeping
the camera model and bundle-adjustment backend compatible with the local
COLMAP/Caspar build.
