# pano-3dgs

[English](README.md) | [简体中文](README.zh-CN.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![uv](https://img.shields.io/badge/package%20manager-uv-6f42c1)](#)
[![PyCOLMAP](https://img.shields.io/badge/SfM-PyCOLMAP-2f6f6f)](#)
[![SAM3](https://img.shields.io/badge/masks-SAM3-111827)](#)

把已经拼接好的 360 equirectangular MP4 转成适合 3D Gaussian Splatting
工具导入的 COLMAP 数据集。默认路线直接用 COLMAP `EQUIRECTANGULAR`
相机重建，再导出为常见 3DGS importer 更容易接受的 perspective `PINHOLE`
数据集。

**标签：** `360-video` `equirectangular` `3dgs` `colmap` `pycolmap` `sam3`
`caspar-ba` `equirect-sfm` `panorama-sfm`

```text
360 MP4
  -> 清晰全景帧
  -> SAM3 动态物体 mask
  -> EQUIRECTANGULAR PyCOLMAP SfM
  -> 3DGS 可导入的 PINHOLE COLMAP 数据集
```

当前主线完全依赖 PyCOLMAP；旧的 COLMAP binary 工作流已经从 CLI 中移除。

更完整的中文流程说明见 [workflow_summary.md](docs/workflow_summary.md)。

## 功能

- 从 360 视频中按时间窗口抽取清晰帧。
- 使用 `third_party/sam3` submodule 中的官方 SAM3 生成动态物体 mask。
- 直接在 `EQUIRECTANGULAR` 全景相机上运行 PyCOLMAP feature extraction、matching 和 mapping。
- 可把全景重建导出成重叠的 perspective `PINHOLE` 数据集，供标准 3DGS importer 使用。
- 仍然保留旧的 perspective-rig `panorama-sfm` 路线。
- 本地 PyCOLMAP wheel 支持时可使用 Caspar bundle adjustment。
- 同时写出 COLMAP binary 和 text sparse model。
- 使用 TOML 配置，CLI 参数可覆盖配置文件。

## 环境要求

- Linux + Python 3.10+。
- `uv` 用于依赖管理和运行 CLI。
- 推荐工作流需要 CUDA-enabled PyCOLMAP。
- NVIDIA runtime packages 需要匹配 Torch / PyCOLMAP wheel。
- 推荐下载 SAM3 权重用于动态物体 mask。

SAM3 在 Linux x86_64 下固定使用：

```text
torch==2.10.0
torchvision==0.25.0
```

## 快速开始

```bash
cd /path/to/pano-3dgs
git submodule update --init --recursive
uv sync
cp pano3dgs.example.toml pano3dgs.toml
```

安装与本机 Python、平台和 CUDA runtime 匹配的 PyCOLMAP wheel。
[COLMAP Build v4.1.0 release](https://github.com/lyehe/build_gpu_colmap/releases/tag/v4.1.0)
提供 COLMAP 4.1.0 archives 和匹配的 `pycolmap` wheels：

```bash
uv pip install /path/to/pycolmap-*.whl
```

下载 SAM3 权重。官方模型页是
[Hugging Face facebook/sam3](https://huggingface.co/facebook/sam3)。如果使用
[ModelScope facebook/sam3](https://www.modelscope.cn/models/facebook/sam3/summary)，运行：

```bash
scripts/download_sam3_modelscope.sh
```

脚本默认写入：

```text
models/facebook/sam3
```

运行完整流程：

```bash
uv run pano-3dgs run --video /path/to/video.mp4
```

默认输出目录来自视频文件名。例如 `/path/to/my_scene.mp4` 会写到：

```text
runs/my_scene_2hz_7680/
```

## 配置

本地配置文件是 `pano3dgs.toml`。CLI 会自动查找当前目录或父目录中的
`pano3dgs.toml` / `pano-3dgs.toml`。

优先级：

```text
CLI 参数 > TOML 配置 > 内置默认值
```

关键默认值：

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
workers = 0

[panorama_sfm]
render_type = "perspective_overlapping"
virtual_camera_model = "pinhole"
matcher = "sequential"
mapper = "incremental"
ba_backend = "caspar"
```

使用指定配置文件：

```bash
uv run pano-3dgs --config path/to/file.toml run --video /path/to/video.mp4
```

## 命令

### 一条命令

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4
```

默认会执行 `extract -> sam3/masks -> equirect-sfm -> export-pinhole-3dgs`。
如果要走旧的 perspective rig SfM 路线，使用 `--sfm-workflow panorama`。

常用覆盖参数：

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4 \
  --scene custom_scene \
  --rate-hz 3 \
  --gpu-index 1
```

复用已有动态 mask：

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4 \
  --dynamic-mask-dir /path/to/dynamic_masks
```

跳过 SAM3：

```bash
uv run pano-3dgs run \
  --video /path/to/video.mp4 \
  --skip-sam3
```

### 分步运行

```bash
VIDEO=/path/to/video.mp4
RUN=runs/$(basename "$VIDEO" .mp4)_2hz_7680
```

抽取清晰全景帧：

```bash
uv run pano-3dgs extract \
  --video "$VIDEO" \
  --run "$RUN"
```

运行 SAM3，并默认写出合并后的 `colmap_masks/`：

```bash
uv run pano-3dgs sam3 \
  --run "$RUN"
```

运行直接 equirectangular SfM：

```bash
uv run pano-3dgs equirect-sfm \
  --run "$RUN" \
  --clean-equirect-sfm
```

导出标准 perspective `PINHOLE` 3DGS 数据集：

```bash
uv run pano-3dgs export-pinhole-3dgs \
  --run "$RUN" \
  --clean-pinhole-3dgs
```

运行旧的 perspective panorama SfM：

```bash
uv run pano-3dgs panorama-sfm \
  --run "$RUN" \
  --clean-panorama-sfm
```

显式指定 Caspar 配置：

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

## 输出

默认 run 目录结构：

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

标准 3DGS 导入通常使用：

```text
pinhole_3dgs/images/
pinhole_3dgs/sparse/0/
```

`equirect_sfm/sparse/0` 保留直接全景重建结果，主要用于明确支持
COLMAP `EQUIRECTANGULAR` 相机模型的工具。

`panorama_sfm/` 只会在直接运行 `panorama-sfm`，或 `run --sfm-workflow panorama`
时写出。

## Mask

所有 mask 使用 COLMAP 约定：

```text
white = keep
black = ignore
```

`sam3` 会写出 `dynamic_masks/`，并默认写出合并后的 `colmap_masks/`。
也可以单独运行合并命令：

```bash
uv run pano-3dgs masks --run "$RUN"
```

SfM 或 PINHOLE 导出命令启用输入 mask 时，每张选中的全景图都必须在
`colmap_masks/` 中有对应文件。缺失 mask 会直接报错，不会静默改成无 mask 处理。

`[masks].heuristics = "auto"` 表示：优先使用 SAM3 mask；只有缺少动态 mask
的帧才使用 sky / zenith / nadir 启发式 mask。

## PyCOLMAP 注意事项

Caspar 只用于 incremental mapper。保持：

```toml
[panorama_sfm]
mapper = "incremental"
virtual_camera_model = "pinhole"
```

公共 `pycolmap-cuda12` wheel 可能暴露 `CASPAR` enum，但实际没有编译 Caspar。
CLI 会尽早检查并报错。

默认关闭 loop detection，避免运行时下载 vocabulary tree。如果需要开启：

```bash
uv run pano-3dgs panorama-sfm \
  --run "$RUN" \
  --panorama-loop-detection \
  --panorama-vocab-tree-path /path/to/vocab_tree_faiss_flickr100K_words256K.bin
```

## 重建缓存控制

已有 perspective images、masks、features、matches 会尽量复用。数据库复用前会
同时校验相机模型和完整图片集合。Perspective 输出还会记录 `render_config.json`；
修改渲染几何参数或输入 mask 模式时会自动重新渲染。

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

`--rerun-*-matching` 会保留已有 features，只清空 match 表。修改 feature 模型或
feature mask 后使用 `--rerun-*-features`；修改 matcher 设置后使用
`--rerun-*-matching`。只有需要完整重建生成目录时才使用 `--clean-*`。

## 故障排查

如果受限环境中 `uv` cache 不可写：

```bash
export UV_CACHE_DIR=/tmp/uv-cache
```

如果 `import torch` 报 `libcudnn.so.9`，安装与当前 Torch 匹配的 NVIDIA runtime
packages。

如果 PyCOLMAP 看不到 CUDA device，检查驱动可见性，以及安装的 wheel 是否确实启用了 CUDA。
