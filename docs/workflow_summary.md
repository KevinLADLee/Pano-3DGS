# pano-3dgs 工作流总结

本文档总结当前 `pano-3dgs` 的整体流程、工具选择、backend 选择，以及这些选择背后的假设。

## 目标

`pano-3dgs` 的目标是把已经拼接好的 360 equirectangular MP4 视频转换成标准 COLMAP 数据集，使其可以被常见的 3D Gaussian Splatting 工具导入。

当前有两条 SfM 路线：

- `equirect-sfm`：直接把 2:1 全景帧作为 COLMAP `EQUIRECTANGULAR` 相机重建。这是无人机 360 全景视频的主测路线。
- `panorama-sfm`：把全景帧渲染成 perspective `PINHOLE` 虚拟 rig 再重建。它更兼容只接受普通 pinhole COLMAP 数据集的 3DGS importer。

## 推荐工作流

对一个新视频，当前推荐流程是：

```bash
VIDEO=/path/to/video.mp4
RUN=runs/$(basename "$VIDEO" .mp4)_2hz_7680

uv run pano-3dgs extract \
  --video "$VIDEO" \
  --run "$RUN" \
  --window-seconds 0.5

uv run pano-3dgs sam3 \
  --run "$RUN"

uv run pano-3dgs equirect-sfm \
  --run "$RUN" \
  --feature-type aliked_n16rot \
  --feature-matcher aliked_lightglue \
  --aliked-model-path models/colmap/aliked-n16rot.onnx \
  --aliked-matcher-model-path models/colmap/aliked-lightglue.onnx \
  --clean-equirect-sfm

uv run pano-3dgs export-pinhole-3dgs \
  --run "$RUN" \
  --clean-pinhole-3dgs
```

如果要退回 ONNX brute-force matcher，应把 `--feature-matcher` 改为 `aliked_bruteforce`，并把 `--aliked-matcher-model-path` 指向 `bruteforce-matcher.onnx`。

旧的 perspective rig 路线仍然保留：

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

现在 `sam3` 命令默认会同时写出 `dynamic_masks/` 和合并后的 `colmap_masks/`，所以通常不需要再单独运行 `masks` 命令。

`run` 命令会执行同一条推荐主线：`extract -> sam3/masks -> SfM`。默认 `--sfm-workflow equirect`，即直接运行 `equirect-sfm`，并在 `[pinhole_3dgs].enabled = true` 时继续导出 `pinhole_3dgs/`；需要旧的 perspective rig 路线时显式传入 `--sfm-workflow panorama`。省略 `--scene` 时，run 目录名来自 MP4 文件名。

## 参数配置

当前推荐用 TOML 管理项目参数：

```bash
cp pano3dgs.example.toml pano3dgs.toml
```

CLI 会自动查找当前目录或父目录里的 `pano3dgs.toml` / `pano-3dgs.toml`，也可以显式传入：

```bash
uv run pano-3dgs --config pano3dgs.toml panorama-sfm --run "$RUN"
```

参数优先级是：

```text
CLI 参数 > TOML 配置 > 内置默认值
```

持久化参数写在 TOML 中。CLI 参数用于单次命令覆盖。

例如把默认的 equirect SfM 参数写入 `pano3dgs.toml` 后，`equirect-sfm` 可以简化为：

```bash
uv run pano-3dgs equirect-sfm \
  --run "$RUN" \
  --clean-equirect-sfm
```

## 输出目录

典型 run 目录如下：

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

标准 3DGS 导入优先使用 `pinhole_3dgs/images/` 和 `pinhole_3dgs/sparse/0/`。`equirect_sfm/sparse/0` 保留直接全景重建结果，主要用于检查，或用于明确支持 COLMAP `EQUIRECTANGULAR` 相机模型的工具。`panorama_sfm/` 只在直接运行 `panorama-sfm`，或 `run --sfm-workflow panorama` 时写出。

## 步骤细节

### 1. 帧抽取

`extract` 使用 OpenCV 打开 MP4，把视频切成固定时间窗口，对每个窗口内的所有帧打分，并写出该窗口里最清晰的一帧。

当前默认行为：

- `--window-seconds 0.5`，相当于普通视频下 2 Hz 采样。
- 清晰度使用灰度图 Laplacian variance。
- 打分前会按需缩放到 `--scale-width 1920`。
- 默认 ROI 是 `0.0,0.08,1.0,0.92`，打分时忽略最顶部 zenith 和最底部 nadir 区域。
- 输出文件名包含序号、时间戳和原始 frame index。
- `frames/sharpest_frames.csv` 记录被选中帧的 metadata 和 score。

这个抽帧逻辑是有意保持简单的：它保证时间上均匀覆盖，并尽量避开模糊帧；但它还不会根据相机位姿、场景 overlap、optical flow 或特征 baseline 做判断。

从三维重建角度看，如果输入视频是平滑行走或手持扫描，这个抽帧策略可以作为第一阶段采样。但它不能保证选出的帧在空间上足够分散。如果相机移动很慢，2 Hz 仍然可能产生很多过近视角；如果相机移动很快或快速转向，2 Hz 也可能漏掉有用覆盖。

实用采样建议：

- 正常行走采集：`--window-seconds 0.5` 或 `run --rate-hz 2`。
- 很慢的采集或长时间静止：使用 1 Hz 或 0.5 Hz。
- 快速移动、转弯、复杂遮挡、狭窄空间：使用 3 Hz 或更密集采样，之后再做 pruning。

更强的后续版本应采用两阶段采样：先密集选出清晰候选帧，再用 feature overlap、optical flow 或估计 pose baseline 去掉过近视角，同时保留 loop 和覆盖关键点。

### 2. SAM3 动态物体 mask

`sam3` 使用官方 SAM3 实现，代码位于 `third_party/sam3` submodule。

项目依赖指向本地 submodule：

```toml
[tool.uv.sources]
sam3 = { path = "third_party/sam3", editable = true }
```

运行时也固定从 `third_party/sam3` 导入 `sam3` package，不再读取 `[sam3].repo` 或 `--sam3-repo`。

SAM3 从 TOML 的 `[sam3].model` 或 `--sam3-model` 加载 checkpoint。默认权重目录是：

```text
models/facebook/sam3
```

`[sam3].model` 可以指向模型目录，也可以直接指向 `sam3.pt`。

官方模型页是 [Hugging Face facebook/sam3](https://huggingface.co/facebook/sam3)。

从 ModelScope 下载时使用项目脚本：

```bash
scripts/download_sam3_modelscope.sh
```

脚本执行的命令是：

```bash
modelscope download --model facebook/sam3 --local_dir models/facebook/sam3
```

ModelScope 页面是 [facebook/sam3](https://www.modelscope.cn/models/facebook/sam3/summary)。

默认 SAM3 prompt 是 `sky`，用于先屏蔽通常不提供稳定几何约束的天空区域。本地 TOML 可以扩展这些 prompts，例如加入人、相机设备、三脚架、自拍杆或手机等动态/采集设备对象。prompt 过宽可能移除有用的远处背景约束，过窄则可能把动态物体留给 SfM。

SAM3 写出的 mask 使用 COLMAP 约定：

```text
white = keep
black = ignore
```

同时会在 `dynamic_mask_debug/` 写出红色 overlay 方便检查。

### 3. COLMAP mask 合并

`colmap_masks/` 是最终传给 feature extraction 的 mask。

mask 合并步骤会优先使用已有 dynamic masks。在 TOML `[masks].heuristics = "auto"` 时，只有缺少 dynamic SAM3 mask 的帧才会使用启发式 mask。如果强制开启 heuristics，则还可以 mask 掉类似明亮天空的顶部区域，以及固定比例的 zenith / nadir 区域。

当前默认策略是保守的：优先使用 SAM3 mask，除非需要 fallback，否则不额外叠加启发式 mask。这样可以保留更多图像内容给 SfM 使用。

### 4. Perspective Panorama SfM

`panorama-sfm` 会把每一张 equirectangular 全景帧渲染成一个虚拟 perspective 相机 rig，然后在这些 perspective 图像上运行 PyCOLMAP。

推荐 render type 是：

```text
perspective_overlapping
```

它会为每张 panorama 渲染 12 个虚拟相机：

```text
4 yaw steps x 3 pitch angles = 12 views
```

每个虚拟相机的水平和垂直视场角都是 90 度。推荐相机模型是：

```text
PINHOLE
```

当前 Caspar build 下优先使用 `PINHOLE`，而不是 `SIMPLE_PINHOLE`。原因是这套 COLMAP/Caspar 组合在 Caspar bundle-adjustment 路径里不支持 `SIMPLE_PINHOLE`。

来自 `colmap_masks/` 的全景输入 mask 会被投影到虚拟视图，并和 panorama renderer 生成的每个虚拟相机 mask 做交集。

Feature extraction 使用 PyCOLMAP，关键设置包括：

- 默认要求 CUDA。
- SIFT `max_num_features` 来自 `--max-features`。
- 使用 `CameraMode.PER_FOLDER`，因此每个虚拟相机文件夹对应一个 camera。
- feature extraction 后再应用 rig configuration。

Matching 默认使用 sequential matching：

- `--panorama-matcher sequential`
- `--overlap 25`
- `quadratic_overlap = True`
- `expand_rig_images = True`
- `rig_verification = True`
- `skip_image_pairs_in_same_frame = True`

默认关闭 loop detection，避免 PyCOLMAP 在运行时尝试下载 vocabulary tree。如果需要 loop detection，应提前准备本地 vocabulary tree，并通过 `--panorama-vocab-tree-path` 显式传入。

### 5. Mapper 和 Bundle Adjustment

推荐 mapper/backend 组合是：

```text
--panorama-mapper incremental
--panorama-ba-backend caspar
```

Caspar 只用于 incremental mapper。global mapper 路径使用标准 Ceres bundle adjustment。

incremental mapping 会有意固定 rig 和 camera intrinsics：

- `ba_refine_sensor_from_rig = False`
- `ba_refine_focal_length = False`
- `ba_refine_principal_point = False`
- `ba_refine_extra_params = False`

这和虚拟 panorama rig 的构造一致：每个虚拟相机的内参和 rig 内相对位姿都是已知的，需要估计的是每个源 panorama frame 在世界坐标系里的 pose。

Caspar 依赖本地 COLMAP/PyCOLMAP build 真实开启并支持 Caspar。公共 wheel 可能暴露 enum 名称，但实际并未用 Caspar 编译，所以 CLI 会尽早检查并给出明确错误。

## 工具和 Backend 选择

### `uv`

`uv` 负责 Python 环境管理和 CLI 运行。它让项目依赖更可复现，同时仍允许把本机相关的包安装进环境，例如本地预编译 PyCOLMAP wheel。

### 官方 SAM3 Submodule

SAM3 以 `third_party/sam3` 形式 vendored 到项目中，并作为 editable 本地 dependency 引用。运行时固定从该 submodule 导入 package。

fresh clone 后应运行：

```bash
git submodule update --init --recursive
uv sync
```

模型权重来源：

- [官方 Hugging Face](https://huggingface.co/facebook/sam3)
- [ModelScope facebook/sam3](https://www.modelscope.cn/models/facebook/sam3/summary)

ModelScope 下载脚本：

```bash
scripts/download_sam3_modelscope.sh
```

脚本默认写入 `models/facebook/sam3`，和 `[sam3].model` 默认值一致。

### CUDA Torch

Torch 和 torchvision 在 Linux x86_64 环境下固定为 CUDA 兼容版本：

```text
torch==2.10.0
torchvision==0.25.0
```

运行环境还需要匹配的 NVIDIA runtime packages，包括 cuDNN 9，否则 `import torch` 可能找不到 `libcudnn.so.9` 等库。

### PyCOLMAP

推荐工作流依赖 PyCOLMAP，不依赖 COLMAP binary。`equirect-sfm`、`panorama-sfm` 和 ALIKED/LightGlue 路径都建议使用 CUDA-enabled PyCOLMAP。COLMAP 4.1.0 prebuild 和匹配的 PyCOLMAP wheels 可以从 [COLMAP Build v4.1.0 release](https://github.com/lyehe/build_gpu_colmap/releases/tag/v4.1.0) 下载。

安装与当前机器 Python 版本、平台、CUDA/runtime 变体匹配的 PyCOLMAP wheel：

```text
/path/to/pycolmap-*.whl
```

PyCOLMAP 有意没有固定在 `pyproject.toml` 中，因为正确的 wheel 和具体机器、CUDA build 绑定。

Windows full build 建议通过项目脚本安装和验证，避免手工补 DLL：

```powershell
.\scripts\setup_pycolmap_full.ps1 `
  -WheelPath D:\Projects\build_gpu_colmap\third_party\colmap-for-pycolmap\wheelhouse\pycolmap-4.1.0-cp312-cp312-win_amd64.whl `
  -NoDeps
```

脚本会安装本地 wheel，尝试把 Torch 自带的 cuDNN 9 runtime DLL 复制到 `pycolmap.libs`，并检查 CUDA、ALIKED、LightGlue 和 Caspar 是否可见。

`equirect-sfm` 的 `images/`、`masks/`、`database.db`、`sparse/0`、`sparse_txt/0` 都由 PyCOLMAP 写出。`panorama-sfm` 额外会写出 `sparse_equirectangular/0` 和 `sparse_equirectangular_txt/0`，用于把 perspective rig 结果转换回全景帧坐标。

## 已知限制

- Frame extraction 还不是 geometry-aware。它可能保留过近的帧，也可能在快速移动时漏掉覆盖。
- SAM3 mask 依赖 prompt 质量。过宽 prompt 可能移除稳定结构；过窄 prompt 可能把动态物体留给 SfM。
- `equirect-sfm` 和 `panorama-sfm` 都假设输入是真 360 panorama，且宽度等于高度的两倍。
- Caspar 支持取决于本地 COLMAP/PyCOLMAP build，而不只是 Python enum 是否存在。
- 直接运行 `uv sync` 可能移除手动安装的、机器相关的 PyCOLMAP，因为 PyCOLMAP 没有作为项目 dependency 固定。

## 当前建议

用于较正式的 reconstruction 测试时，建议使用：

```text
extract 2 Hz sharp frames
-> SAM3 dynamic masks
-> merged COLMAP masks
-> equirect-sfm with EQUIRECTANGULAR cameras
-> optional export-pinhole-3dgs
-> perspective_overlapping PINHOLE virtual cameras
-> sequential matching
-> incremental mapper
-> Caspar BA backend when available
```

这条路线先保留全景相机的原始几何，再导出常见 3DGS importer 更容易消费的 `PINHOLE` 数据集。旧的 `panorama-sfm` 路线仍适合需要直接在 perspective rig 上提特征和建图的兼容场景。
