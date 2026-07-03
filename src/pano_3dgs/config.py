from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - project currently runs on Python 3.11 locally.
    import tomli as tomllib

from pano_3dgs.defaults import DEFAULT_COLMAP, DEFAULT_FACES, DEFAULT_SAM3_REPO
from pano_3dgs.utils import parse_faces, parse_list


@dataclass(frozen=True)
class Settings:
    colmap: Path = Path(DEFAULT_COLMAP)
    runs_dir: Path = Path("runs")
    rate_hz: float = 2.0
    equirect_width: int = 7680
    equirect_height: int = 3840

    window_seconds: float = 0.5
    scale_width: int = 1920
    roi: tuple[float, float, float, float] = (0.0, 0.08, 1.0, 0.92)
    fallback_fps: float = 30.0
    frame_jpg_quality: int = 98
    progress: int = 300

    gpu_index: str = "0"
    threads: int = 8
    clean_colmap: bool = False
    max_features: int = 12000
    overlap: int = 25

    sam3_model: Path | None = None
    sam3_repo: Path | None = DEFAULT_SAM3_REPO
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    sam3_prompts: list[str] | None = None
    skip_sam3: bool = False
    score: float = 0.35
    min_area: float = 0.00005
    max_area: float = 0.80
    dilate: int = 7

    dynamic_mask_dir: Path | None = None
    mask_heuristics: bool | None = None
    mask_progress: int = 50
    sky_mask: bool = True
    zenith_mask: float = 0.04
    nadir_mask: float = 0.04

    face_size: int = 2048
    faces: list[str] | None = None
    fov: float = 90.0
    image_ext: str = "jpg"
    cubemap_jpg_quality: int = 95
    cubemap_workers: int = 0
    mask_name_mode: str = "colmap"

    pycolmap_path: Path | None = None
    require_pycolmap_cuda: bool = True
    panorama_sfm_output: Path | None = None
    pano_render_type: str = "perspective_overlapping"
    panorama_virtual_camera_model: str = "pinhole"
    panorama_matcher: str = "sequential"
    panorama_mapper: str = "incremental"
    panorama_ba_backend: str = "ceres"
    panorama_loop_detection: bool = False
    panorama_vocab_tree_path: Path | None = None
    panorama_workers: int = 0
    panorama_use_input_masks: bool = True
    rerender_perspective: bool = False
    rerun_panorama_features: bool = False
    rerun_panorama_matching: bool = False
    clean_panorama_sfm: bool = False

    @property
    def resolved_faces(self) -> list[str]:
        return list(self.faces or DEFAULT_FACES)

    @property
    def resolved_sam3_prompts(self) -> list[str]:
        return list(self.sam3_prompts or [])


ENV_FIELDS: dict[str, str] = {
    "PANO3DGS_COLMAP": "colmap",
    "PANO3DGS_RUNS_DIR": "runs_dir",
    "PANO3DGS_RATE_HZ": "rate_hz",
    "PANO3DGS_EQUIRECT_WIDTH": "equirect_width",
    "PANO3DGS_EQUIRECT_HEIGHT": "equirect_height",
    "PANO3DGS_WINDOW_SECONDS": "window_seconds",
    "PANO3DGS_SCALE_WIDTH": "scale_width",
    "PANO3DGS_FALLBACK_FPS": "fallback_fps",
    "PANO3DGS_FRAME_JPG_QUALITY": "frame_jpg_quality",
    "PANO3DGS_PROGRESS": "progress",
    "PANO3DGS_GPU_INDEX": "gpu_index",
    "PANO3DGS_THREADS": "threads",
    "PANO3DGS_CLEAN_COLMAP": "clean_colmap",
    "PANO3DGS_MAX_FEATURES": "max_features",
    "PANO3DGS_OVERLAP": "overlap",
    "PANO3DGS_SAM3_MODEL": "sam3_model",
    "PANO3DGS_SAM3_REPO": "sam3_repo",
    "PANO3DGS_DEVICE": "device",
    "PANO3DGS_DTYPE": "dtype",
    "PANO3DGS_SAM3_PROMPTS": "sam3_prompts",
    "PANO3DGS_SKIP_SAM3": "skip_sam3",
    "PANO3DGS_SAM3_SCORE": "score",
    "PANO3DGS_SAM3_MIN_AREA": "min_area",
    "PANO3DGS_SAM3_MAX_AREA": "max_area",
    "PANO3DGS_DILATE": "dilate",
    "PANO3DGS_DYNAMIC_MASK_DIR": "dynamic_mask_dir",
    "PANO3DGS_MASK_HEURISTICS": "mask_heuristics",
    "PANO3DGS_MASK_PROGRESS": "mask_progress",
    "PANO3DGS_SKY_MASK": "sky_mask",
    "PANO3DGS_ZENITH_MASK": "zenith_mask",
    "PANO3DGS_NADIR_MASK": "nadir_mask",
    "PANO3DGS_FACE_SIZE": "face_size",
    "PANO3DGS_FACES": "faces",
    "PANO3DGS_FOV": "fov",
    "PANO3DGS_IMAGE_EXT": "image_ext",
    "PANO3DGS_CUBEMAP_JPG_QUALITY": "cubemap_jpg_quality",
    "PANO3DGS_CUBEMAP_WORKERS": "cubemap_workers",
    "PANO3DGS_MASK_NAME_MODE": "mask_name_mode",
    "PANO3DGS_PYCOLMAP_PATH": "pycolmap_path",
    "PANO3DGS_REQUIRE_PYCOLMAP_CUDA": "require_pycolmap_cuda",
    "PANO3DGS_PANORAMA_SFM_OUTPUT": "panorama_sfm_output",
    "PANO3DGS_PANO_RENDER_TYPE": "pano_render_type",
    "PANO3DGS_PANORAMA_VIRTUAL_CAMERA_MODEL": "panorama_virtual_camera_model",
    "PANO3DGS_PANORAMA_MATCHER": "panorama_matcher",
    "PANO3DGS_PANORAMA_MAPPER": "panorama_mapper",
    "PANO3DGS_PANORAMA_BA_BACKEND": "panorama_ba_backend",
    "PANO3DGS_PANORAMA_LOOP_DETECTION": "panorama_loop_detection",
    "PANO3DGS_PANORAMA_VOCAB_TREE_PATH": "panorama_vocab_tree_path",
    "PANO3DGS_PANORAMA_WORKERS": "panorama_workers",
    "PANO3DGS_PANORAMA_USE_INPUT_MASKS": "panorama_use_input_masks",
    "PANO3DGS_RERENDER_PERSPECTIVE": "rerender_perspective",
    "PANO3DGS_RERUN_PANORAMA_FEATURES": "rerun_panorama_features",
    "PANO3DGS_RERUN_PANORAMA_MATCHING": "rerun_panorama_matching",
    "PANO3DGS_CLEAN_PANORAMA_SFM": "clean_panorama_sfm",
}

CONFIG_SECTIONS: dict[str, dict[str, str]] = {
    "paths": {
        "colmap": "colmap",
        "runs_dir": "runs_dir",
    },
    "run": {
        "rate_hz": "rate_hz",
        "equirect_width": "equirect_width",
        "equirect_height": "equirect_height",
    },
    "extract": {
        "window_seconds": "window_seconds",
        "scale_width": "scale_width",
        "roi": "roi",
        "fallback_fps": "fallback_fps",
        "frame_jpg_quality": "frame_jpg_quality",
        "progress": "progress",
    },
    "colmap": {
        "gpu_index": "gpu_index",
        "threads": "threads",
        "clean_colmap": "clean_colmap",
        "max_features": "max_features",
        "overlap": "overlap",
    },
    "sam3": {
        "model": "sam3_model",
        "repo": "sam3_repo",
        "device": "device",
        "dtype": "dtype",
        "prompts": "sam3_prompts",
        "skip": "skip_sam3",
        "score": "score",
        "min_area": "min_area",
        "max_area": "max_area",
        "dilate": "dilate",
    },
    "masks": {
        "dynamic_mask_dir": "dynamic_mask_dir",
        "heuristics": "mask_heuristics",
        "progress": "mask_progress",
        "sky_mask": "sky_mask",
        "zenith_mask": "zenith_mask",
        "nadir_mask": "nadir_mask",
    },
    "cubemap": {
        "face_size": "face_size",
        "faces": "faces",
        "fov": "fov",
        "image_ext": "image_ext",
        "jpg_quality": "cubemap_jpg_quality",
        "workers": "cubemap_workers",
        "mask_name_mode": "mask_name_mode",
    },
    "panorama_sfm": {
        "pycolmap_path": "pycolmap_path",
        "require_pycolmap_cuda": "require_pycolmap_cuda",
        "output": "panorama_sfm_output",
        "render_type": "pano_render_type",
        "virtual_camera_model": "panorama_virtual_camera_model",
        "matcher": "panorama_matcher",
        "mapper": "panorama_mapper",
        "ba_backend": "panorama_ba_backend",
        "loop_detection": "panorama_loop_detection",
        "vocab_tree_path": "panorama_vocab_tree_path",
        "workers": "panorama_workers",
        "use_input_masks": "panorama_use_input_masks",
        "rerender_perspective": "rerender_perspective",
        "rerun_features": "rerun_panorama_features",
        "rerun_matching": "rerun_panorama_matching",
        "clean": "clean_panorama_sfm",
    },
}

PATH_FIELDS = {
    "colmap",
    "runs_dir",
    "sam3_model",
    "sam3_repo",
    "dynamic_mask_dir",
    "pycolmap_path",
    "panorama_sfm_output",
    "panorama_vocab_tree_path",
}

BOOL_FIELDS = {
    "clean_colmap",
    "skip_sam3",
    "sky_mask",
    "require_pycolmap_cuda",
    "panorama_loop_detection",
    "panorama_use_input_masks",
    "rerender_perspective",
    "rerun_panorama_features",
    "rerun_panorama_matching",
    "clean_panorama_sfm",
}

INT_FIELDS = {
    "equirect_width",
    "equirect_height",
    "scale_width",
    "frame_jpg_quality",
    "progress",
    "threads",
    "max_features",
    "overlap",
    "dilate",
    "mask_progress",
    "face_size",
    "cubemap_workers",
    "panorama_workers",
}

FLOAT_FIELDS = {
    "rate_hz",
    "window_seconds",
    "fallback_fps",
    "score",
    "min_area",
    "max_area",
    "zenith_mask",
    "nadir_mask",
    "fov",
}


def load_dotenv(start: Path | None = None) -> None:
    current = (start or Path.cwd()).resolve()
    for directory in [current] + list(current.parents):
        env_path = directory / ".env"
        if not env_path.exists():
            continue
        with env_path.open() as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return


def find_nearest_config(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for directory in [current] + list(current.parents):
        for name in ("pano3dgs.toml", "pano-3dgs.toml"):
            path = directory / name
            if path.exists():
                return path
    return None


def find_cli_config(argv: list[str] | None) -> Path | None:
    values = list(sys.argv[1:] if argv is None else argv)
    for idx, value in enumerate(values):
        if value == "--config":
            if idx + 1 >= len(values):
                raise SystemExit("--config requires a TOML path")
            return Path(values[idx + 1])
        if value.startswith("--config="):
            return Path(value.split("=", 1)[1])
    return None


def load_settings(config_path: Path | None = None) -> Settings:
    settings = _apply_mapping(Settings(), _env_mapping())
    if config_path is None:
        config_path = find_nearest_config()
    if config_path is not None:
        settings = _apply_mapping(settings, read_toml_config(config_path))
    return settings


def read_toml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"config file not found: {path}")
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"invalid TOML config {path}: {exc}") from exc
    return _flatten_config(data)


def _env_mapping() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for env_name, field_name in ENV_FIELDS.items():
        if env_name in os.environ:
            out[field_name] = os.environ[env_name]
    return out


def _flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    valid_fields = {field.name for field in fields(Settings)}
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in CONFIG_SECTIONS:
            if not isinstance(value, dict):
                raise SystemExit(f"config section [{key}] must be a table")
            for section_key, section_value in value.items():
                field_name = CONFIG_SECTIONS[key].get(section_key)
                if field_name is None:
                    raise SystemExit(f"unknown config key: {key}.{section_key}")
                out[field_name] = section_value
        elif key in valid_fields:
            out[key] = value
        else:
            raise SystemExit(f"unknown config section/key: {key}")
    return out


def _apply_mapping(settings: Settings, values: dict[str, Any]) -> Settings:
    converted = {key: _convert_value(key, value) for key, value in values.items()}
    return replace(settings, **converted)


def _convert_value(field_name: str, value: Any) -> Any:
    if value in ("", None):
        return None if field_name in PATH_FIELDS or field_name == "sam3_prompts" else value
    if field_name in PATH_FIELDS:
        return Path(value)
    if field_name in BOOL_FIELDS:
        return _parse_bool(value)
    if field_name == "mask_heuristics":
        return _parse_bool_or_none(value)
    if field_name in INT_FIELDS:
        return int(value)
    if field_name in FLOAT_FIELDS:
        return float(value)
    if field_name == "faces":
        return parse_faces(value)
    if field_name == "sam3_prompts":
        return parse_list(value)
    if field_name in {"roi"}:
        return _parse_box_value(value)
    return str(value) if isinstance(value, Path) else value


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _parse_bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"", "auto", "none"}:
        return None
    return lowered in {"1", "true", "yes", "on"}


def _parse_box_value(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, str):
        parts = [float(part.strip()) for part in value.split(",")]
    else:
        parts = [float(part) for part in value]
    if len(parts) != 4:
        raise SystemExit("roi must contain four normalized coordinates")
    x0, y0, x1, y1 = parts
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise SystemExit("roi coordinates must be normalized to [0,1]")
    return x0, y0, x1, y1
