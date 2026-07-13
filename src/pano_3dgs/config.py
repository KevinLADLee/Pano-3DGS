from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - project currently runs on Python 3.11 locally.
    import tomli as tomllib

from pano_3dgs.defaults import DEFAULT_SAM3_MODEL
from pano_3dgs.utils import parse_list


@dataclass(frozen=True)
class Settings:
    runs_dir: Path = Path("runs")
    rate_hz: float = 2.0
    equirect_width: int = 7680
    equirect_height: int = 3840
    sfm_workflow: str = "equirect"

    window_seconds: float = 0.5
    scale_width: int = 1920
    roi: tuple[float, float, float, float] = (0.0, 0.08, 1.0, 0.92)
    fallback_fps: float = 30.0
    frame_jpg_quality: int = 98
    progress: int = 300

    gpu_index: str = "0"
    threads: int = 8
    max_features: int = 12000
    feature_type: str = "sift"
    feature_matcher: str = "auto"
    aliked_model_path: Path | None = None
    aliked_matcher_model_path: Path | None = None
    overlap: int = 25

    sam3_model: Path = DEFAULT_SAM3_MODEL
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    sam3_prompts: list[str] | None = field(default_factory=lambda: ["sky"])
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

    pycolmap_path: Path | None = None
    require_pycolmap_cuda: bool = True
    panorama_sfm_output: Path | None = None
    pano_render_type: str = "perspective_overlapping"
    panorama_virtual_camera_model: str = "pinhole"
    panorama_matcher: str = "sequential"
    panorama_mapper: str = "incremental"
    panorama_ba_backend: str = "caspar"
    panorama_loop_detection: bool = False
    panorama_vocab_tree_path: Path | None = None
    panorama_workers: int = 0
    panorama_use_input_masks: bool = True
    rerender_perspective: bool = False
    rerun_panorama_features: bool = False
    rerun_panorama_matching: bool = False
    clean_panorama_sfm: bool = False

    equirect_sfm_output: Path | None = None
    equirect_matcher: str = "sequential"
    equirect_mapper: str = "incremental"
    equirect_ba_backend: str = "caspar"
    equirect_loop_detection: bool = False
    equirect_vocab_tree_path: Path | None = None
    equirect_use_input_masks: bool = True
    rerun_equirect_features: bool = False
    rerun_equirect_matching: bool = False
    clean_equirect_sfm: bool = False

    export_pinhole_3dgs: bool = True
    pinhole_3dgs_output: Path | None = None
    pinhole_3dgs_input_sparse: Path | None = None
    pinhole_3dgs_render_type: str = "perspective_overlapping"
    pinhole_3dgs_workers: int = 0
    pinhole_3dgs_use_input_masks: bool = True
    pinhole_3dgs_rerender: bool = False
    pinhole_3dgs_min_track_length: int = 2
    clean_pinhole_3dgs: bool = False

    @property
    def resolved_sam3_prompts(self) -> list[str]:
        return list(self.sam3_prompts or [])


CONFIG_SECTIONS: dict[str, dict[str, str]] = {
    "paths": {
        "runs_dir": "runs_dir",
    },
    "run": {
        "rate_hz": "rate_hz",
        "equirect_width": "equirect_width",
        "equirect_height": "equirect_height",
        "sfm_workflow": "sfm_workflow",
    },
    "extract": {
        "window_seconds": "window_seconds",
        "scale_width": "scale_width",
        "roi": "roi",
        "fallback_fps": "fallback_fps",
        "frame_jpg_quality": "frame_jpg_quality",
        "progress": "progress",
    },
    "sfm": {
        "gpu_index": "gpu_index",
        "threads": "threads",
        "max_features": "max_features",
        "feature_type": "feature_type",
        "feature_matcher": "feature_matcher",
        "aliked_model_path": "aliked_model_path",
        "aliked_matcher_model_path": "aliked_matcher_model_path",
        "overlap": "overlap",
    },
    "pycolmap": {
        "path": "pycolmap_path",
        "require_cuda": "require_pycolmap_cuda",
    },
    "sam3": {
        "model": "sam3_model",
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
    "equirect_sfm": {
        "pycolmap_path": "pycolmap_path",
        "require_pycolmap_cuda": "require_pycolmap_cuda",
        "output": "equirect_sfm_output",
        "matcher": "equirect_matcher",
        "mapper": "equirect_mapper",
        "ba_backend": "equirect_ba_backend",
        "loop_detection": "equirect_loop_detection",
        "vocab_tree_path": "equirect_vocab_tree_path",
        "use_input_masks": "equirect_use_input_masks",
        "rerun_features": "rerun_equirect_features",
        "rerun_matching": "rerun_equirect_matching",
        "clean": "clean_equirect_sfm",
    },
    "pinhole_3dgs": {
        "enabled": "export_pinhole_3dgs",
        "output": "pinhole_3dgs_output",
        "input_sparse": "pinhole_3dgs_input_sparse",
        "render_type": "pinhole_3dgs_render_type",
        "workers": "pinhole_3dgs_workers",
        "use_input_masks": "pinhole_3dgs_use_input_masks",
        "rerender": "pinhole_3dgs_rerender",
        "min_track_length": "pinhole_3dgs_min_track_length",
        "clean": "clean_pinhole_3dgs",
    },
}

PATH_FIELDS = {
    "runs_dir",
    "sam3_model",
    "dynamic_mask_dir",
    "pycolmap_path",
    "aliked_model_path",
    "aliked_matcher_model_path",
    "panorama_sfm_output",
    "panorama_vocab_tree_path",
    "equirect_sfm_output",
    "equirect_vocab_tree_path",
    "pinhole_3dgs_output",
    "pinhole_3dgs_input_sparse",
}

BOOL_FIELDS = {
    "skip_sam3",
    "sky_mask",
    "require_pycolmap_cuda",
    "panorama_loop_detection",
    "panorama_use_input_masks",
    "rerender_perspective",
    "rerun_panorama_features",
    "rerun_panorama_matching",
    "clean_panorama_sfm",
    "equirect_loop_detection",
    "equirect_use_input_masks",
    "rerun_equirect_features",
    "rerun_equirect_matching",
    "clean_equirect_sfm",
    "export_pinhole_3dgs",
    "pinhole_3dgs_use_input_masks",
    "pinhole_3dgs_rerender",
    "clean_pinhole_3dgs",
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
    "panorama_workers",
    "pinhole_3dgs_workers",
    "pinhole_3dgs_min_track_length",
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
}


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
    settings = Settings()
    if config_path is None:
        config_path = find_nearest_config()
    if config_path is not None:
        settings = _apply_mapping(settings, read_toml_config(config_path))
    return settings


def read_toml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"config file not found: {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"invalid TOML config {path}: {exc}") from exc
    return _flatten_config(data)


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
        return (
            None if field_name in PATH_FIELDS or field_name == "sam3_prompts" else value
        )
    if field_name in PATH_FIELDS:
        return Path(value)
    if field_name in BOOL_FIELDS:
        return _parse_bool(field_name, value)
    if field_name == "mask_heuristics":
        return _parse_bool_or_none(field_name, value)
    if field_name in INT_FIELDS:
        return int(value)
    if field_name in FLOAT_FIELDS:
        return float(value)
    if field_name == "sam3_prompts":
        return parse_list(value)
    if field_name in {"roi"}:
        return _parse_box_value(value)
    return str(value) if isinstance(value, Path) else value


def _parse_bool(field_name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"invalid boolean for {field_name}: {value!r}")


def _parse_bool_or_none(field_name: str, value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"", "auto", "none"}:
        return None
    return _parse_bool(field_name, value)


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
