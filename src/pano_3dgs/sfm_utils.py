from __future__ import annotations

import argparse
import importlib
import os
import sqlite3
import sys
import time
from collections.abc import Collection
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


_DLL_DIRECTORY_HANDLES: list[object] = []
_REGISTERED_DLL_DIRECTORIES: set[Path] = set()


@dataclass(frozen=True)
class FeatureDatabasePlan:
    extract_features: bool
    match_features: bool


@dataclass(frozen=True)
class FeatureDatabaseInfo:
    camera_model_ids: set[int]
    image_names: set[str]


def _register_windows_dll_directory(path: Path) -> None:
    resolved = path.resolve()
    if resolved in _REGISTERED_DLL_DIRECTORIES or not resolved.is_dir():
        return
    os.environ["PATH"] = f"{resolved}{os.pathsep}{os.environ.get('PATH', '')}"
    _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(resolved)))
    _REGISTERED_DLL_DIRECTORIES.add(resolved)


def add_windows_pycolmap_dll_dirs(pycolmap_path: Path | None = None) -> None:
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    roots = [Path(path) for path in sys.path if path]
    if pycolmap_path is not None:
        roots.insert(0, pycolmap_path)
    for root in roots:
        for candidate in (root / "pycolmap.libs", root.parent / "pycolmap.libs"):
            try:
                _register_windows_dll_directory(candidate)
            except OSError:
                continue

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    _register_windows_dll_directory(Path(torch.__file__).resolve().parent / "lib")


def import_pycolmap(pycolmap_path: Path | None = None, require_cuda: bool = True):
    if pycolmap_path is not None:
        import_path = str(pycolmap_path.resolve())
        if import_path not in sys.path:
            sys.path.insert(0, import_path)
    add_windows_pycolmap_dll_dirs(pycolmap_path)
    try:
        pycolmap = importlib.import_module("pycolmap")
    except ImportError as exc:
        raise SystemExit(
            "PyCOLMAP is required for SfM. Install the CUDA + cuDSS wheel that "
            "matches this Python version and platform."
        ) from exc
    if require_cuda:
        if not getattr(pycolmap, "has_cuda", False):
            raise SystemExit("Imported PyCOLMAP does not report CUDA support.")
        if pycolmap.get_num_cuda_devices() < 1:
            raise SystemExit(
                "PyCOLMAP has CUDA support, but no CUDA device is visible."
            )
    print(
        f"pycolmap={getattr(pycolmap, '__version__', 'unknown')} "
        f"cuda={getattr(pycolmap, 'has_cuda', False)}",
        flush=True,
    )
    return pycolmap


def inspect_feature_database(database_path: Path) -> FeatureDatabaseInfo:
    with closing(sqlite3.connect(database_path)) as database:
        camera_rows = database.execute("SELECT DISTINCT model FROM cameras").fetchall()
        image_rows = database.execute("SELECT name FROM images").fetchall()
    return FeatureDatabaseInfo(
        camera_model_ids={int(row[0]) for row in camera_rows},
        image_names={str(row[0]) for row in image_rows},
    )


def remove_database_files(database_path: Path, reason: str) -> None:
    print(f"removing SfM database {reason}: {database_path}", flush=True)
    paths = (
        database_path,
        database_path.with_name(f"{database_path.name}-wal"),
        database_path.with_name(f"{database_path.name}-shm"),
    )
    last_error: PermissionError | None = None
    for attempt in range(10):
        try:
            for path in paths:
                if path.exists():
                    path.unlink()
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.2 * (attempt + 1))
    raise SystemExit(
        f"cannot remove locked database after retries: {database_path}. "
        "Close any pano-3dgs/PyCOLMAP/Python process using this run directory, "
        "then rerun the command."
    ) from last_error


def clear_database_matches(database_path: Path) -> None:
    match_tables = {"matches", "two_view_geometries", "inlier_matches"}
    with closing(sqlite3.connect(database_path)) as database:
        table_rows = database.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        existing_tables = {str(row[0]) for row in table_rows}
        for table_name in sorted(match_tables & existing_tables):
            database.execute(f"DELETE FROM {table_name}")
        database.commit()


def camera_model_id_name(pycolmap, model_id: int) -> str:
    try:
        return pycolmap.CameraModelId(model_id).name
    except ValueError:
        return f"UNKNOWN({model_id})"


def _format_name_mismatch(actual: set[str], expected: set[str]) -> str:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    parts = []
    if missing:
        parts.append(f"missing={len(missing)} ({', '.join(missing[:3])})")
    if unexpected:
        parts.append(f"unexpected={len(unexpected)} ({', '.join(unexpected[:3])})")
    return ", ".join(parts)


def prepare_feature_database(
    pycolmap,
    database_path: Path,
    *,
    expected_camera_model_id: int,
    expected_image_names: Collection[str],
    rerun_features: bool,
    rerun_matching: bool,
) -> FeatureDatabasePlan:
    if rerun_features:
        if database_path.exists():
            remove_database_files(database_path, "to rerun feature extraction")
        return FeatureDatabasePlan(extract_features=True, match_features=True)
    if not database_path.exists():
        return FeatureDatabasePlan(extract_features=True, match_features=True)

    try:
        info = inspect_feature_database(database_path)
    except sqlite3.Error as exc:
        remove_database_files(database_path, f"because it is invalid ({exc})")
        return FeatureDatabasePlan(extract_features=True, match_features=True)

    if info.camera_model_ids != {expected_camera_model_id}:
        existing_names = ", ".join(
            camera_model_id_name(pycolmap, model_id)
            for model_id in sorted(info.camera_model_ids)
        )
        expected_name = camera_model_id_name(pycolmap, expected_camera_model_id)
        remove_database_files(
            database_path,
            f"because camera model changed: {existing_names or 'none'} -> {expected_name}",
        )
        return FeatureDatabasePlan(extract_features=True, match_features=True)

    expected_names = set(expected_image_names)
    if info.image_names != expected_names:
        mismatch = _format_name_mismatch(info.image_names, expected_names)
        remove_database_files(
            database_path, f"because the image set changed: {mismatch}"
        )
        return FeatureDatabasePlan(extract_features=True, match_features=True)

    if rerun_matching:
        clear_database_matches(database_path)
        return FeatureDatabasePlan(extract_features=False, match_features=True)
    return FeatureDatabasePlan(extract_features=False, match_features=False)


def pycolmap_device(pycolmap, require_cuda: bool):
    return pycolmap.Device.cuda if require_cuda else pycolmap.Device.auto


def configure_feature_extraction_options(pycolmap, args: argparse.Namespace):
    options = pycolmap.FeatureExtractionOptions()
    options.num_threads = args.threads
    options.gpu_index = str(args.gpu_index)

    if args.feature_type == "sift":
        options.type = pycolmap.FeatureExtractorType.SIFT
        options.sift.max_num_features = args.max_features
        return options

    if args.feature_type not in {"aliked_n16rot", "aliked_n32"}:
        raise SystemExit(f"unknown feature type: {args.feature_type}")
    if not args.aliked_model_path:
        raise SystemExit(
            f"--feature-type {args.feature_type} requires --aliked-model-path"
        )
    if not args.aliked_model_path.is_file():
        raise SystemExit(f"ALIKED model file not found: {args.aliked_model_path}")

    options.aliked.max_num_features = args.max_features
    if args.feature_type == "aliked_n16rot":
        options.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
        options.aliked.n16rot_model_path = str(args.aliked_model_path)
    else:
        options.type = pycolmap.FeatureExtractorType.ALIKED_N32
        options.aliked.n32_model_path = str(args.aliked_model_path)
    return options


def configure_feature_matching_options(
    pycolmap,
    args: argparse.Namespace,
    *,
    rig_verification: bool = True,
):
    options = pycolmap.FeatureMatchingOptions()
    options.num_threads = args.threads
    options.gpu_index = str(args.gpu_index)
    options.rig_verification = rig_verification
    options.skip_image_pairs_in_same_frame = rig_verification

    matcher = args.feature_matcher
    if matcher == "auto":
        matcher = (
            "sift_bruteforce" if args.feature_type == "sift" else "aliked_bruteforce"
        )

    if matcher == "sift_bruteforce":
        if args.feature_type != "sift":
            raise SystemExit(
                "feature matcher sift_bruteforce requires --feature-type sift"
            )
        options.type = pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
        options.guided_matching = True
        return options

    if matcher not in {"aliked_bruteforce", "aliked_lightglue"}:
        raise SystemExit(f"unknown feature matcher: {matcher}")
    if args.feature_type not in {"aliked_n16rot", "aliked_n32"}:
        raise SystemExit(f"feature matcher {matcher} requires an ALIKED feature type")
    if not args.aliked_matcher_model_path:
        raise SystemExit(
            f"feature matcher {matcher} requires --aliked-matcher-model-path"
        )
    if not args.aliked_matcher_model_path.is_file():
        raise SystemExit(
            f"ALIKED matcher model file not found: {args.aliked_matcher_model_path}"
        )

    options.guided_matching = False
    if matcher == "aliked_bruteforce":
        options.type = pycolmap.FeatureMatcherType.ALIKED_BRUTEFORCE
        options.aliked.brute_force.model_path = str(args.aliked_matcher_model_path)
    else:
        options.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
        options.aliked.lightglue.model_path = str(args.aliked_matcher_model_path)
    return options


def pycolmap_ba_backend(pycolmap, name: str):
    if name == "ceres":
        return pycolmap.BundleAdjustmentBackend.CERES
    if name == "caspar":
        if not hasattr(pycolmap, "CasparBundleAdjustmentOptions"):
            raise SystemExit(
                "This PyCOLMAP build does not expose Caspar bundle adjustment options."
            )
        return pycolmap.BundleAdjustmentBackend.CASPAR
    raise SystemExit(f"unknown bundle adjustment backend: {name}")


def configure_incremental_ba(
    pycolmap,
    options,
    *,
    backend_name: str,
    num_threads: int,
    require_cuda: bool,
    gpu_index: str,
) -> None:
    backend = pycolmap_ba_backend(pycolmap, backend_name)
    options.num_threads = num_threads
    options.ba_use_gpu = require_cuda
    options.ba_gpu_index = str(gpu_index)
    options.ba_local_backend = backend
    options.ba_global_backend = backend
    try:
        valid = options.check()
    except Exception as exc:
        if backend_name == "caspar":
            raise SystemExit(
                "Caspar was requested, but this PyCOLMAP build was not compiled "
                "with CASPAR_ENABLED."
            ) from exc
        raise
    if valid is False:
        raise SystemExit(
            f"invalid incremental mapper options for BA backend {backend_name}"
        )


def run_matcher(
    pycolmap,
    database_path: Path,
    matching_options,
    device,
    *,
    matcher: str,
    overlap: int,
    loop_detection: bool,
    vocab_tree_path: Path | None,
    num_threads: int,
) -> None:
    if matcher == "sequential":
        pairing_options = pycolmap.SequentialPairingOptions()
        pairing_options.overlap = overlap
        pairing_options.quadratic_overlap = True
        pairing_options.expand_rig_images = True
        pairing_options.loop_detection = loop_detection
        pairing_options.num_threads = num_threads
        if vocab_tree_path:
            pairing_options.vocab_tree_path = str(vocab_tree_path)
        elif loop_detection:
            raise SystemExit(
                "sequential loop detection requires a local vocabulary tree"
            )
        pycolmap.match_sequential(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
        return
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(
            database_path, matching_options=matching_options, device=device
        )
        return
    if matcher == "vocabtree":
        if not vocab_tree_path:
            raise SystemExit("vocabtree matcher requires a local vocabulary tree")
        pairing_options = pycolmap.VocabTreePairingOptions()
        pairing_options.vocab_tree_path = str(vocab_tree_path)
        pairing_options.num_threads = num_threads
        pycolmap.match_vocabtree(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
        return
    if matcher == "spatial":
        pycolmap.match_spatial(
            database_path, matching_options=matching_options, device=device
        )
        return
    raise SystemExit(f"unknown matcher: {matcher}")
