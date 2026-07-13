from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass


DEFAULT_RELEASE_API_URL = (
    "https://api.github.com/repos/lyehe/build_gpu_colmap/releases/tags/v4.1.0"
)
DEFAULT_RELEASE_DOWNLOAD_BASE = (
    "https://github.com/lyehe/build_gpu_colmap/releases/download/v4.1.0"
)
DEFAULT_PYCOLMAP_VERSION = "4.1.0"


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    browser_download_url: str


def current_python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def current_platform_tag() -> str:
    machine = platform.machine().lower()
    if sys.platform == "win32":
        if machine not in {"amd64", "x86_64"}:
            raise SystemExit(
                f"unsupported Windows architecture for prebuilt PyCOLMAP: {machine}"
            )
        return "win_amd64"
    if sys.platform == "linux":
        if machine not in {"x86_64", "amd64"}:
            raise SystemExit(
                f"unsupported Linux architecture for prebuilt PyCOLMAP: {machine}"
            )
        return "manylinux_2_35_x86_64"
    raise SystemExit(f"unsupported platform for prebuilt PyCOLMAP: {sys.platform}")


def default_variant() -> str:
    if sys.platform == "win32":
        return "cuda.cudss"
    if sys.platform == "linux":
        return "cu128.bundled.cudss"
    raise SystemExit(
        f"unsupported platform for a default PyCOLMAP wheel: {sys.platform}"
    )


def load_release_assets(release_api_url: str) -> list[ReleaseAsset]:
    request = urllib.request.Request(
        release_api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "pano-3dgs-pycolmap-installer",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [
        ReleaseAsset(
            name=asset["name"], browser_download_url=asset["browser_download_url"]
        )
        for asset in payload.get("assets", [])
    ]


def select_pycolmap_wheel(
    assets: list[ReleaseAsset],
    *,
    python_tag: str,
    platform_tag: str,
    variant: str,
) -> ReleaseAsset:
    prefix = "pycolmap-"
    variant_part = f"+{variant}-"
    tag_part = f"-{python_tag}-{python_tag}-{platform_tag}.whl"
    matches = [
        asset
        for asset in assets
        if asset.name.startswith(prefix)
        and variant_part in asset.name
        and asset.name.endswith(tag_part)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(
            "no matching PyCOLMAP wheel found for "
            f"variant={variant!r}, python={python_tag!r}, platform={platform_tag!r}"
        )
    names = ", ".join(asset.name for asset in matches)
    raise SystemExit(f"multiple matching PyCOLMAP wheels found: {names}")


def build_pycolmap_wheel(
    *,
    version: str,
    python_tag: str,
    platform_tag: str,
    variant: str,
    download_base_url: str,
) -> ReleaseAsset:
    name = f"pycolmap-{version}+{variant}-{python_tag}-{python_tag}-{platform_tag}.whl"
    return ReleaseAsset(
        name=name,
        browser_download_url=f"{download_base_url.rstrip('/')}/{name}",
    )


def install_wheel(url: str, installer: str) -> None:
    if installer == "uv":
        if shutil.which("uv") is None:
            raise SystemExit("uv was requested but was not found on PATH")
        command = ["uv", "pip", "install", "--python", sys.executable, url]
    elif installer == "pip":
        command = [sys.executable, "-m", "pip", "install", url]
    else:
        raise SystemExit(f"unknown installer: {installer}")
    subprocess.run(command, check=True)


def install_pycolmap(args: argparse.Namespace) -> None:
    python_tag = args.python_tag or current_python_tag()
    platform_tag = args.platform_tag or current_platform_tag()
    variant = args.variant
    if variant == "auto":
        variant = default_variant()

    wheel = build_pycolmap_wheel(
        version=args.pycolmap_version,
        python_tag=python_tag,
        platform_tag=platform_tag,
        variant=variant,
        download_base_url=args.release_download_base,
    )
    if args.check_release_assets:
        assets = load_release_assets(args.release_api_url)
        wheel = select_pycolmap_wheel(
            assets,
            python_tag=python_tag,
            platform_tag=platform_tag,
            variant=variant,
        )
    print(f"selected PyCOLMAP wheel: {wheel.name}", flush=True)
    print(wheel.browser_download_url, flush=True)
    if args.print_only:
        return
    install_wheel(wheel.browser_download_url, args.installer)
