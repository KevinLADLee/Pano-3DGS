from __future__ import annotations

from pathlib import Path

from pano_3dgs.utils import ensure_dir


def write_reconstruction_pair(reconstruction, binary_dir: Path, text_dir: Path) -> None:
    ensure_dir(binary_dir)
    ensure_dir(text_dir)
    reconstruction.write_binary(binary_dir)
    reconstruction.write_text(text_dir)
