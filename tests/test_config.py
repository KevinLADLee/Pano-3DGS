from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pano_3dgs.config import load_settings


class ConfigBooleanTests(unittest.TestCase):
    def test_rejects_unknown_boolean_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "pano3dgs.toml"
            config_path.write_text(
                '[masks]\nsky_mask = "flase"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SystemExit, "sky_mask.*flase"):
                load_settings(config_path)

    def test_accepts_explicit_false_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "pano3dgs.toml"
            config_path.write_text(
                '[masks]\nsky_mask = "off"\n',
                encoding="utf-8",
            )

            self.assertFalse(load_settings(config_path).sky_mask)


if __name__ == "__main__":
    unittest.main()
