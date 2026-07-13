from __future__ import annotations

import sys
import unittest
from unittest import mock

from pano_3dgs.pycolmap_release import (
    ReleaseAsset,
    default_variant,
    install_wheel,
    select_pycolmap_wheel,
)


class WheelSelectionTests(unittest.TestCase):
    def test_selects_exact_variant_and_python_tag(self) -> None:
        expected = ReleaseAsset(
            name="pycolmap-4.1.0+cuda.cudss-cp312-cp312-win_amd64.whl",
            browser_download_url="https://example.invalid/expected.whl",
        )
        assets = [
            ReleaseAsset(
                name="pycolmap-4.1.0+cpu-cp312-cp312-win_amd64.whl",
                browser_download_url="https://example.invalid/cpu.whl",
            ),
            expected,
        ]

        selected = select_pycolmap_wheel(
            assets,
            python_tag="cp312",
            platform_tag="win_amd64",
            variant="cuda.cudss",
        )

        self.assertEqual(selected, expected)

    def test_default_variant_has_no_unsupported_platform_fallback(self) -> None:
        with mock.patch("pano_3dgs.pycolmap_release.sys.platform", "darwin"):
            with self.assertRaisesRegex(SystemExit, "unsupported platform"):
                default_variant()


class WheelInstallTests(unittest.TestCase):
    @mock.patch("pano_3dgs.pycolmap_release.subprocess.run")
    @mock.patch("pano_3dgs.pycolmap_release.shutil.which", return_value="uv.exe")
    def test_uv_installs_into_current_interpreter(
        self,
        _which: mock.Mock,
        run: mock.Mock,
    ) -> None:
        install_wheel("https://example.invalid/pycolmap.whl", "uv")

        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["uv", "pip", "install", "--python"])
        self.assertEqual(command[4], sys.executable)
        self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
