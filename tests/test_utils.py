from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pano_3dgs.panorama_sfm import require_complete_mask_set
from pano_3dgs.utils import choose_worker_count, link_or_copy_file


class LinkOrCopyFileTests(unittest.TestCase):
    def test_windows_copy_is_reused_when_source_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.bin"
            destination = root / "output" / "destination.bin"
            source.write_bytes(b"frame")

            with mock.patch("pano_3dgs.utils.sys.platform", "win32"):
                self.assertEqual(link_or_copy_file(source, destination), "copied")
                self.assertEqual(link_or_copy_file(source, destination), "existing")

                source.write_bytes(b"updated frame")
                self.assertEqual(link_or_copy_file(source, destination), "copied")
                self.assertEqual(destination.read_bytes(), b"updated frame")

    def test_linux_uses_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"frame")

            with mock.patch("pano_3dgs.utils.sys.platform", "linux"):
                self.assertEqual(link_or_copy_file(source, destination), "linked")
            self.assertTrue(source.samefile(destination))

    def test_linux_hardlink_failure_does_not_fallback_to_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"frame")

            with (
                mock.patch("pano_3dgs.utils.sys.platform", "linux"),
                mock.patch("pano_3dgs.utils.os.link", side_effect=OSError("cross-device link")),
                mock.patch("pano_3dgs.utils.shutil.copy2") as copy_file,
            ):
                with self.assertRaisesRegex(OSError, "cross-device"):
                    link_or_copy_file(source, destination)
            copy_file.assert_not_called()


class WorkerChoiceTests(unittest.TestCase):
    @mock.patch("pano_3dgs.utils.psutil.virtual_memory")
    @mock.patch("pano_3dgs.utils.psutil.cpu_count", return_value=32)
    def test_windows_auto_workers_respect_platform_and_memory_limits(
        self,
        _cpu_count: mock.Mock,
        virtual_memory: mock.Mock,
    ) -> None:
        virtual_memory.return_value.available = 10 * 1024**3
        with mock.patch("pano_3dgs.utils.sys.platform", "win32"):
            choice = choose_worker_count(
                0,
                100,
                estimated_per_worker_bytes=2 * 1024**3,
                min_available_memory_bytes=0,
                memory_fraction=1.0,
            )
        self.assertEqual(choice.workers, 4)

    def test_manual_workers_are_bounded_by_item_count(self) -> None:
        choice = choose_worker_count(
            8,
            3,
            estimated_per_worker_bytes=0,
        )
        self.assertEqual(choice.workers, 3)


class CompleteMaskSetTests(unittest.TestCase):
    def test_rejects_missing_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_dir = Path(temp_dir)
            (mask_dir / "frame_0001.jpg.png").write_bytes(b"mask")

            with self.assertRaisesRegex(SystemExit, "frame_0002.jpg"):
                require_complete_mask_set(
                    ["frame_0001.jpg", "frame_0002.jpg"],
                    mask_dir,
                )


if __name__ == "__main__":
    unittest.main()
