from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pycolmap
from PIL import Image

from pano_3dgs.panorama_sfm import (
    PANO_RENDER_OPTIONS,
    PanoRenderOptions,
    build_render_configuration,
    render_perspective_images,
    render_configuration_changed,
    virtual_image_names,
    write_render_configuration,
)


class RenderConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.options = PanoRenderOptions(
            num_steps_yaw=4,
            pitches_deg=(-35.0, 0.0, 35.0),
            hfov_deg=90.0,
            vfov_deg=90.0,
        )
        self.configuration = build_render_configuration(
            pano_width=7680,
            pano_height=3840,
            render_options=self.options,
            camera_model="pinhole",
            use_source_masks=True,
        )

    def test_unchanged_manifest_reuses_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / "images"
            image_dir.mkdir()
            (image_dir / "frame.jpg").write_bytes(b"frame")
            write_render_configuration(image_dir, self.configuration)

            self.assertFalse(
                render_configuration_changed(image_dir, self.configuration)
            )

    def test_missing_or_changed_manifest_requires_rerender(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / "images"
            image_dir.mkdir()
            (image_dir / "frame.jpg").write_bytes(b"frame")

            self.assertTrue(render_configuration_changed(image_dir, self.configuration))
            write_render_configuration(image_dir, self.configuration)
            changed = dict(self.configuration, use_source_masks=False)
            self.assertTrue(render_configuration_changed(image_dir, changed))


class VirtualImageNameTests(unittest.TestCase):
    def test_names_cover_every_camera_and_panorama(self) -> None:
        options = PanoRenderOptions(
            num_steps_yaw=2,
            pitches_deg=(0.0, 30.0),
            hfov_deg=90.0,
            vfov_deg=90.0,
        )

        names = virtual_image_names(["a.jpg", "nested/b.jpg"], options)

        self.assertEqual(len(names), 8)
        self.assertIn("pano_camera0/a.jpg", names)
        self.assertIn("pano_camera3/nested/b.jpg", names)


class PerspectiveRenderSmokeTests(unittest.TestCase):
    def test_renders_small_equirectangular_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pano_dir = root / "panos"
            image_dir = root / "output" / "images"
            mask_dir = root / "output" / "masks"
            pano_dir.mkdir()
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            Image.new("RGB", (64, 32), color=(20, 40, 60)).save(pano_dir / "frame.jpg")

            with redirect_stdout(io.StringIO()):
                processor = render_perspective_images(
                    pycolmap,
                    ["frame.jpg"],
                    pano_dir,
                    image_dir,
                    mask_dir,
                    None,
                    PANO_RENDER_OPTIONS["perspective_non_overlapping"],
                    "pinhole",
                    1,
                    False,
                )

            self.assertEqual(processor.camera.model_name, "PINHOLE")
            self.assertEqual(len(list(image_dir.rglob("*.jpg"))), 4)
            self.assertEqual(len(list(mask_dir.rglob("*.png"))), 4)
            self.assertTrue((root / "output" / "render_config.json").is_file())


if __name__ == "__main__":
    unittest.main()
