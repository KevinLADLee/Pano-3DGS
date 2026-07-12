from __future__ import annotations

import io
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path

from pano_3dgs.sfm_utils import prepare_feature_database


def create_feature_database(path: Path, image_names: list[str]) -> None:
    with closing(sqlite3.connect(path)) as database:
        database.execute("CREATE TABLE cameras (model INTEGER NOT NULL)")
        database.execute("CREATE TABLE images (name TEXT NOT NULL)")
        database.execute("CREATE TABLE matches (pair_id INTEGER)")
        database.execute("CREATE TABLE two_view_geometries (pair_id INTEGER)")
        database.execute("INSERT INTO cameras VALUES (1)")
        database.executemany(
            "INSERT INTO images VALUES (?)",
            [(name,) for name in image_names],
        )
        database.execute("INSERT INTO matches VALUES (1)")
        database.execute("INSERT INTO two_view_geometries VALUES (1)")
        database.commit()


class FeatureDatabasePlanTests(unittest.TestCase):
    def test_reuses_complete_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "database.db"
            create_feature_database(database_path, ["a.jpg", "b.jpg"])

            plan = prepare_feature_database(
                object(),
                database_path,
                expected_camera_model_id=1,
                expected_image_names=["a.jpg", "b.jpg"],
                rerun_features=False,
                rerun_matching=False,
            )

            self.assertFalse(plan.extract_features)
            self.assertFalse(plan.match_features)

    def test_rerun_matching_preserves_features_and_clears_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "database.db"
            create_feature_database(database_path, ["a.jpg"])

            plan = prepare_feature_database(
                object(),
                database_path,
                expected_camera_model_id=1,
                expected_image_names=["a.jpg"],
                rerun_features=False,
                rerun_matching=True,
            )

            self.assertFalse(plan.extract_features)
            self.assertTrue(plan.match_features)
            with closing(sqlite3.connect(database_path)) as database:
                self.assertEqual(
                    database.execute("SELECT COUNT(*) FROM cameras").fetchone()[0], 1
                )
                self.assertEqual(
                    database.execute("SELECT COUNT(*) FROM images").fetchone()[0], 1
                )
                self.assertEqual(
                    database.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 0
                )
                self.assertEqual(
                    database.execute(
                        "SELECT COUNT(*) FROM two_view_geometries"
                    ).fetchone()[0],
                    0,
                )

    def test_changed_image_set_rebuilds_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "database.db"
            create_feature_database(database_path, ["old.jpg"])

            with redirect_stdout(io.StringIO()):
                plan = prepare_feature_database(
                    object(),
                    database_path,
                    expected_camera_model_id=1,
                    expected_image_names=["new.jpg"],
                    rerun_features=False,
                    rerun_matching=False,
                )

            self.assertTrue(plan.extract_features)
            self.assertTrue(plan.match_features)
            self.assertFalse(database_path.exists())


if __name__ == "__main__":
    unittest.main()
