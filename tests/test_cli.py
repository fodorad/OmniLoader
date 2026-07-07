import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from omniloader.cli import main
from tests.fixtures import VIDEO_F, VIDEO_T, write_hdf5


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.h5 = root / "data.h5"
        write_hdf5(self.h5, subset="train", n=5)
        self.config = root / "config.json"
        self.config.write_text(
            json.dumps(
                {
                    "datasets": [
                        {
                            "adapter": "hdf5",
                            "args": {"h5_path": str(self.h5), "subset": "train"},
                            "schema": {
                                "features": [
                                    {"name": "video", "feature_dim": VIDEO_F, "time_dim": VIDEO_T}
                                ],
                                "targets": [
                                    {"name": "sentiment", "placeholder": -5.0},
                                    {"name": "emotion", "dtype": "int64", "placeholder": -1},
                                ],
                            },
                        }
                    ]
                }
            )
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_describe(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["describe", str(self.config)])
        self.assertEqual(code, 0)
        self.assertIn("coverage", out.getvalue())

    def test_validate_ok(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["validate", str(self.config)])
        self.assertEqual(code, 0)
        self.assertIn("OK", out.getvalue())

    def test_validate_reports_issue(self):
        # Break the schema so validation fails (wrong feature_dim).
        data = json.loads(self.config.read_text())
        data["datasets"][0]["schema"]["features"][0]["feature_dim"] = 999
        self.config.write_text(json.dumps(data))
        with redirect_stdout(io.StringIO()):
            code = main(["validate", str(self.config)])
        self.assertEqual(code, 1)

    def test_compute_stats(self):
        out_path = Path(self.tmp.name) / "stats.json"
        with redirect_stdout(io.StringIO()):
            code = main(["compute-stats", str(self.config), "-o", str(out_path)])
        self.assertEqual(code, 0)
        self.assertTrue(out_path.exists())
        stats = json.loads(out_path.read_text())
        self.assertIn("video", stats)
        self.assertIn("mean", stats["video"])

    def test_compute_stats_per_dataset(self):
        out_path = Path(self.tmp.name) / "ds_stats.json"
        with redirect_stdout(io.StringIO()):
            code = main(["compute-stats", str(self.config), "-o", str(out_path), "--per-dataset"])
        self.assertEqual(code, 0)
        stats = json.loads(out_path.read_text())
        # Grouped by the `dataset` metadata the fixture stamps ("synthetic").
        self.assertIn("synthetic", stats)
        self.assertIn("video", stats["synthetic"])
        self.assertIn("mean", stats["synthetic"]["video"])

    def test_class_weights_for_loss(self):
        out_path = Path(self.tmp.name) / "cw.json"
        args = [
            "class-weights-for-loss",
            str(self.config),
            "--target",
            "emotion",
            "-o",
            str(out_path),
        ]
        with redirect_stdout(io.StringIO()):
            code = main(args)
        self.assertEqual(code, 0)
        data = json.loads(out_path.read_text())
        self.assertEqual(data["target"], "emotion")
        self.assertEqual(len(data["weights"]), data["num_classes"])
        self.assertEqual(sum(data["counts"]), 5)  # every one of the 5 samples has an emotion

    def test_no_datasets_errors(self):
        empty = Path(self.tmp.name) / "empty.json"
        empty.write_text("{}")
        with self.assertRaises(SystemExit):
            main(["describe", str(empty)])


if __name__ == "__main__":
    unittest.main()
