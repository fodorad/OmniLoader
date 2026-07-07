import tempfile
import unittest
from pathlib import Path

import torch

from omniloader.data.datasets import HDF5Dataset
from omniloader.data.factory import build_datasets, schema_from_dict
from tests.fixtures import VIDEO_F, VIDEO_T, write_hdf5


class TestSchemaFromDict(unittest.TestCase):
    def test_builds_specs_with_string_dtype(self):
        schema = schema_from_dict(
            {
                "features": [{"name": "video", "feature_dim": 8, "time_dim": 6}],
                "targets": [{"name": "emotion", "dtype": "int64", "placeholder": -1}],
            }
        )
        self.assertEqual(schema.keys, {"video", "emotion"})
        self.assertEqual(schema.targets[0].dtype, torch.int64)


class TestBuildDatasets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "data.h5"
        write_hdf5(self.path, subset="train", n=5)

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_hdf5_dataset(self):
        entries = [
            {
                "adapter": "hdf5",
                "args": {"h5_path": str(self.path), "subset": "train"},
                "schema": {
                    "features": [{"name": "video", "feature_dim": VIDEO_F, "time_dim": VIDEO_T}],
                    "targets": [{"name": "sentiment", "placeholder": -5.0}],
                },
            }
        ]
        datasets, schemas = build_datasets(entries)
        self.assertEqual(len(datasets), 1)
        self.assertIsInstance(datasets[0], HDF5Dataset)
        self.assertEqual(len(datasets[0]), 5)
        self.assertEqual(schemas[0].keys, {"video", "sentiment"})

    def test_unknown_adapter_raises(self):
        with self.assertRaises(ValueError):
            build_datasets([{"adapter": "bogus", "schema": {}}])


if __name__ == "__main__":
    unittest.main()
