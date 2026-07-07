import tempfile
import unittest
from pathlib import Path

import torch

from omniloader.data.datasets import DictTensorDataset, HDF5Dataset
from tests.fixtures import VIDEO_F, VIDEO_T, write_hdf5


class TestDictTensorDataset(unittest.TestCase):
    def test_getitem_and_len(self):
        ds = DictTensorDataset(
            {"x": torch.randn(5, 3), "y": torch.arange(5)},
            metadata={"key": [f"s{i}" for i in range(5)]},
        )
        self.assertEqual(len(ds), 5)
        sample = ds[2]
        self.assertEqual(sample["x"].shape, (3,))
        self.assertEqual(int(sample["y"]), 2)
        self.assertEqual(sample["key"], "s2")

    def test_requires_tensors(self):
        with self.assertRaises(ValueError):
            DictTensorDataset({})

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            DictTensorDataset({"x": torch.randn(3, 2), "y": torch.randn(4, 2)})


class TestHDF5Dataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "data.h5"
        write_hdf5(self.path, subset="train", n=5)

    def tearDown(self):
        self.tmp.cleanup()

    def test_len_and_sorted_ids(self):
        ds = HDF5Dataset(self.path, "train")
        self.assertEqual(len(ds), 5)
        self.assertEqual(ds.sample_ids[0], "sample_000")

    def test_getitem_decodes_values(self):
        ds = HDF5Dataset(self.path, "train")
        sample = ds[0]
        self.assertEqual(sample["video"].shape, (VIDEO_T, VIDEO_F))
        self.assertEqual(sample["valence"].shape, (VIDEO_T,))
        self.assertEqual(sample["sentiment"].shape, ())
        self.assertEqual(sample["dataset"], "synthetic")
        self.assertEqual(sample["key"], "clip_0")

    def test_key_subset(self):
        ds = HDF5Dataset(self.path, "train", keys=["video"])
        sample = ds[1]
        self.assertIn("video", sample)
        self.assertNotIn("valence", sample)

    def test_missing_subset_raises(self):
        with self.assertRaises(KeyError):
            HDF5Dataset(self.path, "nonexistent")

    def test_pickle_drops_handle(self):
        ds = HDF5Dataset(self.path, "train")
        _ = ds[0]  # opens the handle
        state = ds.__getstate__()
        self.assertIsNone(state["_file"])
        self.assertEqual(len(state["_cache"]), 0)

    def test_lru_cache_hits_and_evicts(self):
        ds = HDF5Dataset(self.path, "train", cache_size=2)
        a = ds[0]
        self.assertIs(ds[0], a)  # cache hit returns the same object
        ds[1]
        ds[2]  # exceeds cache_size 2 -> index 0 evicted
        self.assertNotIn(0, ds._cache)
        self.assertIn(2, ds._cache)

    def test_preload_reads_all(self):
        ds = HDF5Dataset(self.path, "train", preload=True)
        self.assertIsNotNone(ds._preloaded)
        self.assertEqual(ds[0]["video"].shape, (VIDEO_T, VIDEO_F))

    def test_decodes_varied_dtypes(self):
        import h5py
        import numpy as np

        extra = Path(self.tmp.name) / "varied.h5"
        with h5py.File(extra, "w") as f:
            sg = f.create_group("train").create_group("s0")
            sg.create_dataset("flag", data=np.bool_(True))
            sg.create_dataset("count", data=np.int64(7))
            sg.create_dataset("labels", data=np.array(["a", "b"], dtype="S1"))
        sample = HDF5Dataset(extra, "train")[0]
        self.assertTrue(bool(sample["flag"]))
        self.assertEqual(int(sample["count"]), 7)
        self.assertEqual(sample["labels"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
