import tempfile
import unittest
from pathlib import Path

import numpy as np

from omniloader.data.npy import NpyFolderDataset


class TestNpyFolderDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        for i in range(4):
            sample_dir = root / f"sample_{i:02d}"
            sample_dir.mkdir()
            np.save(sample_dir / "video.npy", np.random.randn(6, 8).astype("float32"))
            np.save(sample_dir / "label.npy", np.array(i, dtype="int64"))
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_len_and_keys(self):
        ds = NpyFolderDataset(self.root)
        self.assertEqual(len(ds), 4)
        self.assertEqual(sorted(ds.keys), ["label", "video"])

    def test_getitem_returns_tensors_and_key(self):
        ds = NpyFolderDataset(self.root)
        sample = ds[0]
        self.assertEqual(sample["video"].shape, (6, 8))
        self.assertEqual(sample["key"], "sample_00")
        self.assertEqual(int(sample["label"]), 0)

    def test_keys_subset(self):
        ds = NpyFolderDataset(self.root, keys=["video"])
        self.assertNotIn("label", ds[1])

    def test_no_mmap(self):
        ds = NpyFolderDataset(self.root, mmap=False)
        self.assertEqual(ds[2]["video"].shape, (6, 8))

    def test_missing_file_skipped(self):
        # A sample lacking one key just omits it (union schema fills the gap later).
        (self.root / "sample_00" / "label.npy").unlink()
        ds = NpyFolderDataset(self.root, keys=["video", "label"])
        self.assertNotIn("label", ds[0])

    def test_empty_root_raises(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
                NpyFolderDataset(empty)


if __name__ == "__main__":
    unittest.main()
