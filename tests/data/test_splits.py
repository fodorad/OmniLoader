import tempfile
import unittest
from collections import Counter
from pathlib import Path

from omniloader.data.splits import (
    load_split_info,
    save_split_info,
    split_indices,
)


class TestSplitIndices(unittest.TestCase):
    def test_disjoint_and_covers(self):
        splits = split_indices(100, ratios=(0.8, 0.1, 0.1), seed=0)
        self.assertEqual(len(splits["train"]), 80)
        self.assertEqual(len(splits["valid"]), 10)
        self.assertEqual(len(splits["test"]), 10)
        all_idx = splits["train"] + splits["valid"] + splits["test"]
        self.assertEqual(sorted(all_idx), list(range(100)))

    def test_reproducible(self):
        self.assertEqual(split_indices(50, seed=3), split_indices(50, seed=3))

    def test_seed_changes_partition(self):
        self.assertNotEqual(split_indices(50, seed=1), split_indices(50, seed=2))

    def test_stratified_preserves_proportions(self):
        labels = [0] * 80 + [1] * 20
        splits = split_indices(100, ratios=(0.5, 0.5), seed=0, stratify=labels, names=("a", "b"))
        for name in ("a", "b"):
            counts = Counter(labels[i] for i in splits[name])
            self.assertEqual(counts[0], 40)
            self.assertEqual(counts[1], 10)

    def test_validation(self):
        with self.assertRaises(ValueError):
            split_indices(10, ratios=(0.5, 0.5), names=("only_one",))
        with self.assertRaises(ValueError):
            split_indices(10, stratify=[0, 1])  # wrong length

    def test_save_load_round_trip(self):
        splits = split_indices(20, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            save_split_info(splits, path, meta={"seed": 0, "ratios": [0.8, 0.1, 0.1]})
            loaded = load_split_info(path)
        self.assertEqual(loaded, splits)


if __name__ == "__main__":
    unittest.main()
