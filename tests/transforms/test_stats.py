import tempfile
import unittest
from pathlib import Path

import torch

from omniloader.transforms import Normalize
from omniloader.transforms.stats import (
    compute_dataset_stats,
    compute_feature_stats,
    compute_stats,
    load_stats,
    save_stats,
)


class TestComputeFeatureStats(unittest.TestCase):
    def test_sequence_feature_mean_std(self):
        # Two samples, feature dim 2, all positions valid.
        samples = [
            {
                "x": torch.tensor([[1.0, 10.0], [3.0, 30.0]]),
                "x_mask": torch.ones(2, dtype=torch.bool),
            },
            {
                "x": torch.tensor([[5.0, 50.0], [7.0, 70.0]]),
                "x_mask": torch.ones(2, dtype=torch.bool),
            },
        ]
        stats = compute_feature_stats(samples, keys=["x"])
        mean, std = stats["x"]
        self.assertTrue(torch.allclose(mean, torch.tensor([4.0, 40.0])))
        self.assertEqual(mean.shape, (2,))
        self.assertTrue((std > 0).all())

    def test_ignores_masked_positions(self):
        samples = [
            {
                "x": torch.tensor([[2.0], [999.0]]),  # second row is padding
                "x_mask": torch.tensor([True, False]),
            }
        ]
        stats = compute_feature_stats(samples, keys=["x"])
        mean, _ = stats["x"]
        self.assertTrue(torch.allclose(mean, torch.tensor([2.0])))

    def test_vector_scalar_key(self):
        samples = [
            {"s": torch.tensor(2.0), "s_mask": torch.tensor(True)},
            {"s": torch.tensor(4.0), "s_mask": torch.tensor(True)},
            {"s": torch.tensor(0.0), "s_mask": torch.tensor(False)},  # ignored
        ]
        stats = compute_feature_stats(samples, keys=["s"])
        mean, _ = stats["s"]
        self.assertAlmostEqual(float(mean), 3.0)

    def test_missing_key_across_all_raises(self):
        samples = [{"y": torch.ones(2), "y_mask": torch.ones(2, dtype=torch.bool)}]
        with self.assertRaises(ValueError):
            compute_feature_stats(samples, keys=["x"])

    def test_stats_feed_normalize_to_zero_mean(self):
        samples = [
            {"x": torch.tensor([[1.0], [3.0]]), "x_mask": torch.ones(2, dtype=torch.bool)},
            {"x": torch.tensor([[5.0], [7.0]]), "x_mask": torch.ones(2, dtype=torch.bool)},
        ]
        stats = compute_feature_stats(samples, keys=["x"])
        normalize = Normalize(stats)
        out = normalize({"x": torch.tensor([[4.0]]), "x_mask": torch.ones(1, dtype=torch.bool)})
        # 4.0 is the global mean -> normalizes to ~0.
        self.assertTrue(torch.allclose(out["x"], torch.zeros(1, 1), atol=1e-4))


class TestComputeStats(unittest.TestCase):
    def _samples(self):
        return [
            {"x": torch.tensor([[1.0], [2.0]]), "x_mask": torch.ones(2, dtype=torch.bool)},
            {"x": torch.tensor([[3.0], [4.0]]), "x_mask": torch.ones(2, dtype=torch.bool)},
        ]

    def test_full_stat_set(self):
        stats = compute_stats(self._samples(), keys=["x"])["x"]
        self.assertEqual(set(stats), {"mean", "std", "min", "max", "median", "iqr"})
        self.assertAlmostEqual(float(stats["mean"]), 2.5, places=4)
        self.assertAlmostEqual(float(stats["min"]), 1.0, places=4)
        self.assertAlmostEqual(float(stats["max"]), 4.0, places=4)

    def test_missing_key_raises(self):
        with self.assertRaises(ValueError):
            compute_stats(self._samples(), keys=["missing"])

    def test_save_load_round_trip(self):
        stats = compute_stats(self._samples(), keys=["x"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            save_stats(stats, path)
            loaded = load_stats(path)
        for field in stats["x"]:
            self.assertTrue(torch.allclose(stats["x"][field], loaded["x"][field], atol=1e-5))


class TestComputeDatasetStats(unittest.TestCase):
    def _samples(self):
        return [
            {
                "dataset": "A",
                "x": torch.tensor([[1.0, 10.0], [3.0, 30.0]]),
                "x_mask": torch.ones(2, dtype=torch.bool),
            },
            {
                "dataset": "A",
                "x": torch.tensor([[5.0, 50.0], [7.0, 70.0]]),
                "x_mask": torch.ones(2, dtype=torch.bool),
            },
            {
                "dataset": "B",
                "x": torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
                "x_mask": torch.ones(2, dtype=torch.bool),
            },
        ]

    def test_groups_by_dataset(self):
        stats = compute_dataset_stats(self._samples(), keys=["x"])
        self.assertEqual(set(stats), {"A", "B"})
        self.assertTrue(torch.allclose(stats["A"]["x"]["mean"], torch.tensor([4.0, 40.0])))
        self.assertTrue(torch.allclose(stats["B"]["x"]["mean"], torch.tensor([1.0, 1.0])))

    def test_excludes_masked_positions(self):
        samples = [
            {
                "dataset": "A",
                "x": torch.tensor([[2.0], [999.0]]),
                "x_mask": torch.tensor([True, False]),
            },
        ]
        stats = compute_dataset_stats(samples, keys=["x"])
        self.assertAlmostEqual(float(stats["A"]["x"]["mean"]), 2.0, places=4)

    def test_missing_dataset_key_raises(self):
        with self.assertRaises(ValueError):
            compute_dataset_stats(
                [{"x": torch.tensor([[1.0]]), "x_mask": torch.tensor([True])}], keys=["x"]
            )

    def test_key_absent_from_one_dataset(self):
        samples = [
            {
                "dataset": "A",
                "x": torch.tensor([[1.0], [3.0]]),
                "x_mask": torch.ones(2, dtype=torch.bool),
            },
            {
                "dataset": "B",
                "y": torch.tensor([[2.0], [4.0]]),
                "y_mask": torch.ones(2, dtype=torch.bool),
            },
        ]
        stats = compute_dataset_stats(samples, keys=["x", "y"])
        self.assertIn("x", stats["A"])
        self.assertNotIn("x", stats["B"])  # B never provided x
        self.assertIn("y", stats["B"])

    def test_nested_save_load_round_trip(self):
        stats = compute_dataset_stats(self._samples(), keys=["x"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ds_stats.json"
            save_stats(stats, path)
            loaded = load_stats(path)
        for ds in stats:
            for field in stats[ds]["x"]:
                self.assertTrue(
                    torch.allclose(stats[ds]["x"][field], loaded[ds]["x"][field], atol=1e-5)
                )


if __name__ == "__main__":
    unittest.main()
