import unittest

import torch

from omniloader.schema.spec import DatasetSchema, TensorSpec, UnifiedSchema
from omniloader.transforms.normalize import (
    InstanceNormalize,
    MinMaxNormalize,
    Normalize,
    PerDatasetNormalize,
    RobustNormalize,
)


def make_sample(values):
    t = torch.tensor(values, dtype=torch.float32)
    return {"x": t, "x_mask": torch.ones(t.shape[0], dtype=torch.bool)}


class TestStatsNormalizers(unittest.TestCase):
    def test_normalize_pair_and_dict(self):
        out = Normalize({"x": (2.0, 2.0)})(make_sample([[2.0], [4.0]]))
        self.assertTrue(torch.allclose(out["x"], torch.tensor([[0.0], [1.0]]), atol=1e-4))
        out2 = Normalize({"x": {"mean": 2.0, "std": 2.0}})(make_sample([[2.0], [4.0]]))
        self.assertTrue(torch.allclose(out["x"], out2["x"]))

    def test_min_max_scales_to_unit(self):
        out = MinMaxNormalize({"x": {"min": 0.0, "max": 10.0}})(make_sample([[0.0], [5.0], [10.0]]))
        self.assertTrue(torch.allclose(out["x"], torch.tensor([[0.0], [0.5], [1.0]]), atol=1e-4))

    def test_robust_uses_median_iqr(self):
        out = RobustNormalize({"x": {"median": 4.0, "iqr": 2.0}})(make_sample([[4.0], [6.0]]))
        self.assertTrue(torch.allclose(out["x"], torch.tensor([[0.0], [1.0]]), atol=1e-4))

    def test_normalizers_run_in_eval(self):
        out = Normalize({"x": (0.0, 1.0)})(make_sample([[3.0]]), training=False)
        self.assertTrue(torch.allclose(out["x"], torch.tensor([[3.0]])))

    def test_only_valid_positions_touched(self):
        sample = {
            "x": torch.tensor([[2.0], [99.0]]),
            "x_mask": torch.tensor([True, False]),
        }
        out = Normalize({"x": (2.0, 1.0)})(sample)
        self.assertEqual(float(out["x"][1, 0]), 99.0)  # padding untouched


class TestInstanceNormalize(unittest.TestCase):
    def test_sequence_zero_mean_unit_std(self):
        out = InstanceNormalize(keys=["x"])(make_sample([[1.0], [3.0], [5.0]]))
        self.assertAlmostEqual(float(out["x"].mean()), 0.0, places=4)
        self.assertAlmostEqual(float(out["x"].std()), 1.0, places=2)

    def test_vector_value(self):
        sample = {"x": torch.tensor([2.0, 4.0, 6.0]), "x_mask": torch.tensor(True)}
        out = InstanceNormalize(keys=["x"])(sample)
        self.assertAlmostEqual(float(out["x"].mean()), 0.0, places=4)

    def test_all_invalid_is_noop(self):
        sample = {"x": torch.tensor([[1.0]]), "x_mask": torch.tensor([False])}
        out = InstanceNormalize(keys=["x"])(sample)
        self.assertEqual(float(out["x"][0, 0]), 1.0)

    def test_resolves_from_schema(self):
        schema = UnifiedSchema(
            [DatasetSchema(features=[TensorSpec("x", feature_dim=1, time_dim=3)])]
        )
        self.assertEqual(InstanceNormalize(schema=schema).keys, ["x"])


class TestPerDatasetNormalize(unittest.TestCase):
    def _stats(self):
        # A: mean 4 std 2; B: mean 10 std 5
        return {
            "A": {"x": {"mean": 4.0, "std": 2.0}},
            "B": {"x": {"mean": 10.0, "std": 5.0}},
        }

    def _sample(self, dataset, values):
        t = torch.tensor(values, dtype=torch.float32)
        return {"dataset": dataset, "x": t, "x_mask": torch.ones(t.shape[0], dtype=torch.bool)}

    def test_uses_own_dataset_stats(self):
        norm = PerDatasetNormalize(self._stats())
        out_a = norm(self._sample("A", [[4.0], [6.0]]))  # (x-4)/2
        self.assertTrue(torch.allclose(out_a["x"], torch.tensor([[0.0], [1.0]]), atol=1e-4))
        out_b = norm(self._sample("B", [[10.0], [15.0]]))  # (x-10)/5
        self.assertTrue(torch.allclose(out_b["x"], torch.tensor([[0.0], [1.0]]), atol=1e-4))

    def test_unknown_dataset_instance_fallback(self):
        norm = PerDatasetNormalize(self._stats(), fallback="instance")
        out = norm(self._sample("UNSEEN", [[2.0], [4.0], [6.0]]))
        # standardized by its own valid stats -> ~0 mean
        self.assertAlmostEqual(float(out["x"].mean()), 0.0, places=4)

    def test_unknown_dataset_union_fallback(self):
        norm = PerDatasetNormalize(
            self._stats(), fallback="union", union_stats={"x": {"mean": 0.0, "std": 2.0}}
        )
        out = norm(self._sample("UNSEEN", [[2.0], [4.0]]))
        self.assertTrue(torch.allclose(out["x"], torch.tensor([[1.0], [2.0]]), atol=1e-4))

    def test_unknown_dataset_identity_fallback(self):
        norm = PerDatasetNormalize(self._stats(), fallback="identity")
        out = norm(self._sample("UNSEEN", [[7.0], [9.0]]))
        self.assertTrue(torch.allclose(out["x"], torch.tensor([[7.0], [9.0]])))

    def test_padding_untouched(self):
        norm = PerDatasetNormalize(self._stats())
        sample = {
            "dataset": "A",
            "x": torch.tensor([[4.0], [99.0]]),
            "x_mask": torch.tensor([True, False]),
        }
        out = norm(sample)
        self.assertEqual(float(out["x"][1, 0]), 99.0)

    def test_runs_in_eval(self):
        norm = PerDatasetNormalize(self._stats())
        out = norm(self._sample("A", [[6.0]]), training=False)
        self.assertAlmostEqual(float(out["x"][0, 0]), 1.0, places=4)

    def test_bad_fallback_raises(self):
        with self.assertRaises(ValueError):
            PerDatasetNormalize(self._stats(), fallback="bogus")

    def test_union_fallback_requires_stats(self):
        with self.assertRaises(ValueError):
            PerDatasetNormalize(self._stats(), fallback="union")


if __name__ == "__main__":
    unittest.main()
