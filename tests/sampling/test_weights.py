import unittest

import torch

from omniloader.sampling.weights import (
    class_histogram,
    class_weights_for_loss,
    class_weights_for_sampler,
)


def scalar_samples(class_ids, valid=None):
    valid = valid if valid is not None else [True] * len(class_ids)
    return [{"y": torch.tensor(c), "y_mask": torch.tensor(v)} for c, v in zip(class_ids, valid)]


class TestClassWeightsForSampler(unittest.TestCase):
    def test_inverse_frequency(self):
        # class 0 appears 3x, class 1 once -> class 1 weighted 3x higher.
        weights = class_weights_for_sampler(scalar_samples([0, 0, 0, 1]), "y")
        self.assertAlmostEqual(weights[0], 1 / 3)
        self.assertAlmostEqual(weights[3], 1.0)
        self.assertGreater(weights[3], weights[0])

    def test_flattens_class_mass(self):
        weights = class_weights_for_sampler(scalar_samples([0, 0, 0, 1]), "y")
        mass0 = sum(w for c, w in zip([0, 0, 0, 1], weights) if c == 0)
        mass1 = sum(w for c, w in zip([0, 0, 0, 1], weights) if c == 1)
        self.assertAlmostEqual(mass0, mass1)  # equal total mass per class

    def test_invalid_and_missing_get_zero(self):
        samples = scalar_samples([0, 1], valid=[True, False])
        samples.append({"other": torch.tensor(1)})  # missing target key
        weights = class_weights_for_sampler(samples, "y")
        self.assertEqual(weights[1], 0.0)  # invalid mask
        self.assertEqual(weights[2], 0.0)  # missing key

    def test_sequence_target_uses_mode(self):
        samples = [
            {
                "y": torch.tensor([2, 2, 5]),
                "y_mask": torch.tensor([True, True, True]),
            },
            {"y": torch.tensor([5, 5, 5]), "y_mask": torch.ones(3, dtype=torch.bool)},
        ]
        weights = class_weights_for_sampler(samples, "y")
        # Both samples' representative class is 2 and 5 respectively (each once).
        self.assertEqual(weights, [1.0, 1.0])

    def test_sequence_all_invalid(self):
        samples = [{"y": torch.tensor([1, 1]), "y_mask": torch.zeros(2, dtype=torch.bool)}]
        self.assertEqual(class_weights_for_sampler(samples, "y"), [0.0])


class TestClassHistogram(unittest.TestCase):
    def test_counts_scalar_targets(self):
        hist = class_histogram(scalar_samples([0, 0, 0, 1]), "y")
        self.assertTrue(torch.equal(hist, torch.tensor([3, 1])))
        self.assertEqual(hist.dtype, torch.int64)

    def test_counts_every_valid_step_of_sequence(self):
        # Unlike the per-sample sampler weights, all valid steps are counted.
        samples = [{"y": torch.tensor([2, 2, 5]), "y_mask": torch.tensor([True, True, False])}]
        hist = class_histogram(samples, "y", num_classes=6)
        self.assertEqual(hist.tolist(), [0, 0, 2, 0, 0, 0])  # step 5 masked out

    def test_num_classes_pads_zero_classes(self):
        hist = class_histogram(scalar_samples([0, 2]), "y", num_classes=4)
        self.assertEqual(hist.tolist(), [1, 0, 1, 0])

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            class_histogram(scalar_samples([0, 5]), "y", num_classes=3)


class TestClassWeightsForLoss(unittest.TestCase):
    def test_inverse_weights_average_to_one(self):
        w = class_weights_for_loss(scalar_samples([0, 0, 0, 1]), "y", scheme="inverse")
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)  # normalized
        self.assertGreater(float(w[1]), float(w[0]))  # rare class up-weighted
        # inverse frequency ratio: count0/count1 = 3
        self.assertAlmostEqual(float(w[1] / w[0]), 3.0, places=4)

    def test_effective_scheme_runs_and_balances(self):
        samples = scalar_samples([0, 0, 0, 0, 1])
        w = class_weights_for_loss(samples, "y", scheme="effective", beta=0.9)
        self.assertEqual(w.shape, (2,))
        self.assertGreater(float(w[1]), float(w[0]))

    def test_zero_count_class_gets_zero_weight(self):
        w = class_weights_for_loss(scalar_samples([0, 1]), "y", num_classes=3)
        self.assertEqual(float(w[2]), 0.0)  # class 2 absent

    def test_usable_as_loss_weight(self):
        w = class_weights_for_loss(scalar_samples([0, 0, 1]), "y")
        loss = torch.nn.CrossEntropyLoss(weight=w)
        out = loss(torch.randn(3, 2), torch.tensor([0, 1, 0]))
        self.assertTrue(torch.isfinite(out))

    def test_bad_scheme_raises(self):
        with self.assertRaises(ValueError):
            class_weights_for_loss(scalar_samples([0, 1]), "y", scheme="bogus")


if __name__ == "__main__":
    unittest.main()
