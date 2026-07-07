import unittest

from omniloader.sampling.strategies import (
    AnnealedTemperatureStrategy,
    FixedWeightStrategy,
    ProportionalStrategy,
    RoundRobinStrategy,
    SubsampleConfig,
    TemperatureStrategy,
    _apportion,
    _interleave,
)


class TestApportion(unittest.TestCase):
    def test_sums_to_total(self):
        counts = _apportion([1, 1, 2], total=10)
        self.assertEqual(sum(counts), 10)
        self.assertEqual(counts[2], 5)  # weight 2/4 of 10
        self.assertEqual(sorted(counts[:2]), [2, 3])

    def test_largest_remainder(self):
        # 3 equal weights over 10 -> [4, 3, 3] in some order summing to 10.
        counts = _apportion([1, 1, 1], total=10)
        self.assertEqual(sum(counts), 10)
        self.assertEqual(sorted(counts), [3, 3, 4])


class TestInterleave(unittest.TestCase):
    def test_round_robin_order(self):
        out = _interleave([[0, 1], [10, 11, 12]])
        self.assertEqual(out, [0, 10, 1, 11, 12])

    def test_empty(self):
        self.assertEqual(_interleave([[], []]), [])


class TestStrategies(unittest.TestCase):
    def setUp(self):
        self.sizes = [100, 400]

    def test_validation(self):
        with self.assertRaises(ValueError):
            ProportionalStrategy([])
        with self.assertRaises(ValueError):
            ProportionalStrategy([0, 5])

    def test_proportional_counts(self):
        s = ProportionalStrategy(self.sizes)
        self.assertEqual(s.target_counts(0), [100, 400])
        self.assertEqual(len(s), 500)

    def test_epoch_indices_are_valid_global(self):
        s = ProportionalStrategy(self.sizes, seed=0)
        idx = s.epoch_indices(0)
        self.assertEqual(len(idx), 500)
        self.assertTrue(all(0 <= i < 500 for i in idx))

    def test_temperature_sqrt_scaling(self):
        s = TemperatureStrategy(self.sizes, temperature=2.0)
        counts = s.target_counts(0)
        self.assertEqual(sum(counts), 500)
        # sqrt(100)=10, sqrt(400)=20 -> ratio 1:2 -> ~[167, 333].
        self.assertLess(counts[0], counts[1])
        self.assertAlmostEqual(counts[0] / sum(counts), 1 / 3, places=1)

    def test_temperature_validation(self):
        with self.assertRaises(ValueError):
            TemperatureStrategy(self.sizes, temperature=0.5)

    def test_fixed_uniform_default(self):
        s = FixedWeightStrategy(self.sizes)
        counts = s.target_counts(0)
        self.assertEqual(sum(counts), 500)
        self.assertEqual(counts[0], counts[1])  # 250 / 250

    def test_fixed_explicit_weights_and_epoch_size(self):
        s = FixedWeightStrategy(self.sizes, weights=[1.0, 3.0], epoch_size=100)
        self.assertEqual(s.target_counts(0), [25, 75])

    def test_fixed_weight_validation(self):
        with self.assertRaises(ValueError):
            FixedWeightStrategy(self.sizes, weights=[1.0])
        with self.assertRaises(ValueError):
            FixedWeightStrategy(self.sizes, weights=[0.0, 0.0])

    def test_round_robin_equal_and_interleaved(self):
        s = RoundRobinStrategy(self.sizes, samples_per_dataset=50, seed=0)
        self.assertEqual(s.target_counts(0), [50, 50])
        idx = s.epoch_indices(0)
        self.assertEqual(len(idx), 100)
        # Interleaved: first two indices come from different datasets.
        self.assertLess(idx[0], 100)
        self.assertGreaterEqual(idx[1], 100)

    def test_round_robin_default_smallest(self):
        s = RoundRobinStrategy(self.sizes)
        self.assertEqual(s.target_counts(0), [100, 100])

    def test_annealed_temperature_interpolates(self):
        s = AnnealedTemperatureStrategy(
            self.sizes, start_temperature=5.0, end_temperature=1.0, num_epochs=5
        )
        self.assertAlmostEqual(s.temperature_at(0), 5.0)
        self.assertAlmostEqual(s.temperature_at(4), 1.0)
        self.assertAlmostEqual(s.temperature_at(2), 3.0)
        self.assertAlmostEqual(s.temperature_at(99), 1.0)  # clamped to final
        # Early epoch (high temp) is closer to uniform than the final epoch.
        early = s.target_counts(0)
        late = s.target_counts(4)
        self.assertGreater(early[0], late[0])  # small dataset up-weighted early

    def test_annealed_validation(self):
        with self.assertRaises(ValueError):
            AnnealedTemperatureStrategy(self.sizes, start_temperature=0.5)
        with self.assertRaises(ValueError):
            AnnealedTemperatureStrategy(self.sizes, num_epochs=0)

    def test_annealed_single_epoch(self):
        s = AnnealedTemperatureStrategy(self.sizes, num_epochs=1, end_temperature=2.0)
        self.assertAlmostEqual(s.temperature_at(0), 2.0)

    def test_weighted_within_dataset_sampling(self):
        # Weight only the last index of the first dataset -> it dominates draws.
        weights = [0.0] * 99 + [1.0]
        s = ProportionalStrategy(
            self.sizes,
            subsample=[SubsampleConfig(sample_weights=weights, effective_size=20), None],
            seed=0,
        )
        idx = s.epoch_indices(0)
        first_ds = [i for i in idx if i < 100]
        self.assertEqual(len(first_ds), 20)
        self.assertTrue(all(i == 99 for i in first_ds))  # only the weighted index

    def test_subsample_effective_size_override(self):
        s = ProportionalStrategy(
            self.sizes,
            subsample=[SubsampleConfig(effective_size=10), None],
        )
        self.assertEqual(s._effective_counts(0), [10, 400])

    def test_subsample_length_mismatch(self):
        with self.assertRaises(ValueError):
            ProportionalStrategy(self.sizes, subsample=[None])

    def test_empty_epoch_indices_when_zeroed(self):
        s = ProportionalStrategy(
            self.sizes,
            subsample=[SubsampleConfig(effective_size=0), SubsampleConfig(effective_size=0)],
        )
        self.assertEqual(s.epoch_indices(0), [])

    def test_reproducible_epoch_indices(self):
        a = ProportionalStrategy(self.sizes, seed=7).epoch_indices(1)
        b = ProportionalStrategy(self.sizes, seed=7).epoch_indices(1)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
