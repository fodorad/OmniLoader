import unittest

from omniloader.sampling.sampler import OmniSampler
from omniloader.sampling.strategies import ProportionalStrategy


class TestOmniSampler(unittest.TestCase):
    def setUp(self):
        self.strategy = ProportionalStrategy([30, 70], seed=5)

    def test_len(self):
        sampler = OmniSampler(self.strategy)
        self.assertEqual(len(sampler), 100)

    def test_iterates_all_indices(self):
        sampler = OmniSampler(self.strategy)
        indices = list(sampler)
        self.assertEqual(len(indices), 100)
        self.assertTrue(all(0 <= i < 100 for i in indices))

    def test_set_epoch_changes_order(self):
        sampler = OmniSampler(self.strategy)
        sampler.set_epoch(0)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertNotEqual(first, second)

    def test_same_epoch_reproducible(self):
        sampler = OmniSampler(self.strategy, epoch=2)
        self.assertEqual(list(sampler), list(sampler))


class TestOmniSamplerDistributed(unittest.TestCase):
    def setUp(self):
        self.strategy = ProportionalStrategy([30, 70], seed=5)

    def test_shards_are_disjoint_and_cover(self):
        r0 = list(OmniSampler(self.strategy, num_replicas=2, rank=0))
        r1 = list(OmniSampler(self.strategy, num_replicas=2, rank=1))
        self.assertEqual(len(r0), 50)
        self.assertEqual(len(r1), 50)
        self.assertEqual(set(r0) & set(r1), set())  # disjoint
        self.assertEqual(sorted(r0 + r1), list(range(100)))  # full coverage

    def test_len_matches_iteration(self):
        for rank in (0, 1, 2):
            s = OmniSampler(self.strategy, num_replicas=3, rank=rank)
            self.assertEqual(len(s), len(list(s)))

    def test_drop_last_even_split(self):
        s0 = OmniSampler(self.strategy, num_replicas=3, rank=0, drop_last=True)
        self.assertEqual(len(s0), 33)  # floor(100 / 3)
        self.assertEqual(len(list(s0)), 33)

    def test_pad_wraps_without_dropping(self):
        # 100 not divisible by 3 -> padded to 102, each rank gets 34.
        s = OmniSampler(self.strategy, num_replicas=3, rank=0)
        self.assertEqual(len(list(s)), 34)

    def test_rank_out_of_range(self):
        with self.assertRaises(ValueError):
            OmniSampler(self.strategy, num_replicas=2, rank=2)

    def test_single_replica_is_identity(self):
        plain = list(OmniSampler(self.strategy, epoch=1))
        sharded = list(OmniSampler(self.strategy, epoch=1, num_replicas=1, rank=0))
        self.assertEqual(plain, sharded)


if __name__ == "__main__":
    unittest.main()
