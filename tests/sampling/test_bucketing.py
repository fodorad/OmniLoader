import unittest

from omniloader import LengthBucketBatchSampler, OmniLoader, OmniSampler
from omniloader.sampling.strategies import ProportionalStrategy
from tests.fixtures import varlen_dataset


class TestLengthBucketBatchSampler(unittest.TestCase):
    def setUp(self):
        # 40 samples with lengths cycling 1..40 (widely varied).
        self.lengths = [((i * 7) % 40) + 1 for i in range(40)]
        self.ds, self.schema = varlen_dataset(self.lengths, time_dim=64)
        self.omni = OmniLoader([self.ds], [self.schema], pad_features=False)
        self.sampler = OmniSampler(ProportionalStrategy([40], seed=0))

    def _bucket(self, **kw):
        return LengthBucketBatchSampler(
            self.sampler, self.omni.sequence_lengths("feat"), batch_size=4, **kw
        )

    def test_validation(self):
        with self.assertRaises(ValueError):
            LengthBucketBatchSampler(self.sampler, self.lengths, batch_size=0)

    def test_sequence_lengths_match(self):
        self.assertEqual(self.omni.sequence_lengths("feat"), self.lengths)

    def test_batch_count_and_coverage(self):
        bucket = self._bucket(bucket_multiplier=2, shuffle=False)
        batches = list(bucket)
        self.assertEqual(len(batches), 10)  # 40 / 4
        self.assertEqual(len(bucket), 10)
        covered = sorted(i for b in batches for i in b)
        self.assertEqual(covered, list(range(40)))

    def test_batches_are_length_homogeneous(self):
        # With a large pool, each batch groups near-equal lengths.
        bucket = self._bucket(bucket_multiplier=100, shuffle=False)
        lengths = self.omni.sequence_lengths("feat")
        spreads = [max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in bucket]
        self.assertLessEqual(max(spreads), 3)  # tight within-batch length spread

    def test_drop_last(self):
        # batch_size 3 over 40 -> 13 full batches, 1 leftover dropped.
        bucket = LengthBucketBatchSampler(
            self.sampler, self.omni.sequence_lengths("feat"), batch_size=3, drop_last=True
        )
        batches = list(bucket)
        self.assertTrue(all(len(b) == 3 for b in batches))
        self.assertEqual(len(bucket), 13)

    def test_shuffle_reproducible_per_epoch(self):
        b1 = list(self._bucket(shuffle=True, seed=1))
        b2 = list(self._bucket(shuffle=True, seed=1))
        self.assertEqual(b1, b2)


if __name__ == "__main__":
    unittest.main()
