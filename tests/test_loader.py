import unittest

import torch
from torch.utils.data import DataLoader

from omniloader import (
    OmniLoader,
    TemperatureStrategy,
    unified_collate,
)
from tests.fixtures import (
    AUDIO_F,
    AUDIO_T,
    VIDEO_F,
    VIDEO_T,
    emotion_dataset,
    sentiment_dataset,
    valence_dataset,
)


class TestOmniLoader(unittest.TestCase):
    def setUp(self):
        self.ds_a, self.schema_a = valence_dataset(n=40)
        self.ds_b, self.schema_b = sentiment_dataset(n=200)
        self.omni = OmniLoader([self.ds_a, self.ds_b], [self.schema_a, self.schema_b])

    def test_length_and_sizes(self):
        self.assertEqual(len(self.omni), 240)
        self.assertEqual(self.omni.dataset_sizes, [40, 200])

    def test_unified_schema_union(self):
        self.assertEqual(self.omni.schema.keys, {"video", "audio", "valence", "sentiment"})

    def test_locate(self):
        self.assertEqual(self.omni.locate(0), (0, 0))
        self.assertEqual(self.omni.locate(39), (0, 39))
        self.assertEqual(self.omni.locate(40), (1, 0))
        self.assertEqual(self.omni.locate(-1), (1, 199))

    def test_locate_out_of_range(self):
        with self.assertRaises(IndexError):
            self.omni.locate(240)

    def test_getitem_from_first_dataset_masks(self):
        sample = self.omni[0]  # from valence dataset
        self.assertEqual(sample["video"].shape, (VIDEO_T, VIDEO_F))
        self.assertTrue(sample["valence_mask"].all())
        self.assertFalse(sample["audio_mask"].any())
        self.assertFalse(sample["sentiment_mask"].any())

    def test_getitem_from_second_dataset_masks(self):
        sample = self.omni[100]  # from sentiment dataset
        self.assertEqual(sample["audio"].shape, (AUDIO_T, AUDIO_F))
        self.assertTrue(sample["audio_mask"].all())
        self.assertTrue(bool(sample["sentiment_mask"]))
        self.assertFalse(sample["valence_mask"].any())

    def test_validation_empty(self):
        with self.assertRaises(ValueError):
            OmniLoader([], [])

    def test_validation_length_mismatch(self):
        with self.assertRaises(ValueError):
            OmniLoader([self.ds_a], [self.schema_a, self.schema_b])

    def test_end_to_end_dataloader_mixed_batch(self):
        sampler = self.omni.make_sampler(
            TemperatureStrategy(self.omni.dataset_sizes, temperature=2.0, seed=0)
        )
        loader = DataLoader(self.omni, batch_size=8, sampler=sampler, collate_fn=unified_collate)
        batch = next(iter(loader))
        self.assertEqual(batch["video"].shape, (8, VIDEO_T, VIDEO_F))
        self.assertEqual(batch["audio"].shape, (8, AUDIO_T, AUDIO_F))
        self.assertEqual(batch["valence"].shape, (8, VIDEO_T))
        self.assertEqual(batch["valence_mask"].shape, (8, VIDEO_T))
        self.assertEqual(batch["sentiment"].shape, (8,))
        self.assertEqual(batch["sentiment_mask"].shape, (8,))
        # Every column is either fully valid or fully placeholder per sample.
        self.assertEqual(batch["sentiment_mask"].dtype, torch.bool)

    def test_default_sampler_proportional(self):
        sampler = self.omni.make_sampler()
        self.assertEqual(len(sampler), 240)

    def test_transform_applied_after_unify(self):
        from omniloader.transforms import Normalize

        omni = OmniLoader(
            [self.ds_a],
            [self.schema_a],
            transform=Normalize({"video": (0.0, 1.0)}),
            training=True,
        )
        sample = omni[0]
        self.assertEqual(sample["video"].shape, (VIDEO_T, VIDEO_F))
        self.assertTrue(torch.is_tensor(sample["video"]))

    def test_three_datasets_shared_feature(self):
        ds_c, schema_c = emotion_dataset(n=12)
        omni = OmniLoader(
            [self.ds_a, self.ds_b, ds_c],
            [self.schema_a, self.schema_b, schema_c],
        )
        self.assertEqual(omni.schema.keys, {"video", "audio", "valence", "sentiment", "emotion"})
        sample = omni[len(self.ds_a) + len(self.ds_b)]  # first emotion sample
        self.assertEqual(sample["emotion"].dtype, torch.int64)
        self.assertTrue(bool(sample["emotion_mask"]))
        self.assertTrue(sample["video_mask"].all())


class TestReproducibleAugmentation(unittest.TestCase):
    def setUp(self):
        self.ds_a, self.schema_a = valence_dataset(n=8)
        from omniloader import GaussianNoise

        self.transform = GaussianNoise(keys=["video"], std=1.0, p=1.0)

    def _loader(self, seed, epoch):
        omni = OmniLoader([self.ds_a], [self.schema_a], transform=self.transform, seed=seed)
        omni.set_epoch(epoch)
        return omni

    def test_same_seed_epoch_reproducible(self):
        a = self._loader(0, 0)[3]
        b = self._loader(0, 0)[3]
        self.assertTrue(torch.equal(a["video"], b["video"]))

    def test_epoch_changes_augmentation(self):
        a = self._loader(0, 0)[3]
        b = self._loader(0, 1)[3]
        self.assertFalse(torch.equal(a["video"], b["video"]))

    def test_index_changes_augmentation(self):
        omni = self._loader(0, 0)
        self.assertFalse(torch.equal(omni[0]["video"], omni[1]["video"]))


if __name__ == "__main__":
    unittest.main()
