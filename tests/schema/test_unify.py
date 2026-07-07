import unittest

import torch

from omniloader.schema.spec import (
    DatasetSchema,
    TensorSpec,
    UnifiedSchema,
)
from omniloader.schema.unify import SampleUnifier


def build_schema():
    a = DatasetSchema(
        features=[TensorSpec("video", feature_dim=4, time_dim=6)],
        targets=[TensorSpec("valence", time_dim=6, placeholder=-5.0)],
    )
    b = DatasetSchema(
        features=[TensorSpec("audio", feature_dim=2, time_dim=6)],
        targets=[TensorSpec("sentiment", placeholder=-5.0)],
    )
    return UnifiedSchema([a, b])


class TestSampleUnifier(unittest.TestCase):
    def setUp(self):
        self.schema = build_schema()
        self.unifier = SampleUnifier(self.schema)

    def test_missing_keys_become_placeholders(self):
        sample = {"video": torch.randn(6, 4), "valence": torch.randn(6), "dataset": "A"}
        out = self.unifier(sample)
        # Present keys have real masks.
        self.assertTrue(out["video_mask"].all())
        self.assertTrue(out["valence_mask"].all())
        # Absent keys are placeholders with all-False masks.
        self.assertEqual(out["audio"].shape, (6, 2))
        self.assertFalse(out["audio_mask"].any())
        self.assertEqual(out["sentiment"].shape, ())
        self.assertFalse(out["sentiment_mask"].any())
        self.assertTrue(torch.all(out["sentiment"] == -5.0))
        self.assertEqual(out["dataset"], "A")

    def test_padding_short_sequence(self):
        out = self.unifier({"video": torch.randn(4, 4)})
        self.assertEqual(out["video"].shape, (6, 4))
        self.assertTrue(out["video_mask"][:4].all())
        self.assertFalse(out["video_mask"][4:].any())

    def test_cropping_long_sequence(self):
        out = self.unifier({"video": torch.randn(9, 4)})
        self.assertEqual(out["video"].shape, (6, 4))
        self.assertTrue(out["video_mask"].all())

    def test_vector_target_scalar_mask(self):
        out = self.unifier({"sentiment": torch.tensor(0.7)})
        self.assertEqual(out["sentiment_mask"].shape, ())
        self.assertTrue(bool(out["sentiment_mask"]))

    def test_supplied_sequence_mask_is_combined(self):
        mask = torch.tensor([True, True, True, False, False, False])
        out = self.unifier({"video": torch.randn(6, 4), "video_mask": mask})
        self.assertTrue(out["video_mask"][:3].all())
        self.assertFalse(out["video_mask"][3:].any())

    def test_supplied_vector_mask_respected(self):
        out = self.unifier({"sentiment": torch.tensor(0.1), "sentiment_mask": torch.tensor(False)})
        self.assertFalse(bool(out["sentiment_mask"]))

    def test_accepts_non_tensor_values(self):
        import numpy as np

        out = self.unifier({"video": np.random.randn(6, 4).astype("float32")})
        self.assertEqual(out["video"].shape, (6, 4))
        self.assertTrue(torch.is_tensor(out["video"]))

    def test_no_pad_mode_uses_supplied_or_full_mask(self):
        unifier = SampleUnifier(self.schema, pad_features=False)
        out = unifier({"video": torch.randn(6, 4)})
        self.assertEqual(out["video"].shape, (6, 4))
        self.assertTrue(out["video_mask"].all())
        supplied = torch.tensor([True] * 3 + [False] * 3)
        out2 = unifier({"video": torch.randn(6, 4), "video_mask": supplied})
        self.assertFalse(out2["video_mask"][3:].any())


if __name__ == "__main__":
    unittest.main()
