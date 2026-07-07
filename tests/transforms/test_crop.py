import unittest

import torch

from omniloader.schema.spec import DatasetSchema, TensorSpec, UnifiedSchema
from omniloader.transforms.crop import CenterCrop, RandomCrop


def schema():
    return UnifiedSchema(
        [
            DatasetSchema(
                features=[TensorSpec("feat", feature_dim=3, time_dim=10)],
                targets=[TensorSpec("seq_target", time_dim=10, placeholder=-1.0)],
            )
        ]
    )


def sample():
    return {
        "feat": torch.arange(10 * 3, dtype=torch.float32).reshape(10, 3),
        "feat_mask": torch.ones(10, dtype=torch.bool),
        "seq_target": torch.arange(10, dtype=torch.float32),
        "seq_target_mask": torch.ones(10, dtype=torch.bool),
    }


class TestRandomCrop(unittest.TestCase):
    def test_crops_to_length(self):
        out = RandomCrop(4, schema=schema())(sample(), generator=torch.Generator().manual_seed(0))
        self.assertEqual(out["feat"].shape, (4, 3))
        self.assertEqual(out["seq_target"].shape, (4,))
        self.assertEqual(out["feat_mask"].shape, (4,))

    def test_shared_offset_keeps_alignment(self):
        # feat[t, 0] == 3*t and seq_target[t] == t; after a shared crop the
        # relation feat[i,0] == 3*seq_target[i] must still hold.
        out = RandomCrop(4, schema=schema())(sample(), generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.allclose(out["feat"][:, 0], 3.0 * out["seq_target"]))

    def test_skipped_in_eval(self):
        out = RandomCrop(4, schema=schema())(sample(), training=False)
        self.assertEqual(out["feat"].shape, (10, 3))  # unchanged

    def test_requires_keys_or_schema(self):
        with self.assertRaises(ValueError):
            RandomCrop(4)


class TestCenterCrop(unittest.TestCase):
    def test_centered_window(self):
        out = CenterCrop(4, keys=["feat", "seq_target"])(sample())
        # valid length 10, length 4 -> start = 3.
        self.assertTrue(torch.allclose(out["seq_target"], torch.arange(3, 7, dtype=torch.float32)))

    def test_runs_in_eval(self):
        out = CenterCrop(4, keys=["feat"])(sample(), training=False)
        self.assertEqual(out["feat"].shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
