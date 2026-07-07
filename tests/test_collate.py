import unittest

import torch

from omniloader import OmniLoader
from omniloader.collate import DynamicCollator, unified_collate
from tests.fixtures import varlen_dataset


class TestUnifiedCollate(unittest.TestCase):
    def test_stacks_tensors_and_lists_metadata(self):
        batch = [
            {"x": torch.ones(3), "dataset": "A"},
            {"x": torch.zeros(3), "dataset": "B"},
        ]
        out = unified_collate(batch)
        self.assertEqual(out["x"].shape, (2, 3))
        self.assertEqual(out["dataset"], ["A", "B"])

    def test_empty_batch_raises(self):
        with self.assertRaises(ValueError):
            unified_collate([])


class TestDynamicCollator(unittest.TestCase):
    def setUp(self):
        # Native lengths 5 and 9; without dynamic padding they'd be forced to 32.
        self.ds, self.schema = varlen_dataset([5, 9], time_dim=32)
        self.omni = OmniLoader([self.ds], [self.schema], pad_features=False)
        self.collate = DynamicCollator(self.omni.schema)

    def test_pads_to_batch_max_not_time_dim(self):
        batch = [self.omni[0], self.omni[1]]
        out = self.collate(batch)
        self.assertEqual(out["feat"].shape, (2, 9, 4))  # batch max = 9, not time_dim 32
        self.assertEqual(out["feat_mask"].shape, (2, 9))
        # Sample 0 (len 5): first 5 valid, rest padding.
        self.assertTrue(out["feat_mask"][0, :5].all())
        self.assertFalse(out["feat_mask"][0, 5:].any())
        self.assertTrue(out["feat_mask"][1].all())  # sample 1 fills the batch length

    def test_scalar_target_and_metadata(self):
        out = self.collate([self.omni[0], self.omni[1]])
        self.assertEqual(out["label"].shape, (2,))
        self.assertEqual(out["label_mask"].shape, (2,))

    def test_keys_subset_leaves_others(self):
        collate = DynamicCollator(self.omni.schema, keys=[])  # no dynamic keys
        out = collate([self.omni[0]])
        # With no dynamic keys, feat stays at its native length (5) via plain stack.
        self.assertEqual(out["feat"].shape, (1, 5, 4))

    def test_empty_batch_raises(self):
        with self.assertRaises(ValueError):
            self.collate([])


if __name__ == "__main__":
    unittest.main()
