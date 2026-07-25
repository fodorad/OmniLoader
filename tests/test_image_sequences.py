"""End-to-end acceptance test for structured (image-sequence) TensorSpec values.

Mixes a dataset that provides an image-sequence feature ``(T, C, H, W)`` with one
that doesn't, through a plain OmniLoader + DataLoader + unified_collate, and checks
the union schema, batch shape and masking behave exactly like any other sequence.
"""

import unittest

import torch
from torch.utils.data import DataLoader

from omniloader import OmniLoader, unified_collate
from omniloader.data.datasets import DictTensorDataset
from omniloader.schema.spec import DatasetSchema, TensorSpec

T, C, H, W = 5, 3, 8, 8


class TestImageSequenceEndToEnd(unittest.TestCase):
    def setUp(self):
        self.ds_with_image = DictTensorDataset(
            {
                "eye_image": torch.randn(6, T, C, H, W),
                "label": torch.randint(0, 2, (6,)).float(),
            }
        )
        self.schema_with_image = DatasetSchema(
            features=[TensorSpec("eye_image", time_dim=T, shape=(C, H, W))],
            targets=[TensorSpec("label")],
        )
        self.ds_without_image = DictTensorDataset({"label": torch.randint(0, 2, (4,)).float()})
        self.schema_without_image = DatasetSchema(targets=[TensorSpec("label")])
        self.omni = OmniLoader(
            [self.ds_with_image, self.ds_without_image],
            [self.schema_with_image, self.schema_without_image],
        )

    def test_union_schema_includes_image_spec(self):
        spec = self.omni.schema.spec("eye_image")
        self.assertEqual(spec.trailing_shape, (C, H, W))
        self.assertEqual(spec.value_shape, (T, C, H, W))

    def test_batches_through_plain_dataloader(self):
        loader = DataLoader(self.omni, batch_size=4, collate_fn=unified_collate)
        batch = next(iter(loader))
        self.assertEqual(batch["eye_image"].shape, (4, T, C, H, W))
        self.assertEqual(batch["eye_image_mask"].shape, (4, T))
        self.assertEqual(batch["label"].shape, (4,))

    def test_masked_placeholder_for_dataset_lacking_image_key(self):
        # Samples 6..9 come from ds_without_image and must be all-False masked.
        loader = DataLoader(self.omni, batch_size=len(self.omni), collate_fn=unified_collate)
        batch = next(iter(loader))
        without_image_mask = batch["eye_image_mask"][6:]
        self.assertFalse(without_image_mask.any())
        self.assertTrue(torch.all(batch["eye_image"][6:] == 0.0))


if __name__ == "__main__":
    unittest.main()
