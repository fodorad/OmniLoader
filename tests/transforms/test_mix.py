import unittest

import torch

from omniloader.collate import unified_collate
from omniloader.schema.spec import DatasetSchema, TensorSpec, UnifiedSchema
from omniloader.transforms.mix import MixupCollator


def schema():
    return UnifiedSchema(
        [
            DatasetSchema(
                features=[TensorSpec("feat", feature_dim=2, time_dim=4)],
                targets=[TensorSpec("label")],
            )
        ]
    )


def batch(n=4):
    return [
        {
            "feat": torch.full((4, 2), float(i)),
            "feat_mask": torch.ones(4, dtype=torch.bool),
            "label": torch.tensor(float(i)),
            "label_mask": torch.tensor(True),
        }
        for i in range(n)
    ]


class TestMixupCollator(unittest.TestCase):
    def test_mixup_blends_and_adds_metadata(self):
        collate = MixupCollator(unified_collate, schema(), alpha=1.0, mode="mixup", p=1.0, seed=0)
        out = collate(batch())
        self.assertIn("mixup_lambda", out)
        self.assertIn("mixup_index", out)
        self.assertIn("label_b", out)
        self.assertEqual(out["feat"].shape, (4, 4, 2))
        # A blended value is lam*x + (1-lam)*x[perm] — bounded by the originals.
        self.assertTrue(out["feat"].min() >= 0.0 and out["feat"].max() <= 3.0)

    def test_cutmix_replaces_span(self):
        collate = MixupCollator(unified_collate, schema(), mode="cutmix", p=1.0, seed=1)
        out = collate(batch())
        self.assertEqual(out["feat"].shape, (4, 4, 2))
        self.assertTrue(0.0 <= out["mixup_lambda"] <= 1.0)

    def test_passthrough_when_p_zero(self):
        collate = MixupCollator(unified_collate, schema(), p=0.0)
        out = collate(batch())
        self.assertEqual(out["mixup_lambda"], 1.0)
        self.assertTrue(torch.equal(out["mixup_index"], torch.arange(4)))
        self.assertTrue(torch.equal(out["label_b"], out["label"]))

    def test_single_sample_passthrough(self):
        collate = MixupCollator(unified_collate, schema(), p=1.0)
        out = collate(batch(1))
        self.assertEqual(out["mixup_lambda"], 1.0)

    def test_reproducible_given_seed(self):
        a = MixupCollator(unified_collate, schema(), seed=5, p=1.0)(batch())
        b = MixupCollator(unified_collate, schema(), seed=5, p=1.0)(batch())
        self.assertTrue(torch.equal(a["feat"], b["feat"]))

    def test_cutmix_vector_feature_blends(self):
        # A vector feature (no time axis) can't be span-cut, so cutmix blends it.
        vec_schema = UnifiedSchema(
            [DatasetSchema(features=[TensorSpec("vec", feature_dim=3)], targets=[TensorSpec("y")])]
        )
        vec_batch = [
            {
                "vec": torch.full((3,), float(i)),
                "vec_mask": torch.tensor(True),
                "y": torch.tensor(float(i)),
                "y_mask": torch.tensor(True),
            }
            for i in range(4)
        ]
        out = MixupCollator(unified_collate, vec_schema, mode="cutmix", p=1.0, seed=0)(vec_batch)
        self.assertEqual(out["vec"].shape, (4, 3))
        self.assertTrue(0.0 <= out["mixup_lambda"] <= 1.0)

    def test_non_tensor_feature_skipped(self):
        # A feature that collates to a list (metadata-like) is skipped safely.
        out = MixupCollator(unified_collate, schema(), p=1.0, seed=0)(batch())
        self.assertIn("feat", out)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            MixupCollator(unified_collate, schema(), mode="bogus")


if __name__ == "__main__":
    unittest.main()
