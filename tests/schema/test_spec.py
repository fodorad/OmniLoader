import unittest

import torch

from omniloader.schema.spec import (
    DatasetSchema,
    TensorSpec,
    UnifiedSchema,
    resolve_dtype,
)


class TestResolveDtype(unittest.TestCase):
    def test_string_and_dtype(self):
        self.assertEqual(resolve_dtype("float32"), torch.float32)
        self.assertEqual(resolve_dtype(torch.int64), torch.int64)

    def test_unknown_dtype_raises(self):
        with self.assertRaises(ValueError):
            resolve_dtype("complex128")


class TestTensorSpec(unittest.TestCase):
    def test_sequence_of_vectors(self):
        spec = TensorSpec("video", feature_dim=32, time_dim=16)
        self.assertTrue(spec.is_sequence)
        self.assertEqual(spec.value_shape, (16, 32))
        self.assertEqual(spec.mask_shape, (16,))
        self.assertEqual(spec.placeholder_value().shape, (16, 32))
        self.assertFalse(spec.placeholder_mask().any())

    def test_sequence_of_scalars(self):
        spec = TensorSpec("valence", time_dim=10, placeholder=-5.0)
        self.assertTrue(spec.is_sequence)
        self.assertEqual(spec.value_shape, (10,))
        self.assertEqual(spec.mask_shape, (10,))
        self.assertTrue(torch.all(spec.placeholder_value() == -5.0))

    def test_scalar_vector(self):
        spec = TensorSpec("sentiment")
        self.assertFalse(spec.is_sequence)
        self.assertEqual(spec.value_shape, ())
        self.assertEqual(spec.mask_shape, ())

    def test_feature_vector(self):
        spec = TensorSpec("embedding", feature_dim=64)
        self.assertFalse(spec.is_sequence)
        self.assertEqual(spec.value_shape, (64,))
        self.assertEqual(spec.mask_shape, ())

    def test_negative_feature_dim_raises(self):
        with self.assertRaises(ValueError):
            TensorSpec("x", feature_dim=-1)

    def test_negative_time_dim_raises(self):
        with self.assertRaises(ValueError):
            TensorSpec("x", time_dim=0)

    def test_dtype_coerced_from_string(self):
        spec = TensorSpec("emotion", dtype="int64", placeholder=-1)
        self.assertEqual(spec.placeholder_value().dtype, torch.int64)


class TestDatasetSchema(unittest.TestCase):
    def test_keys_and_specs(self):
        schema = DatasetSchema(
            features=[TensorSpec("video", feature_dim=4, time_dim=3)],
            targets=[TensorSpec("sentiment")],
        )
        self.assertEqual(schema.keys, {"video", "sentiment"})
        self.assertEqual(len(schema.specs), 2)


class TestUnifiedSchema(unittest.TestCase):
    def setUp(self):
        self.a = DatasetSchema(
            features=[TensorSpec("video", feature_dim=4, time_dim=3)],
            targets=[TensorSpec("valence", time_dim=3)],
        )
        self.b = DatasetSchema(
            features=[
                TensorSpec("video", feature_dim=4, time_dim=3),
                TensorSpec("audio", feature_dim=2, time_dim=5),
            ],
            targets=[TensorSpec("sentiment")],
        )

    def test_union_dedup_and_order(self):
        schema = UnifiedSchema([self.a, self.b])
        self.assertEqual(schema.feature_keys, ["video", "audio"])
        self.assertEqual(schema.target_keys, ["valence", "sentiment"])
        self.assertEqual(len(schema), 4)
        self.assertEqual(schema.keys, {"video", "audio", "valence", "sentiment"})

    def test_spec_lookup(self):
        schema = UnifiedSchema([self.a, self.b])
        self.assertEqual(schema.spec("audio").feature_dim, 2)
        with self.assertRaises(KeyError):
            schema.spec("missing")

    def test_incompatible_specs_raise(self):
        bad = DatasetSchema(features=[TensorSpec("video", feature_dim=99, time_dim=3)])
        with self.assertRaises(ValueError):
            UnifiedSchema([self.a, bad])


if __name__ == "__main__":
    unittest.main()
