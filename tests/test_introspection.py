import unittest

import torch

from omniloader import OmniLoader, describe, validate
from omniloader.data.datasets import DictTensorDataset
from omniloader.schema.spec import DatasetSchema, TensorSpec
from tests.fixtures import emotion_dataset, sentiment_dataset, valence_dataset


class TestDescribe(unittest.TestCase):
    def setUp(self):
        self.ds_a, self.schema_a = valence_dataset(n=10)  # video + valence
        self.ds_b, self.schema_b = sentiment_dataset(n=20)  # audio + sentiment
        self.ds_c, self.schema_c = emotion_dataset(n=8)  # video + emotion (int)

    def test_coverage_matrix(self):
        report = describe(
            [self.ds_a, self.ds_b, self.ds_c],
            [self.schema_a, self.schema_b, self.schema_c],
        )
        self.assertEqual(report.sizes, [10, 20, 8])
        self.assertEqual(
            set(report.union_keys), {"video", "audio", "valence", "sentiment", "emotion"}
        )
        # Dataset A provides video+valence but not audio/sentiment/emotion.
        cov_a = report.coverage[report.names[0]]
        self.assertTrue(cov_a["video"] and cov_a["valence"])
        self.assertFalse(cov_a["audio"] or cov_a["emotion"])

    def test_valid_fraction_and_str(self):
        report = describe([self.ds_a], [self.schema_a])
        self.assertAlmostEqual(report.valid_fraction["video"], 1.0)
        self.assertIn("coverage", str(report))

    def test_class_distribution_for_int_target(self):
        report = describe([self.ds_c], [self.schema_c])
        self.assertIn("emotion", report.class_distributions)
        self.assertEqual(sum(report.class_distributions["emotion"].values()), 8)

    def test_loader_describe_method(self):
        omni = OmniLoader([self.ds_a, self.ds_b], [self.schema_a, self.schema_b])
        report = omni.describe()
        self.assertEqual(report.sizes, [10, 20])

    def test_valid_fraction_reflects_placeholders(self):
        # video is provided only by A (10) not B (20) -> ~1/3 valid across the mix.
        report = describe([self.ds_a, self.ds_b], [self.schema_a, self.schema_b])
        self.assertLess(report.valid_fraction["video"], 0.5)

    def test_name_from_metadata(self):
        ds = DictTensorDataset({"x": torch.randn(3, 2)}, metadata={"dataset": ["MyDS"] * 3})
        schema = DatasetSchema(features=[TensorSpec("x", feature_dim=2)])
        report = describe([ds], [schema])
        self.assertEqual(report.names, ["MyDS"])

    def test_str_lists_classes(self):
        report = describe([self.ds_c], [self.schema_c])
        self.assertIn("classes", str(report))


class TestValidate(unittest.TestCase):
    def test_clean_datasets_have_no_issues(self):
        ds, schema = valence_dataset(n=4)
        self.assertEqual(validate([ds], [schema]), [])

    def test_detects_feature_dim_mismatch(self):
        # Schema declares feature_dim 99 but data provides 4.
        ds = DictTensorDataset({"x": torch.randn(4, 6, 4)})
        schema = DatasetSchema(features=[TensorSpec("x", feature_dim=99, time_dim=6)])
        issues = validate([ds], [schema])
        self.assertTrue(any("feature_dim" in msg for msg in issues))

    def test_detects_rank_mismatch(self):
        # Schema declares a scalar target but data provides a sequence.
        ds = DictTensorDataset({"y": torch.randn(4, 6)})
        schema = DatasetSchema(targets=[TensorSpec("y")])  # vector scalar ()
        issues = validate([ds], [schema])
        self.assertTrue(any("expected 0D" in msg for msg in issues))

    def test_strict_raises(self):
        ds = DictTensorDataset({"x": torch.randn(4, 6, 4)})
        schema = DatasetSchema(features=[TensorSpec("x", feature_dim=99, time_dim=6)])
        with self.assertRaises(ValueError):
            validate([ds], [schema], strict=True)

    def test_missing_key_reported(self):
        ds = DictTensorDataset({"x": torch.randn(4, 3)})
        schema = DatasetSchema(
            features=[TensorSpec("x", feature_dim=3), TensorSpec("z", feature_dim=2)]
        )
        issues = validate([ds], [schema])
        self.assertTrue(any("missing declared key 'z'" in msg for msg in issues))

    def test_loader_validate_method(self):
        ds, schema = valence_dataset(n=4)
        self.assertEqual(OmniLoader([ds], [schema]).validate(), [])


if __name__ == "__main__":
    unittest.main()
