import tempfile
import unittest
from pathlib import Path

import torch

from omniloader.schema.spec import DatasetSchema, TensorSpec, UnifiedSchema
from omniloader.transforms import (
    Compose,
    FeatureDropout,
    FeatureMasking,
    GaussianNoise,
    Normalize,
    SpanMasking,
    TimeWarp,
    build_transform,
    save_stats,
)


def build_schema():
    a = DatasetSchema(
        features=[
            TensorSpec("video", feature_dim=4, time_dim=6),
            TensorSpec("audio", feature_dim=2, time_dim=6),
        ],
        targets=[TensorSpec("valence", time_dim=6, placeholder=-5.0)],
    )
    return UnifiedSchema([a])


def make_sample():
    return {
        "video": torch.ones(6, 4),
        "video_mask": torch.tensor([True, True, True, True, False, False]),
        "audio": torch.ones(6, 2),
        "audio_mask": torch.ones(6, dtype=torch.bool),
        "valence": torch.zeros(6),
        "valence_mask": torch.ones(6, dtype=torch.bool),
    }


class TestNormalize(unittest.TestCase):
    def test_standardizes_only_valid_positions(self):
        tf = Normalize({"video": (1.0, 2.0)})  # (x-1)/2 -> valid ones become 0
        out = tf(make_sample())
        self.assertTrue(torch.allclose(out["video"][:4], torch.zeros(4, 4)))
        # Invalid (padding) rows are left untouched at their original value (1).
        self.assertTrue(torch.allclose(out["video"][4:], torch.ones(2, 4)))

    def test_accepts_dict_stats(self):
        tf = Normalize({"video": {"mean": 1.0, "std": 2.0}})
        out = tf(make_sample())
        self.assertTrue(torch.allclose(out["video"][0], torch.zeros(4)))

    def test_runs_in_eval_mode(self):
        tf = Normalize({"video": (1.0, 2.0)})
        out = tf(make_sample(), training=False)  # train_only is False
        self.assertTrue(torch.allclose(out["video"][0], torch.zeros(4)))

    def test_missing_key_ignored(self):
        tf = Normalize({"missing": (0.0, 1.0)})
        out = tf(make_sample())
        self.assertIn("video", out)


class TestGaussianNoise(unittest.TestCase):
    def test_adds_noise_where_valid(self):
        gen = torch.Generator().manual_seed(0)
        tf = GaussianNoise(keys=["video"], std=0.5, p=1.0)
        out = tf(make_sample(), generator=gen)
        # Valid rows changed, padding rows untouched.
        self.assertFalse(torch.allclose(out["video"][:4], torch.ones(4, 4)))
        self.assertTrue(torch.allclose(out["video"][4:], torch.ones(2, 4)))

    def test_skipped_in_eval(self):
        tf = GaussianNoise(keys=["video"], std=0.5, p=1.0)
        out = tf(make_sample(), training=False)
        self.assertTrue(torch.allclose(out["video"], torch.ones(6, 4)))

    def test_p_zero_no_change(self):
        tf = GaussianNoise(keys=["video"], std=0.5, p=0.0)
        out = tf(make_sample(), generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.allclose(out["video"], torch.ones(6, 4)))


class TestFeatureDropout(unittest.TestCase):
    def test_drops_feature_sets_mask_false(self):
        tf = FeatureDropout(keys=["video", "audio"], p=1.0, keep_at_least_one=False)
        out = tf(make_sample(), generator=torch.Generator().manual_seed(0))
        self.assertTrue(torch.all(out["video"] == 0))
        self.assertFalse(out["video_mask"].any())
        self.assertFalse(out["audio_mask"].any())

    def test_keep_at_least_one(self):
        tf = FeatureDropout(keys=["video", "audio"], p=1.0, keep_at_least_one=True)
        out = tf(make_sample(), generator=torch.Generator().manual_seed(0))
        survivors = int(out["video_mask"].any()) + int(out["audio_mask"].any())
        self.assertGreaterEqual(survivors, 1)

    def test_uses_schema_feature_keys(self):
        tf = FeatureDropout(p=1.0, keep_at_least_one=False, schema=build_schema())
        self.assertEqual(sorted(tf.keys), ["audio", "video"])

    def test_skipped_in_eval(self):
        tf = FeatureDropout(keys=["video"], p=1.0)
        out = tf(make_sample(), training=False)
        self.assertTrue(out["video_mask"].any())


class TestSpanMasking(unittest.TestCase):
    def test_zeros_spans_keeps_mask(self):
        tf = SpanMasking(keys=["video"], num_spans=1, span_len=2, p=1.0)
        out = tf(make_sample(), generator=torch.Generator().manual_seed(0))
        # Some valid position got zeroed, but the validity mask is unchanged.
        self.assertTrue((out["video"][:4] == 0).any())
        self.assertTrue(out["video_mask"][:4].all())

    def test_resolves_sequence_keys_from_schema(self):
        tf = SpanMasking(schema=build_schema(), p=1.0)
        self.assertEqual(sorted(tf.keys), ["audio", "video"])

    def test_no_valid_positions_is_noop(self):
        sample = make_sample()
        sample["video_mask"] = torch.zeros(6, dtype=torch.bool)
        tf = SpanMasking(keys=["video"], p=1.0)
        out = tf(sample, generator=torch.Generator().manual_seed(0))
        self.assertTrue(torch.allclose(out["video"], torch.ones(6, 4)))

    def test_absent_key_skipped(self):
        tf = SpanMasking(keys=["missing"], p=1.0)
        out = tf(make_sample(), generator=torch.Generator().manual_seed(0))
        self.assertTrue(torch.allclose(out["video"], torch.ones(6, 4)))

    def test_requires_keys_or_schema(self):
        with self.assertRaises(ValueError):
            SpanMasking(schema=None)


class TestComposeAndBuild(unittest.TestCase):
    def test_compose_applies_in_order(self):
        tf = Compose([Normalize({"video": (1.0, 1.0)}), GaussianNoise(keys=["video"], p=0.0)])
        out = tf(make_sample())
        self.assertTrue(torch.allclose(out["video"][0], torch.zeros(4)))

    def test_compose_apply_direct(self):
        # apply() runs every child unconditionally (gate handled per-transform).
        tf = Compose([Normalize({"video": (1.0, 2.0)})])
        out = tf.apply(make_sample(), None)
        self.assertTrue(torch.allclose(out["video"][0], torch.zeros(4)))

    def test_build_transform_injects_schema(self):
        schema = build_schema()
        tf = build_transform(
            [
                {"name": "gaussian_noise", "std": 0.1, "p": 0.0},
                {"name": "span_masking", "p": 0.0},
            ],
            schema,
        )
        self.assertIsInstance(tf, Compose)
        # gaussian_noise defaults to all feature keys from the schema.
        self.assertEqual(sorted(tf.transforms[0].keys), ["audio", "video"])

    def test_build_transform_empty_is_none(self):
        self.assertIsNone(build_transform([], build_schema()))

    def test_build_transform_unknown_raises(self):
        with self.assertRaises(ValueError):
            build_transform([{"name": "bogus"}], build_schema())

    def test_build_transform_requires_keys_or_schema(self):
        with self.assertRaises(ValueError):
            build_transform([{"name": "gaussian_noise"}], schema=None)

    def test_build_registers_new_transforms(self):
        schema = build_schema()
        tf = build_transform(
            [
                {"name": "random_crop", "length": 4},
                {"name": "time_warp", "p": 0.0},
                {"name": "feature_masking", "p": 0.0},
                {"name": "instance_normalize"},
                {"name": "min_max_normalize", "stats": {"video": {"min": 0.0, "max": 1.0}}},
            ],
            schema,
        )
        self.assertIsInstance(tf, Compose)
        self.assertEqual(len(tf.transforms), 5)

    def test_build_transform_stats_path(self):
        # A config can reference a saved stats file instead of inlining the numbers.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            save_stats({"video": {"mean": [1.0] * 4, "std": [1.0] * 4}}, path)
            tf = build_transform([{"name": "normalize", "stats_path": str(path)}], build_schema())
            out = tf(make_sample())  # video is ones -> (1 - 1) / 1 = 0 on valid steps
            self.assertTrue(torch.allclose(out["video"][0], torch.zeros(4), atol=1e-4))

    def test_build_transform_per_dataset_stats_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp) / "ds.json"
            union_path = Path(tmp) / "union.json"
            save_stats({"A": {"video": {"mean": [0.0] * 4, "std": [1.0] * 4}}}, ds_path)
            save_stats({"video": {"mean": [1.0] * 4, "std": [1.0] * 4}}, union_path)
            tf = build_transform(
                [
                    {
                        "name": "per_dataset_normalize",
                        "stats_path": str(ds_path),
                        "union_stats_path": str(union_path),
                        "fallback": "union",
                    }
                ],
                build_schema(),
            )
            # make_sample has no `dataset` key -> union fallback: (1 - 1) / 1 = 0.
            out = tf(make_sample())
            self.assertTrue(torch.allclose(out["video"][0], torch.zeros(4), atol=1e-4))


class TestTimeWarp(unittest.TestCase):
    def test_preserves_length_and_dtype(self):
        tf = TimeWarp(keys=["video"], p=1.0)
        out = tf(make_sample(), generator=torch.Generator().manual_seed(0))
        self.assertEqual(out["video"].shape, (6, 4))
        self.assertEqual(out["video"].dtype, torch.float32)

    def test_scalar_sequence(self):
        sample = {
            "s": torch.arange(8, dtype=torch.float32),
            "s_mask": torch.ones(8, dtype=torch.bool),
        }
        out = TimeWarp(keys=["s"], p=1.0)(sample, generator=torch.Generator().manual_seed(1))
        self.assertEqual(out["s"].shape, (8,))

    def test_defaults_to_float_sequence_features(self):
        tf = TimeWarp(schema=build_schema())
        self.assertEqual(sorted(tf.keys), ["audio", "video"])

    def test_skipped_in_eval(self):
        out = TimeWarp(keys=["video"], p=1.0)(make_sample(), training=False)
        self.assertTrue(torch.equal(out["video"], torch.ones(6, 4)))

    def test_requires_keys_or_schema(self):
        with self.assertRaises(ValueError):
            TimeWarp()


class TestFeatureMasking(unittest.TestCase):
    def test_zeros_feature_band(self):
        tf = FeatureMasking(num_masks=2, max_width=3, keys=["video"], p=1.0)
        # The band width is random (may draw 0), so try a few seeds and require
        # that masking zeroes at least one whole feature column at some point.
        found = False
        for seed in range(10):
            out = tf(make_sample(), generator=torch.Generator().manual_seed(seed))
            if bool((out["video"] == 0).all(dim=0).any()):
                found = True
                break
        self.assertTrue(found)

    def test_skipped_in_eval(self):
        out = FeatureMasking(keys=["video"], p=1.0)(make_sample(), training=False)
        self.assertTrue(torch.equal(out["video"], torch.ones(6, 4)))

    def test_scalar_value_skipped(self):
        sample = {"s": torch.tensor(3.0), "s_mask": torch.tensor(True)}
        out = FeatureMasking(keys=["s"], p=1.0)(sample, generator=torch.Generator().manual_seed(0))
        self.assertEqual(float(out["s"]), 3.0)


if __name__ == "__main__":
    unittest.main()
