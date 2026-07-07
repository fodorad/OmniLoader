import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from omniloader.config import OmniConfig, seed_everything
from omniloader.sampling.strategies import (
    AnnealedTemperatureStrategy,
    FixedWeightStrategy,
    ProportionalStrategy,
    RoundRobinStrategy,
    TemperatureStrategy,
)
from omniloader.transforms import Compose
from tests.fixtures import sentiment_dataset, valence_dataset, varlen_dataset

CONFIG_DIR = Path(__file__).parent / "configs"


class TestSeedEverything(unittest.TestCase):
    def test_returns_seed_and_is_reproducible(self):
        self.assertEqual(seed_everything(123), 123)
        a = torch.randn(4)
        seed_everything(123)
        b = torch.randn(4)
        self.assertTrue(torch.equal(a, b))


class TestOmniConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = OmniConfig()
        self.assertEqual(cfg.strategy, "proportional")
        self.assertEqual(cfg.batch_size, 32)

    def test_from_dict_ignores_unknown(self):
        cfg = OmniConfig.from_dict({"seed": 9, "unknown": 1, "batch_size": 4})
        self.assertEqual(cfg.seed, 9)
        self.assertEqual(cfg.batch_size, 4)

    def test_yaml_round_trip(self):
        cfg = OmniConfig(seed=1, strategy="temperature", strategy_kwargs={"temperature": 2.0})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(cfg.to_dict()))
            loaded = OmniConfig.from_yaml(path)
        self.assertEqual(loaded.strategy, "temperature")
        self.assertEqual(loaded.strategy_kwargs["temperature"], 2.0)

    def test_json_round_trip(self):
        cfg = OmniConfig(seed=3, strategy="fixed", strategy_kwargs={"weights": [1, 2]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(cfg.to_dict()))
            loaded = OmniConfig.from_json(path)
        self.assertEqual(loaded.strategy, "fixed")

    def test_from_file_dispatch(self):
        cfg_json = OmniConfig.from_file(CONFIG_DIR / "temperature.json")
        self.assertEqual(cfg_json.strategy, "temperature")
        self.assertEqual(cfg_json.seed, 42)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yml"
            path.write_text(yaml.safe_dump({"strategy": "round_robin"}))
            self.assertEqual(OmniConfig.from_file(path).strategy, "round_robin")

    def test_from_file_bad_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.txt"
            path.write_text("{}")
            with self.assertRaises(ValueError):
                OmniConfig.from_file(path)

    def test_empty_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("")
            cfg = OmniConfig.from_yaml(path)
        self.assertEqual(cfg.strategy, "proportional")

    def test_build_each_strategy(self):
        sizes = [50, 150]
        cases = {
            "proportional": (ProportionalStrategy, {}),
            "temperature": (TemperatureStrategy, {"temperature": 2.0}),
            "annealed_temperature": (AnnealedTemperatureStrategy, {"num_epochs": 3}),
            "fixed": (FixedWeightStrategy, {"weights": [1, 2]}),
            "round_robin": (RoundRobinStrategy, {"samples_per_dataset": 10}),
        }
        for name, (cls, kwargs) in cases.items():
            cfg = OmniConfig(strategy=name, strategy_kwargs=kwargs)
            self.assertIsInstance(cfg.build_strategy(sizes), cls)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            OmniConfig(strategy="bogus").build_strategy([10, 10])


class TestConfigTransforms(unittest.TestCase):
    def setUp(self):
        _, self.schema_a = valence_dataset()
        from omniloader.schema.spec import UnifiedSchema

        self.schema = UnifiedSchema([self.schema_a])

    def test_augmentation_config_builds_transforms(self):
        cfg = OmniConfig.from_file(CONFIG_DIR / "augmentation.json")
        self.assertEqual(cfg.strategy, "annealed_temperature")
        train_tf = cfg.build_transform(self.schema, training=True)
        eval_tf = cfg.build_transform(self.schema, training=False)
        self.assertIsInstance(train_tf, Compose)
        self.assertEqual(len(train_tf.transforms), 4)  # normalize + 3 augmentations
        self.assertEqual(len(eval_tf.transforms), 1)  # normalize only

    def test_no_transforms_returns_none(self):
        cfg = OmniConfig()
        self.assertIsNone(cfg.build_transform(self.schema))


class TestConfigBuilders(unittest.TestCase):
    """End-to-end assembly of loaders/dataloaders from a single config."""

    def setUp(self):
        self.ds_a, self.schema_a = valence_dataset(n=40)
        self.ds_b, self.schema_b = sentiment_dataset(n=60)
        self.datasets = [self.ds_a, self.ds_b]
        self.schemas = [self.schema_a, self.schema_b]

    def test_new_fields_round_trip(self):
        cfg = OmniConfig(
            collate="mixup",
            collate_kwargs={"base": "dynamic", "alpha": 0.4},
            subsample=[None, {"policy": "exhaust", "effective_size": 10}],
            bucketing={"key": "video", "bucket_multiplier": 4},
            num_replicas=2,
            rank=1,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=3,
        )
        restored = OmniConfig.from_dict(cfg.to_dict())
        self.assertEqual(restored.collate, "mixup")
        self.assertEqual(restored.collate_kwargs["alpha"], 0.4)
        self.assertEqual(restored.subsample[1]["effective_size"], 10)
        self.assertEqual(restored.bucketing["key"], "video")
        self.assertEqual(restored.num_replicas, 2)
        self.assertTrue(restored.pin_memory)
        self.assertEqual(restored.prefetch_factor, 3)

    def test_build_subsample(self):
        from omniloader.sampling.strategies import SubsampleConfig
        from omniloader.sampling.subsamplers import ExhaustionPolicy

        self.assertIsNone(OmniConfig().build_subsample())
        cfg = OmniConfig(subsample=[None, {"policy": "exhaust", "effective_size": 5}])
        built = cfg.build_subsample()
        self.assertIsNone(built[0])
        self.assertIsInstance(built[1], SubsampleConfig)
        self.assertEqual(built[1].policy, ExhaustionPolicy.EXHAUST)
        self.assertEqual(built[1].effective_size, 5)

    def test_build_collate_variants(self):
        from omniloader.collate import DynamicCollator, unified_collate
        from omniloader.loader import OmniLoader
        from omniloader.transforms.mix import MixupCollator

        schema = OmniLoader(self.datasets, self.schemas).schema
        self.assertIs(OmniConfig(collate="unified").build_collate(schema), unified_collate)
        self.assertIsInstance(OmniConfig(collate="dynamic").build_collate(schema), DynamicCollator)
        mix = OmniConfig(collate="mixup", collate_kwargs={"base": "dynamic"})
        self.assertIsInstance(mix.build_collate(schema, training=True), MixupCollator)
        # mixup is a train-time augmentation: eval falls back to the base collate.
        self.assertIsInstance(mix.build_collate(schema, training=False), DynamicCollator)

    def test_build_collate_unknown_raises(self):
        from omniloader.schema.spec import UnifiedSchema

        schema = UnifiedSchema(self.schemas)
        with self.assertRaises(ValueError):
            OmniConfig(collate="bogus").build_collate(schema)
        with self.assertRaises(ValueError):
            OmniConfig(collate="mixup", collate_kwargs={"base": "bogus"}).build_collate(schema)

    def test_build_loader_from_given_datasets(self):
        loader = OmniConfig(seed=1).build_loader(self.datasets, self.schemas)
        self.assertEqual(loader.dataset_sizes, [40, 60])
        self.assertEqual(loader.seed, 1)

    def test_build_loader_requires_schemas(self):
        with self.assertRaises(ValueError):
            OmniConfig().build_loader(self.datasets)

    def test_build_loader_from_config_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.h5"
            from tests.fixtures import write_hdf5

            write_hdf5(path, subset="train", n=5)
            cfg = OmniConfig(
                datasets=[
                    {
                        "adapter": "hdf5",
                        "args": {"h5_path": str(path), "subset": "train"},
                        "schema": {
                            "features": [{"name": "video", "feature_dim": 32, "time_dim": 16}],
                            "targets": [{"name": "sentiment", "placeholder": -5.0}],
                        },
                    }
                ],
            )
            loader = cfg.build_loader()
            self.assertEqual(loader.dataset_sizes, [5])

    def test_build_dataloader_train(self):
        cfg = OmniConfig(batch_size=8, strategy="temperature", strategy_kwargs={"temperature": 2.0})
        dl = cfg.build_dataloader(self.datasets, self.schemas)
        batch = next(iter(dl))
        self.assertEqual(batch["video"].shape[0], 8)
        # union schema present with masks
        for key in ("video", "audio", "valence", "sentiment"):
            self.assertIn(f"{key}_mask", batch)

    def test_build_dataloader_eval_is_sequential(self):
        from torch.utils.data import SequentialSampler

        cfg = OmniConfig(batch_size=8)
        dl = cfg.build_dataloader(self.datasets, self.schemas, training=False)
        self.assertIsInstance(dl.sampler, SequentialSampler)
        seen = sum(b["video"].shape[0] for b in dl)
        self.assertEqual(seen, 100)  # every sample exactly once

    def test_build_dataloader_dynamic_collate(self):
        ds, schema = varlen_dataset([3, 7, 5, 9], time_dim=32)
        cfg = OmniConfig(batch_size=4, pad_features=False, collate="dynamic")
        dl = cfg.build_dataloader([ds], [schema])
        batch = next(iter(dl))
        self.assertLessEqual(batch["feat"].shape[1], 9)  # per-batch max, not fixed 32

    def test_build_dataloader_worker_knobs(self):
        # DataLoader only forks workers on iteration, so building is cheap here.
        cfg = OmniConfig(batch_size=8, num_workers=2, persistent_workers=True, prefetch_factor=4)
        dl = cfg.build_dataloader(self.datasets, self.schemas)
        self.assertEqual(dl.num_workers, 2)
        self.assertTrue(dl.persistent_workers)
        self.assertEqual(dl.prefetch_factor, 4)

    def test_build_dataloader_bucketing(self):
        ds, schema = varlen_dataset([2, 8, 3, 9, 4, 7], time_dim=32)
        cfg = OmniConfig(
            batch_size=2,
            pad_features=False,
            collate="dynamic",
            bucketing={"key": "feat", "bucket_multiplier": 3},
        )
        dl = cfg.build_dataloader([ds], [schema])
        batches = list(dl)
        self.assertTrue(all(b["feat"].shape[0] <= 2 for b in batches))
        self.assertEqual(sum(b["feat"].shape[0] for b in batches), 6)


if __name__ == "__main__":
    unittest.main()
