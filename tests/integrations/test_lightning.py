import unittest

from omniloader.config import OmniConfig

try:
    from omniloader.integrations.lightning import OmniDataModule

    HAS_LIGHTNING = True
except ModuleNotFoundError:
    HAS_LIGHTNING = False

from tests.fixtures import sentiment_dataset, valence_dataset


@unittest.skipUnless(HAS_LIGHTNING, "PyTorch Lightning not installed")
class TestOmniDataModule(unittest.TestCase):
    def setUp(self):
        self.ds_a, self.schema_a = valence_dataset(n=40)
        self.ds_b, self.schema_b = sentiment_dataset(n=200)
        self.config = OmniConfig(
            batch_size=8,
            strategy="temperature",
            strategy_kwargs={"temperature": 2.0},
        )

    def test_train_val_test_loaders(self):
        dm = OmniDataModule(
            self.config,
            train=([self.ds_a, self.ds_b], [self.schema_a, self.schema_b]),
            valid=([self.ds_a], [self.schema_a]),
            test=([self.ds_b], [self.schema_b]),
        )
        dm.setup()
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        self.assertEqual(batch["video"].shape[0], 8)
        self.assertEqual(batch["valence_mask"].shape[0], 8)

        self.assertIsNotNone(dm.val_dataloader())
        self.assertIsNotNone(dm.test_dataloader())

    def test_missing_splits_return_none(self):
        dm = OmniDataModule(self.config)
        dm.setup()
        self.assertIsNone(dm.train_dataloader())
        self.assertIsNone(dm.val_dataloader())
        self.assertIsNone(dm.test_dataloader())

    def test_set_epoch_propagates(self):
        dm = OmniDataModule(
            self.config,
            train=([self.ds_a, self.ds_b], [self.schema_a, self.schema_b]),
            valid=([self.ds_a], [self.schema_a]),
        )
        dm.setup()
        dm.set_epoch(3)
        self.assertEqual(dm.loaders["train"].epoch, 3)
        self.assertEqual(dm.loaders["valid"].epoch, 3)
        self.assertEqual(dm.samplers["train"].epoch, 3)

    def test_worker_init_fn_is_set(self):
        from omniloader.utils.seeding import seed_worker

        dm = OmniDataModule(self.config, train=([self.ds_a], [self.schema_a]))
        dm.setup()
        self.assertIs(dm.train_dataloader().worker_init_fn, seed_worker)


if __name__ == "__main__":
    unittest.main()
