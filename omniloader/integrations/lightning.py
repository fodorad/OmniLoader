"""Optional PyTorch Lightning integration.

This module wires OmniLoaders and mixing strategies into a
``LightningDataModule``. Lightning is an optional dependency: importing this
module without it installed raises a clear error, while the rest of OmniLoader
keeps working with plain ``torch.utils.data`` loaders. Install via the
``omniloader[lightning]`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import DataLoader, Sampler

from omniloader.collate import unified_collate
from omniloader.loader import OmniLoader, SizedDataset
from omniloader.utils.seeding import seed_worker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omniloader.config import OmniConfig
    from omniloader.sampling.sampler import OmniSampler
    from omniloader.schema.spec import DatasetSchema, UnifiedSchema

try:
    import lightning.pytorch as pl

    _LightningBase = pl.LightningDataModule
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without Lightning
    raise ModuleNotFoundError(
        "OmniDataModule requires PyTorch Lightning. "
        "Install with: pip install 'omniloader[lightning]'"
    ) from exc


class OmniDataModule(_LightningBase):
    """A ``LightningDataModule`` that mixes datasets per split via OmniLoader.

    The train split is mixed with the configured strategy through an
    :class:`~omniloader.sampling.sampler.OmniSampler`; validation and test splits are
    iterated in order (proportional, no shuffling) for stable evaluation.

    Args:
        config: The run configuration (seed, batch size, strategy, ...).
        train: ``(datasets, schemas)`` for the training split, or ``None``.
        valid: ``(datasets, schemas)`` for the validation split, or ``None``.
        test: ``(datasets, schemas)`` for the test split, or ``None``.
        schema: Optional shared unified schema applied to every split. When
            ``None``, each split unions its own schemas.

    """

    def __init__(
        self,
        config: OmniConfig,
        train: tuple[Sequence[SizedDataset], Sequence[DatasetSchema]] | None = None,
        valid: tuple[Sequence[SizedDataset], Sequence[DatasetSchema]] | None = None,
        test: tuple[Sequence[SizedDataset], Sequence[DatasetSchema]] | None = None,
        schema: UnifiedSchema | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._splits = {"train": train, "valid": valid, "test": test}
        self.schema = schema
        self.loaders: dict[str, OmniLoader] = {}
        self.samplers: dict[str, OmniSampler] = {}

    def _build(self, split: str) -> OmniLoader | None:
        """Build the OmniLoader for a split if it is configured."""
        spec = self._splits[split]
        if spec is None:
            return None
        datasets, schemas = spec
        training = split == "train"
        loader = OmniLoader(
            datasets,
            schemas,
            schema=self.schema,
            pad_features=self.config.pad_features,
            training=training,
            seed=self.config.seed,
        )
        # Normalization applies in every split; augmentations self-skip in eval.
        loader.transform = self.config.build_transform(loader.schema, training=training)
        return loader

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002 (Lightning API)
        """Instantiate the loaders and the training sampler.

        Args:
            stage: Lightning stage hint (unused; all splits are built lazily).

        """
        for split in self._splits:
            loader = self._build(split)
            if loader is not None:
                self.loaders[split] = loader
        if "train" in self.loaders:
            train = self.loaders["train"]
            strategy = self.config.build_strategy(train.dataset_sizes)
            # OmniSampler infers distributed rank/world-size from torch.distributed.
            self.samplers["train"] = train.make_sampler(strategy)

    def set_epoch(self, epoch: int) -> None:
        """Propagate the epoch to every loader and the training sampler.

        Ensures both the mixing draw and per-sample augmentation advance each
        epoch while staying reproducible.

        Args:
            epoch: The upcoming epoch number.

        """
        for loader in self.loaders.values():
            loader.set_epoch(epoch)
        sampler = self.samplers.get("train")
        if sampler is not None:
            sampler.set_epoch(epoch)

    def _dataloader(self, split: str, sampler: Sampler | None) -> DataLoader:
        """Construct a ``DataLoader`` for a split."""
        return DataLoader(
            self.loaders[split],
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=sampler is None and split == "train",
            num_workers=self.config.num_workers,
            collate_fn=unified_collate,
            worker_init_fn=seed_worker,
            drop_last=split == "train",
        )

    def train_dataloader(self) -> DataLoader | None:
        """Return the training ``DataLoader`` (or ``None`` if no train split)."""
        if "train" not in self.loaders:
            return None
        return self._dataloader("train", self.samplers.get("train"))

    def val_dataloader(self) -> DataLoader | None:
        """Return the validation ``DataLoader`` (or ``None``)."""
        if "valid" not in self.loaders:
            return None
        return self._dataloader("valid", None)

    def test_dataloader(self) -> DataLoader | None:
        """Return the test ``DataLoader`` (or ``None``)."""
        if "test" not in self.loaders:
            return None
        return self._dataloader("test", None)
