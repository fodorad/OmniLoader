"""Configuration and reproducibility helpers.

:class:`OmniConfig` bundles the knobs a training run needs — seed, batch size,
worker count, the mixing strategy and the normalization/augmentation pipeline —
and loads from a dict, a JSON file or a YAML file so experiments stay
declarative. :func:`seed_everything` seeds Python, NumPy and PyTorch in one call.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

from omniloader.sampling.strategies import (
    AnnealedTemperatureStrategy,
    FixedWeightStrategy,
    MixingStrategy,
    ProportionalStrategy,
    RoundRobinStrategy,
    SubsampleConfig,
    TemperatureStrategy,
)
from omniloader.sampling.subsamplers import ExhaustionPolicy
from omniloader.transforms import build_transform

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.utils.data import DataLoader

    from omniloader.loader import OmniLoader, SizedDataset
    from omniloader.schema.spec import DatasetSchema, UnifiedSchema
    from omniloader.transforms import Compose

#: Registry mapping strategy names to their classes.
STRATEGIES: dict[str, type[MixingStrategy]] = {
    "proportional": ProportionalStrategy,
    "temperature": TemperatureStrategy,
    "annealed_temperature": AnnealedTemperatureStrategy,
    "fixed": FixedWeightStrategy,
    "round_robin": RoundRobinStrategy,
}

#: Recognised ``collate`` names for :meth:`OmniConfig.build_collate`.
COLLATES: tuple[str, ...] = ("unified", "dynamic", "mixup")


def _subsample_from_dict(entry: dict[str, Any]) -> SubsampleConfig:
    """Build a :class:`SubsampleConfig` from a config dict.

    Args:
        entry: Mapping with any of ``replacement``, ``policy`` (``"fresh"`` or
            ``"exhaust"``), ``effective_size`` and ``sample_weights``.

    Returns:
        The populated :class:`SubsampleConfig`.

    """
    kwargs: dict[str, Any] = dict(entry)
    if "policy" in kwargs and not isinstance(kwargs["policy"], ExhaustionPolicy):
        kwargs["policy"] = ExhaustionPolicy(kwargs["policy"])
    return SubsampleConfig(**kwargs)


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility.

    Args:
        seed: The seed value.

    Returns:
        The seed, for convenience.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@dataclass
class OmniConfig:
    """Declarative configuration for an OmniLoader training run.

    Args:
        seed: Global RNG seed.
        batch_size: Batch size for the ``DataLoader``.
        num_workers: Number of worker processes for the ``DataLoader``.
        pad_features: Whether the unifier pads/crops sequence values.
        strategy: Name of the mixing strategy (a key of :data:`STRATEGIES`).
        strategy_kwargs: Extra keyword arguments forwarded to the strategy
            (e.g. ``temperature``, ``weights``, ``samples_per_dataset``).
        transforms: Ordered list of transform config dicts (each with a ``name``
            key) applied during training. Stats-based normalizers may reference a
            saved stats file via ``stats_path`` (or ``union_stats_path``) instead of
            inlining ``stats``. See :func:`~omniloader.transforms.build_transform`.
        eval_transforms: Transforms applied during evaluation. When ``None``,
            only the non-augmenting (train-agnostic) transforms in
            :attr:`transforms`, such as ``normalize``, take effect.
        datasets: Declarative dataset entries (adapter + args + schema) built by
            :meth:`build_datasets`. See :mod:`omniloader.data.factory`.
        subsample: Optional per-dataset subsampling entries (one per dataset, or
            ``None`` to skip a dataset), each a dict for :class:`SubsampleConfig`.
            Built by :meth:`build_subsample` and fed to :meth:`build_strategy`.
        collate: Collate function name — one of :data:`COLLATES` (``"unified"``,
            ``"dynamic"`` or ``"mixup"``). See :meth:`build_collate`. ``"dynamic"``
            only reduces padding with :attr:`pad_features` set to ``False`` (under a
            fixed ``time_dim`` it behaves like ``"unified"``).
        collate_kwargs: Extra keyword arguments for the collator. For ``"dynamic"``
            an optional ``keys`` list; for ``"mixup"`` a ``base`` collate name plus
            ``alpha``/``mode``/``p``.
        bucketing: Optional length-bucketing entry with a ``key`` (sequence name to
            bucket by) and optional ``batch_size``/``bucket_multiplier``/
            ``drop_last``/``shuffle``. When set, :meth:`build_dataloader` batches
            with :class:`~omniloader.sampling.bucketing.LengthBucketBatchSampler`.
            Only meaningful with ``pad_features=False`` (a no-op under a fixed
            ``time_dim``).
        num_replicas: Distributed replica count for the sampler. ``None`` infers
            it from ``torch.distributed``.
        rank: This process's distributed rank. ``None`` infers it.
        drop_last: Drop the uneven tail when sharding across replicas.
        pin_memory: Copy tensors into pinned memory before returning them (faster
            host-to-GPU transfer). Forwarded to the ``DataLoader``.
        persistent_workers: Keep worker processes alive between epochs. Only takes
            effect when :attr:`num_workers` > 0.
        prefetch_factor: Batches prefetched per worker. Only applied when
            :attr:`num_workers` > 0; ``None`` uses the PyTorch default.

    """

    seed: int = 0
    batch_size: int = 32
    num_workers: int = 0
    pad_features: bool = True
    strategy: str = "proportional"
    strategy_kwargs: dict[str, Any] = field(default_factory=dict)
    transforms: list[dict[str, Any]] = field(default_factory=list)
    eval_transforms: list[dict[str, Any]] | None = None
    datasets: list[dict[str, Any]] = field(default_factory=list)
    subsample: list[dict[str, Any] | None] | None = None
    collate: str = "unified"
    collate_kwargs: dict[str, Any] = field(default_factory=dict)
    bucketing: dict[str, Any] | None = None
    num_replicas: int | None = None
    rank: int | None = None
    drop_last: bool = False
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OmniConfig:
        """Build a config from a plain dict, ignoring unknown keys.

        Args:
            data: Mapping of field names to values.

        Returns:
            The populated config.

        """
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in fields})

    @classmethod
    def from_yaml(cls, path: str | Path) -> OmniConfig:
        """Load a config from a YAML file."""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def from_json(cls, path: str | Path) -> OmniConfig:
        """Load a config from a JSON file."""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f) or {})

    @classmethod
    def from_file(cls, path: str | Path) -> OmniConfig:
        """Load a config from a ``.json``, ``.yaml`` or ``.yml`` file by extension.

        Args:
            path: Path to a config file.

        Returns:
            The populated config.

        Raises:
            ValueError: If the file extension is not recognised.

        """
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            return cls.from_json(path)
        if suffix in {".yaml", ".yml"}:
            return cls.from_yaml(path)
        raise ValueError(f"Unsupported config extension {suffix!r}; use .json/.yaml/.yml")

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain dict."""
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pad_features": self.pad_features,
            "strategy": self.strategy,
            "strategy_kwargs": dict(self.strategy_kwargs),
            "transforms": [dict(t) for t in self.transforms],
            "eval_transforms": (
                None if self.eval_transforms is None else [dict(t) for t in self.eval_transforms]
            ),
            "datasets": [dict(d) for d in self.datasets],
            "subsample": (
                None
                if self.subsample is None
                else [None if s is None else dict(s) for s in self.subsample]
            ),
            "collate": self.collate,
            "collate_kwargs": dict(self.collate_kwargs),
            "bucketing": None if self.bucketing is None else dict(self.bucketing),
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "drop_last": self.drop_last,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
        }

    def build_strategy(
        self,
        sizes: Sequence[int],
        subsample: Sequence[SubsampleConfig | None] | None = None,
    ) -> MixingStrategy:
        """Instantiate the configured mixing strategy for the given sizes.

        Args:
            sizes: Per-dataset sample counts.
            subsample: Optional per-dataset subsample configs.

        Returns:
            The configured :class:`~omniloader.sampling.strategies.MixingStrategy`.

        Raises:
            ValueError: If :attr:`strategy` is not a known strategy name.

        """
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy {self.strategy!r}; expected one of {sorted(STRATEGIES)}"
            )
        strategy_cls = STRATEGIES[self.strategy]
        return strategy_cls(sizes, seed=self.seed, subsample=subsample, **self.strategy_kwargs)

    def build_transform(self, schema: UnifiedSchema, training: bool = True) -> Compose | None:
        """Build the transform pipeline for a split.

        Args:
            schema: The unified schema, injected into schema-aware transforms.
            training: When ``True``, uses :attr:`transforms`; otherwise uses
                :attr:`eval_transforms` if set, else :attr:`transforms` (whose
                augmentations self-skip in eval mode).

        Returns:
            A :class:`~omniloader.transforms.Compose`, or ``None`` if empty.

        """
        configs = self.transforms
        if not training and self.eval_transforms is not None:
            configs = self.eval_transforms
        return build_transform(configs, schema)

    def build_datasets(self) -> tuple[list[SizedDataset], list[DatasetSchema]]:
        """Construct the datasets/schemas declared in :attr:`datasets`.

        Returns:
            A ``(datasets, schemas)`` tuple. See
            :func:`omniloader.data.factory.build_datasets`.

        """
        from omniloader.data.factory import build_datasets

        return build_datasets(self.datasets)

    def build_subsample(self) -> list[SubsampleConfig | None] | None:
        """Build the per-dataset :class:`SubsampleConfig` list from :attr:`subsample`.

        Returns:
            One :class:`SubsampleConfig` (or ``None``) per dataset, or ``None`` when
            :attr:`subsample` is unset.

        """
        if self.subsample is None:
            return None
        return [None if entry is None else _subsample_from_dict(entry) for entry in self.subsample]

    def build_collate(
        self, schema: UnifiedSchema, training: bool = True
    ) -> Callable[[Sequence[dict[str, Any]]], dict[str, Any]]:
        """Build the collate function named by :attr:`collate`.

        Args:
            schema: The unified schema, injected into schema-aware collators.
            training: When ``False``, the batch-level ``"mixup"`` augmentation is
                disabled and its ``base`` collator is used instead.

        Returns:
            A callable turning a list of samples into a batched dict.

        Raises:
            ValueError: If :attr:`collate` (or a ``mixup`` ``base``) is unknown.

        """
        from omniloader.collate import DynamicCollator, unified_collate

        if self.collate not in COLLATES:
            raise ValueError(f"Unknown collate {self.collate!r}; expected one of {list(COLLATES)}")

        def _base(name: str, kwargs: dict[str, Any]) -> Any:
            if name == "unified":
                return unified_collate
            if name == "dynamic":
                return DynamicCollator(schema, keys=kwargs.get("keys"))
            raise ValueError(f"Unknown collate {name!r}; expected 'unified' or 'dynamic'")

        if self.collate != "mixup":
            return _base(self.collate, self.collate_kwargs)

        from omniloader.transforms.mix import MixupCollator

        kwargs = dict(self.collate_kwargs)
        base = _base(kwargs.pop("base", "unified"), kwargs)
        kwargs.pop("keys", None)  # consumed by the base collator only
        if not training:
            return base  # mixing is a train-time augmentation
        kwargs.setdefault("seed", self.seed)
        return MixupCollator(base, schema, **kwargs)

    def build_loader(
        self,
        datasets: Sequence[SizedDataset] | None = None,
        schemas: Sequence[DatasetSchema] | None = None,
        training: bool = True,
    ) -> OmniLoader:
        """Build a fully configured :class:`~omniloader.loader.OmniLoader`.

        Args:
            datasets: Source datasets. When ``None``, they are built from
                :attr:`datasets` via :meth:`build_datasets`.
            schemas: Per-dataset schemas. Required when ``datasets`` is given;
                ignored (rebuilt) when ``datasets`` is ``None``.
            training: Whether augmentations are active (``False`` for val/test).

        Returns:
            The loader with its transform pipeline wired in.

        Raises:
            ValueError: If ``datasets`` is given without ``schemas``.

        """
        from omniloader.loader import OmniLoader

        if datasets is None:
            datasets, schemas = self.build_datasets()
        elif schemas is None:
            raise ValueError("schemas must be given when datasets is provided")
        loader = OmniLoader(
            datasets,
            schemas,
            pad_features=self.pad_features,
            training=training,
            seed=self.seed,
        )
        loader.transform = self.build_transform(loader.schema, training=training)
        return loader

    def build_dataloader(
        self,
        datasets: Sequence[SizedDataset] | None = None,
        schemas: Sequence[DatasetSchema] | None = None,
        training: bool = True,
        epoch: int = 0,
    ) -> DataLoader:
        """Assemble a ready-to-iterate ``DataLoader`` entirely from this config.

        Wires together the loader, mixing strategy, distributed sampler, collate
        function and (optionally) length bucketing. Training loaders draw indices
        through the configured :class:`~omniloader.sampling.strategies.MixingStrategy`;
        evaluation loaders (``training=False``) iterate sequentially for a stable,
        full-coverage pass.

        Args:
            datasets: Source datasets, or ``None`` to build them from
                :attr:`datasets`.
            schemas: Per-dataset schemas (required with ``datasets``).
            training: Whether this is the training split.
            epoch: Initial epoch for the sampler and per-sample augmentation.

        Returns:
            A configured :class:`torch.utils.data.DataLoader`.

        """
        from torch.utils.data import DataLoader, SequentialSampler

        from omniloader.sampling.sampler import OmniSampler
        from omniloader.utils.seeding import seed_worker

        loader = self.build_loader(datasets, schemas, training=training)
        loader.set_epoch(epoch)
        collate = self.build_collate(loader.schema, training=training)

        sampler: OmniSampler | None = None
        if training:
            strategy = self.build_strategy(loader.dataset_sizes, subsample=self.build_subsample())
            sampler = OmniSampler(
                strategy,
                epoch=epoch,
                num_replicas=self.num_replicas,
                rank=self.rank,
                drop_last=self.drop_last,
            )

        # persistent_workers/prefetch_factor are only valid with worker processes.
        kwargs: dict[str, Any] = {
            "num_workers": self.num_workers,
            "collate_fn": collate,
            "worker_init_fn": seed_worker,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor

        if self.bucketing is not None:
            from omniloader.sampling.bucketing import LengthBucketBatchSampler

            opts = dict(self.bucketing)
            key = opts.pop("key")
            batch_size = opts.pop("batch_size", self.batch_size)
            base_sampler = sampler if sampler is not None else SequentialSampler(loader)
            opts.setdefault("shuffle", training)
            batch_sampler = LengthBucketBatchSampler(
                base_sampler,
                loader.sequence_lengths(key),
                batch_size,
                seed=self.seed,
                **opts,
            )
            return DataLoader(loader, batch_sampler=batch_sampler, **kwargs)

        return DataLoader(loader, batch_size=self.batch_size, sampler=sampler, **kwargs)
