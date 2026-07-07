"""The OmniLoader meta-dataset that unifies several disjoint datasets.

:class:`OmniLoader` concatenates any number of source datasets, each paired with
a :class:`~omniloader.schema.spec.DatasetSchema`, and presents them as one dataset whose
every sample follows the shared :class:`~omniloader.schema.spec.UnifiedSchema`. Keys a
given source lacks are filled with placeholder tensors and all-``False`` masks by
the :class:`~omniloader.schema.unify.SampleUnifier`, so a single batch may freely mix
samples from different datasets. A :class:`~omniloader.sampling.strategies.MixingStrategy`
(obtained via :meth:`OmniLoader.make_sampler`) controls how the datasets are
balanced across an epoch.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from torch.utils.data import Dataset

from omniloader.sampling.sampler import OmniSampler
from omniloader.sampling.strategies import (
    MixingStrategy,
    ProportionalStrategy,
    SubsampleConfig,
)
from omniloader.schema.spec import DatasetSchema, UnifiedSchema
from omniloader.schema.unify import SampleUnifier

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omniloader.introspection import Report
    from omniloader.transforms import Transform


@runtime_checkable
class SizedDataset(Protocol):
    """A map-style dataset that reports its length and returns dict samples."""

    def __len__(self) -> int:
        """Return the number of samples."""
        ...

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the sample at ``index``."""
        ...


class OmniLoader(Dataset):
    """Unify multiple datasets into a single masked, concatenated dataset.

    Args:
        datasets: The source datasets. Each must implement ``__len__`` and
            return a ``dict`` from ``__getitem__``.
        schemas: One :class:`DatasetSchema` per dataset, declaring what it
            provides. Merged into a :class:`UnifiedSchema` when ``schema`` is
            not given.
        schema: Optional pre-built unified schema. When ``None``, it is the
            union of ``schemas``.
        pad_features: Whether the unifier pads/crops sequence values to their
            declared ``time_dim``. See :class:`~omniloader.schema.unify.SampleUnifier`.
        transform: Optional per-sample transform (normalization/augmentation)
            applied after unification. See :mod:`omniloader.transforms`.
        training: Whether augmentations in ``transform`` are active. Set
            ``False`` for validation/test loaders.
        seed: Base seed for per-sample augmentation randomness. Combined with the
            current epoch (see :meth:`set_epoch`) and the sample index so that
            augmentation is reproducible and independent of worker count.

    Raises:
        ValueError: If ``datasets`` is empty or its length differs from
            ``schemas``.

    """

    def __init__(
        self,
        datasets: Sequence[SizedDataset],
        schemas: Sequence[DatasetSchema],
        schema: UnifiedSchema | None = None,
        pad_features: bool = True,
        transform: Transform | None = None,
        training: bool = True,
        seed: int = 0,
    ) -> None:
        if not datasets:
            raise ValueError("OmniLoader requires at least one dataset")
        if len(datasets) != len(schemas):
            raise ValueError("datasets and schemas must have the same length")
        self.datasets = list(datasets)
        self.schemas = list(schemas)
        self.schema = schema if schema is not None else UnifiedSchema(self.schemas)
        self.unifier = SampleUnifier(self.schema, pad_features=pad_features)
        self.transform = transform
        self.training = training
        self.seed = seed
        self.epoch = 0
        self.dataset_sizes = [len(d) for d in self.datasets]
        self._cumulative = _cumulative_sizes(self.dataset_sizes)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed per-sample augmentation.

        Call once per epoch (the :class:`~omniloader.integrations.lightning.OmniDataModule`
        does this automatically) so augmentation differs across epochs yet stays
        reproducible.

        Args:
            epoch: The upcoming epoch number.

        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Total number of samples across all source datasets."""
        return self._cumulative[-1] if self._cumulative else 0

    def locate(self, index: int) -> tuple[int, int]:
        """Map a global index to ``(dataset_index, local_index)``.

        Args:
            index: Global index in ``[0, len(self))``. Negative indices count
                from the end.

        Returns:
            The owning dataset's position and the local index within it.

        Raises:
            IndexError: If ``index`` is out of range.

        """
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(f"Index {index} out of range for {length} samples")
        dataset_index = bisect.bisect_right(self._cumulative, index)
        start = self._cumulative[dataset_index - 1] if dataset_index > 0 else 0
        return dataset_index, index - start

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the unified sample at a global index.

        Args:
            index: Global index into the concatenated dataset.

        Returns:
            The sample mapped onto the unified schema with masks and metadata.

        """
        dataset_index, local_index = self.locate(index)
        raw = self.datasets[dataset_index][local_index]
        sample = self.unifier(raw)  # raw dict -> full union schema + masks
        if self.transform is not None:
            # Seed per sample from (seed, epoch, index) so augmentation is
            # reproducible and independent of the number of DataLoader workers.
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch * 2_147_483_647 + index)
            sample = self.transform(sample, training=self.training, generator=generator)
        return sample

    def describe(self, max_samples: int = 64) -> Report:
        """Summarise this loader's datasets (coverage, valid fractions, classes).

        Args:
            max_samples: Samples read per dataset for the statistics.

        Returns:
            A :class:`~omniloader.introspection.Report`.

        """
        from omniloader.introspection import describe

        return describe(self.datasets, self.schemas, max_samples=max_samples)

    def validate(self, num_samples: int = 4, strict: bool = False) -> list[str]:
        """Check each dataset's raw samples against its declared specs.

        Args:
            num_samples: Samples probed per dataset.
            strict: Raise :class:`ValueError` if any mismatch is found.

        Returns:
            A list of issue strings (empty when everything matches).

        """
        from omniloader.introspection import validate

        return validate(self.datasets, self.schemas, num_samples=num_samples, strict=strict)

    def sequence_lengths(self, key: str) -> list[int]:
        """Return the native sequence length of ``key`` for every sample.

        Reads each source sample's raw value (bypassing unification/transforms)
        and reports the length of its sequence axis, or ``0`` when the sample
        lacks the key. Intended to feed
        :class:`~omniloader.sampling.bucketing.LengthBucketBatchSampler`.

        Note:
            This iterates the entire dataset once — ``O(N)`` reads. For HDF5 back
            ends prefer computing it once and caching the result.

        Args:
            key: A sequence feature or target name.

        Returns:
            A list of per-sample lengths, indexed by global sample index.

        """
        lengths: list[int] = []
        for dataset_index, local_index in (self.locate(i) for i in range(len(self))):
            raw = self.datasets[dataset_index][local_index]
            value = raw.get(key)
            if value is not None and hasattr(value, "shape") and value.ndim >= 1:
                lengths.append(int(value.shape[0]))
            else:
                lengths.append(0)
        return lengths

    def make_sampler(
        self,
        strategy: MixingStrategy | None = None,
        seed: int = 0,
        subsample: Sequence[SubsampleConfig | None] | None = None,
        epoch: int = 0,
    ) -> OmniSampler:
        """Build an :class:`OmniSampler` for driving a ``DataLoader``.

        Args:
            strategy: A mixing strategy. When ``None``, a
                :class:`~omniloader.sampling.strategies.ProportionalStrategy` over the
                dataset sizes is used.
            seed: Base seed for the default strategy (ignored if ``strategy``
                is given).
            subsample: Per-dataset subsample configs for the default strategy.
            epoch: Initial epoch for the sampler.

        Returns:
            A sampler yielding global indices per epoch.

        """
        if strategy is None:
            strategy = ProportionalStrategy(self.dataset_sizes, seed=seed, subsample=subsample)
        return OmniSampler(strategy, epoch=epoch)


def _cumulative_sizes(sizes: Sequence[int]) -> list[int]:
    """Return cumulative sums used to map global indices to datasets."""
    cumulative = []
    running = 0
    for size in sizes:
        running += size
        cumulative.append(running)
    return cumulative
