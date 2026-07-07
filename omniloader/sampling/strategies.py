"""Dataset-mixing strategies that decide the per-epoch sampling scheme.

Each strategy answers one question: *given several datasets of possibly very
different sizes, how many samples from each should the network see per epoch, and
in what order?* Strategies return a flat list of **global** indices (into the
concatenated OmniLoader dataset) via :meth:`MixingStrategy.epoch_indices`, so a
plain :class:`~omniloader.sampling.sampler.OmniSampler` can drive an ordinary
``DataLoader``.

Available strategies:

* :class:`ProportionalStrategy` — sample in proportion to true dataset sizes.
* :class:`TemperatureStrategy` — re-weight by ``size ** (1 / temperature)``
  (``temperature=2`` recovers square-root scaling).
* :class:`FixedWeightStrategy` — explicit per-dataset weights, or uniform.
* :class:`RoundRobinStrategy` — a fixed effective size per dataset, interleaved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from omniloader.sampling.subsamplers import ExhaustionPolicy, IndexPool

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class SubsampleConfig:
    """Per-dataset sampling options.

    Args:
        replacement: Whether indices may repeat within an epoch.
        policy: Exhaustion policy for without-replacement draws.
        effective_size: Override for how many samples this dataset contributes
            per epoch. When ``None``, the strategy decides.
        sample_weights: Optional per-sample weights (length = dataset size) for
            skewed within-dataset sampling, e.g. class balancing.

    """

    replacement: bool = False
    policy: ExhaustionPolicy = ExhaustionPolicy.FRESH
    effective_size: int | None = None
    sample_weights: Sequence[float] | None = None


class MixingStrategy(ABC):
    """Base class turning per-dataset sizes into per-epoch global index lists.

    Args:
        sizes: Number of samples in each dataset, in OmniLoader order.
        seed: Base seed for reproducible draws and shuffles.
        subsample: Optional per-dataset :class:`SubsampleConfig`. Missing
            entries fall back to a default config.

    Raises:
        ValueError: If ``sizes`` is empty or contains a non-positive entry.

    """

    def __init__(
        self,
        sizes: Sequence[int],
        seed: int = 0,
        subsample: Sequence[SubsampleConfig | None] | None = None,
    ) -> None:
        if not sizes:
            raise ValueError("At least one dataset is required")
        if any(s <= 0 for s in sizes):
            raise ValueError(f"Dataset sizes must be positive, got {list(sizes)}")
        self.sizes = list(sizes)
        self.seed = seed
        self.offsets = _prefix_offsets(self.sizes)
        configs = list(subsample) if subsample is not None else [None] * len(sizes)
        if len(configs) != len(sizes):
            raise ValueError("subsample must have one entry per dataset")
        self.configs = [c or SubsampleConfig() for c in configs]
        self.pools = [
            IndexPool(size, cfg.replacement, cfg.policy, seed + i, cfg.sample_weights)
            for i, (size, cfg) in enumerate(zip(self.sizes, self.configs))
        ]

    @abstractmethod
    def target_counts(self, epoch: int) -> list[int]:
        """Number of samples each dataset contributes in ``epoch``.

        Args:
            epoch: The epoch number (used only by epoch-dependent strategies
                such as :class:`AnnealedTemperatureStrategy`).

        Returns:
            A list with one non-negative count per dataset.

        """

    def _effective_counts(self, epoch: int) -> list[int]:
        """Apply per-dataset ``effective_size`` overrides on top of the strategy."""
        counts = self.target_counts(epoch)
        return [
            cfg.effective_size if cfg.effective_size is not None else count
            for cfg, count in zip(self.configs, counts)
        ]

    def epoch_indices(self, epoch: int) -> list[int]:
        """Build the shuffled list of global indices to visit this epoch.

        Args:
            epoch: The epoch number, used to seed draws and the final shuffle.

        Returns:
            A flat list of global indices into the concatenated dataset.

        """
        indices: list[int] = []
        for pool, offset, count in zip(self.pools, self.offsets, self._effective_counts(epoch)):
            indices.extend(offset + i for i in pool.draw(count, epoch))
        return _shuffle(indices, self.seed + epoch)

    def __len__(self) -> int:
        """Total number of samples produced per epoch (evaluated at epoch 0)."""
        return sum(self._effective_counts(0))


class ProportionalStrategy(MixingStrategy):
    """Sample each dataset in proportion to its true size (full concatenation)."""

    def target_counts(self, epoch: int) -> list[int]:  # noqa: ARG002 (epoch-independent)
        """Return each dataset's full size."""
        return list(self.sizes)


class TemperatureStrategy(MixingStrategy):
    """Re-weight datasets by ``size ** (1 / temperature)``.

    ``temperature=1`` reduces to proportional sampling; ``temperature=2`` is
    square-root scaling; larger values push the mixture toward uniform. The
    total epoch size is preserved (equal to the sum of dataset sizes) unless
    per-dataset ``effective_size`` overrides are set.

    Args:
        sizes: Per-dataset sizes.
        temperature: Sampling temperature ``>= 1``.
        seed: Base seed.
        subsample: Optional per-dataset configs.

    Raises:
        ValueError: If ``temperature`` is less than 1.

    """

    def __init__(
        self,
        sizes: Sequence[int],
        temperature: float = 2.0,
        seed: int = 0,
        subsample: Sequence[SubsampleConfig | None] | None = None,
    ) -> None:
        if temperature < 1:
            raise ValueError(f"temperature must be >= 1, got {temperature}")
        self.temperature = temperature
        super().__init__(sizes, seed, subsample)

    def target_counts(self, epoch: int) -> list[int]:  # noqa: ARG002 (epoch-independent)
        """Return per-dataset counts weighted by ``size ** (1 / temperature)``."""
        return _temperature_counts(self.sizes, self.temperature)


class FixedWeightStrategy(MixingStrategy):
    """Sample datasets according to explicit weights (or uniformly).

    Args:
        sizes: Per-dataset sizes.
        weights: Relative weight per dataset. When ``None``, all datasets are
            weighted equally (uniform mixing). Need not sum to one.
        epoch_size: Total samples per epoch. Defaults to the sum of sizes.
        seed: Base seed.
        subsample: Optional per-dataset configs.

    Raises:
        ValueError: If ``weights`` has the wrong length or is not positive.

    """

    def __init__(
        self,
        sizes: Sequence[int],
        weights: Sequence[float] | None = None,
        epoch_size: int | None = None,
        seed: int = 0,
        subsample: Sequence[SubsampleConfig | None] | None = None,
    ) -> None:
        if weights is None:
            weights = [1.0] * len(sizes)
        if len(weights) != len(sizes):
            raise ValueError("weights must have one entry per dataset")
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("weights must be non-negative with a positive sum")
        self.weights = list(weights)
        self.epoch_size = epoch_size if epoch_size is not None else sum(sizes)
        super().__init__(sizes, seed, subsample)

    def target_counts(self, epoch: int) -> list[int]:  # noqa: ARG002 (epoch-independent)
        """Return per-dataset counts apportioned by the fixed weights."""
        return _apportion(self.weights, total=self.epoch_size)


class RoundRobinStrategy(MixingStrategy):
    """Draw a fixed number of samples per dataset and interleave them.

    Unlike the weighted strategies, this one takes an explicit effective size
    per dataset (default: the smallest dataset's size, so every dataset
    contributes equally) and interleaves the datasets round-robin rather than
    globally shuffling, which keeps batches diverse across datasets.

    Args:
        sizes: Per-dataset sizes.
        samples_per_dataset: Number of samples to draw from each dataset each
            epoch. When ``None``, uses the size of the smallest dataset.
        seed: Base seed.
        subsample: Optional per-dataset configs (``effective_size`` overrides
            ``samples_per_dataset`` for that dataset).

    """

    def __init__(
        self,
        sizes: Sequence[int],
        samples_per_dataset: int | None = None,
        seed: int = 0,
        subsample: Sequence[SubsampleConfig | None] | None = None,
    ) -> None:
        self.samples_per_dataset = samples_per_dataset
        super().__init__(sizes, seed, subsample)

    def target_counts(self, epoch: int) -> list[int]:  # noqa: ARG002 (epoch-independent)
        """Return an equal count per dataset (smallest size unless overridden)."""
        count = (
            self.samples_per_dataset if self.samples_per_dataset is not None else min(self.sizes)
        )
        return [count] * len(self.sizes)

    def epoch_indices(self, epoch: int) -> list[int]:
        """Interleave datasets round-robin instead of globally shuffling.

        Args:
            epoch: The epoch number.

        Returns:
            Global indices ordered so consecutive samples come from different
            datasets where possible.

        """
        per_dataset = [
            [offset + i for i in pool.draw(count, epoch)]
            for pool, offset, count in zip(self.pools, self.offsets, self._effective_counts(epoch))
        ]
        return _interleave(per_dataset)


class AnnealedTemperatureStrategy(MixingStrategy):
    """Temperature sampling whose temperature anneals across epochs.

    Multilingual/multi-task training often starts near-uniform (high temperature,
    up-weighting small datasets) and anneals toward proportional (temperature 1)
    so later epochs match the true data distribution. The temperature is linearly
    interpolated from ``start_temperature`` at epoch 0 to ``end_temperature`` at
    epoch ``num_epochs - 1`` (and held there afterwards).

    Args:
        sizes: Per-dataset sizes.
        start_temperature: Temperature at epoch 0 (``>= 1``).
        end_temperature: Temperature at the final epoch (``>= 1``).
        num_epochs: Number of epochs over which to anneal.
        seed: Base seed.
        subsample: Optional per-dataset configs.

    Raises:
        ValueError: If a temperature is below 1 or ``num_epochs`` is not positive.

    """

    def __init__(
        self,
        sizes: Sequence[int],
        start_temperature: float = 5.0,
        end_temperature: float = 1.0,
        num_epochs: int = 10,
        seed: int = 0,
        subsample: Sequence[SubsampleConfig | None] | None = None,
    ) -> None:
        if start_temperature < 1 or end_temperature < 1:
            raise ValueError("temperatures must be >= 1")
        if num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}")
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.num_epochs = num_epochs
        super().__init__(sizes, seed, subsample)

    def temperature_at(self, epoch: int) -> float:
        """Return the interpolated temperature for ``epoch``."""
        if self.num_epochs <= 1:
            return self.end_temperature
        frac = min(epoch, self.num_epochs - 1) / (self.num_epochs - 1)  # 0 -> 1
        return self.start_temperature + frac * (self.end_temperature - self.start_temperature)

    def target_counts(self, epoch: int) -> list[int]:
        """Return counts using the epoch's annealed temperature."""
        return _temperature_counts(self.sizes, self.temperature_at(epoch))


def _temperature_counts(sizes: Sequence[int], temperature: float) -> list[int]:
    """Apportion ``sum(sizes)`` samples by ``size ** (1 / temperature)`` weights."""
    weights = [s ** (1.0 / temperature) for s in sizes]
    return _apportion(weights, total=sum(sizes))


def _prefix_offsets(sizes: Sequence[int]) -> list[int]:
    """Return the starting global offset of each dataset in the concatenation."""
    offsets = []
    running = 0
    for size in sizes:
        offsets.append(running)
        running += size
    return offsets


def _apportion(weights: Sequence[float], total: int) -> list[int]:
    """Distribute ``total`` items across buckets by weight using largest remainder.

    Args:
        weights: Non-negative relative weights.
        total: Total number of items to distribute.

    Returns:
        Integer counts summing exactly to ``total``.

    """
    weight_sum = sum(weights)
    exact = [w / weight_sum * total for w in weights]
    floors = [int(x) for x in exact]
    remainder = total - sum(floors)
    # Hand out the leftover to the largest fractional parts.
    order = sorted(range(len(weights)), key=lambda i: exact[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors


def _shuffle(indices: list[int], seed: int) -> list[int]:
    """Deterministically shuffle a list of indices.

    Args:
        indices: Indices to shuffle.
        seed: Seed for the permutation.

    Returns:
        A new shuffled list.

    """
    if not indices:
        return []
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(len(indices), generator=gen).tolist()
    return [indices[i] for i in perm]


def _interleave(groups: Sequence[list[int]]) -> list[int]:
    """Interleave several index lists round-robin, dropping exhausted groups.

    Args:
        groups: One index list per dataset.

    Returns:
        A single interleaved list preserving each group's internal order.

    """
    out: list[int] = []
    cursors = [0] * len(groups)
    remaining = sum(len(g) for g in groups)
    while remaining:
        for gi, group in enumerate(groups):
            if cursors[gi] < len(group):
                out.append(group[cursors[gi]])
                cursors[gi] += 1
                remaining -= 1
    return out
