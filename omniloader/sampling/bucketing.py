"""Length-bucketed batching to minimize padding waste.

:class:`LengthBucketBatchSampler` wraps an index sampler (e.g.
:class:`~omniloader.sampling.sampler.OmniSampler`) and a per-sample length array,
and yields *batches* of indices whose sequences have similar length. Used as
``DataLoader(batch_sampler=...)`` together with
:class:`~omniloader.collate.DynamicCollator`, it keeps each batch's padded length
close to its true length.

The classic pooling scheme is used: the sampler's index stream is cut into pools
of ``batch_size * bucket_multiplier``; each pool is sorted by length and split
into batches; the batch order is then shuffled so length still varies across
steps (only within-batch length is homogenised).

This only reduces padding waste in the **native-length** regime
(``OmniLoader(pad_features=False)`` + :class:`~omniloader.collate.DynamicCollator`).
When every sample is padded/cropped to a fixed ``time_dim`` the batches are already
equal length, so grouping by length changes no shapes and bucketing is a no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


@runtime_checkable
class SizedSampler(Protocol):
    """An index sampler that reports its length (as PyTorch samplers do)."""

    def __iter__(self) -> Iterator[int]:
        """Iterate the sampler's indices."""
        ...

    def __len__(self) -> int:
        """Number of indices produced per epoch."""
        ...


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Yield batches of indices grouped by sequence length.

    Only meaningful in the native-length regime (``pad_features=False`` +
    :class:`~omniloader.collate.DynamicCollator`); under a fixed ``time_dim`` all
    batches are already equal length and this changes nothing.

    Args:
        sampler: An index sampler yielding global indices (its length defines the
            epoch size). If it exposes an ``epoch`` attribute, it seeds the batch
            shuffle for reproducibility across epochs.
        lengths: Per-index sequence lengths, indexable by the sampler's indices
            (e.g. from :meth:`~omniloader.loader.OmniLoader.sequence_lengths`).
        batch_size: Number of samples per batch.
        bucket_multiplier: Pool size in batches; larger pools bucket more tightly
            but reduce cross-length shuffling.
        drop_last: Drop a trailing batch smaller than ``batch_size``.
        shuffle: Shuffle the order of the produced batches each epoch.
        seed: Base seed for the batch-order shuffle.

    Raises:
        ValueError: If ``batch_size`` or ``bucket_multiplier`` is not positive.

    """

    def __init__(
        self,
        sampler: SizedSampler,
        lengths: Sequence[int],
        batch_size: int,
        bucket_multiplier: int = 100,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0 or bucket_multiplier <= 0:
            raise ValueError("batch_size and bucket_multiplier must be positive")
        self.sampler = sampler
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_multiplier = bucket_multiplier
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

    def _batches(self) -> list[list[int]]:
        """Build this epoch's batches: pool, sort by length, split."""
        indices = list(self.sampler)
        pool_size = self.batch_size * self.bucket_multiplier
        batches: list[list[int]] = []
        for start in range(0, len(indices), pool_size):
            pool = sorted(indices[start : start + pool_size], key=lambda i: self.lengths[i])
            for j in range(0, len(pool), self.batch_size):
                batch = pool[j : j + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate length-bucketed batches for the current epoch."""
        batches = self._batches()
        if self.shuffle:
            epoch = int(getattr(self.sampler, "epoch", 0))
            gen = torch.Generator()
            gen.manual_seed(self.seed + epoch)
            order = torch.randperm(len(batches), generator=gen).tolist()
            batches = [batches[i] for i in order]
        yield from batches

    def __len__(self) -> int:
        """Number of batches produced per epoch."""
        n = len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return -(-n // self.batch_size)  # ceil
