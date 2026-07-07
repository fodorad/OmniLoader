"""A ``Sampler`` that turns a mixing strategy into a per-epoch index stream.

:class:`OmniSampler` is the bridge between a
:class:`~omniloader.sampling.strategies.MixingStrategy` and PyTorch's ``DataLoader``.
Each epoch it asks the strategy for the global indices to visit; call
:meth:`OmniSampler.set_epoch` before every epoch to advance the reproducible draw.

It is **distributed-aware**: when ``num_replicas``/``rank`` are given (or inferred
from an initialized ``torch.distributed`` process group) it shards the epoch's
indices across replicas so each rank sees a disjoint slice, mirroring
``torch.utils.data.DistributedSampler``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.distributed as dist
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from collections.abc import Iterator

    from omniloader.sampling.strategies import MixingStrategy


def _resolve_dist(num_replicas: int | None, rank: int | None) -> tuple[int, int]:
    """Resolve ``(num_replicas, rank)``, inferring from ``torch.distributed``.

    Args:
        num_replicas: Explicit replica count, or ``None`` to infer.
        rank: Explicit rank, or ``None`` to infer.

    Returns:
        The resolved ``(num_replicas, rank)``; ``(1, 0)`` when not distributed.

    Raises:
        ValueError: If ``rank`` is outside ``[0, num_replicas)``.

    """
    if num_replicas is None:
        num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if rank is None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if not 0 <= rank < num_replicas:
        raise ValueError(f"rank {rank} out of range for {num_replicas} replicas")
    return num_replicas, rank


class OmniSampler(Sampler[int]):
    """Yield global indices for one epoch according to a mixing strategy.

    Args:
        strategy: The mixing strategy that computes the epoch's index list.
        epoch: Initial epoch number.
        num_replicas: Number of distributed replicas. ``None`` infers from
            ``torch.distributed`` (or ``1`` when not distributed).
        rank: This process's replica index. ``None`` infers from
            ``torch.distributed`` (or ``0``).
        drop_last: When sharding, drop the tail so every rank gets exactly the
            same number of samples. When ``False`` (default), the index list is
            padded by wrapping so no sample is skipped.

    """

    def __init__(
        self,
        strategy: MixingStrategy,
        epoch: int = 0,
        num_replicas: int | None = None,
        rank: int | None = None,
        drop_last: bool = False,
    ) -> None:
        self.strategy = strategy
        self.epoch = epoch
        self.num_replicas, self.rank = _resolve_dist(num_replicas, rank)
        self.drop_last = drop_last

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed the next iteration.

        Args:
            epoch: The upcoming epoch number.

        """
        self.epoch = epoch

    def _shard(self, indices: list[int]) -> list[int]:
        """Return this rank's slice of ``indices`` (identity when single-process)."""
        if self.num_replicas == 1:
            return indices
        if self.drop_last:
            usable = (len(indices) // self.num_replicas) * self.num_replicas
            indices = indices[:usable]
        elif len(indices) % self.num_replicas:
            # Pad by wrapping from the front so the split is even.
            pad = self.num_replicas - (len(indices) % self.num_replicas)
            indices = indices + indices[:pad]
        return indices[self.rank :: self.num_replicas]

    def __iter__(self) -> Iterator[int]:
        """Iterate this rank's global indices for the current epoch."""
        return iter(self._shard(self.strategy.epoch_indices(self.epoch)))

    def __len__(self) -> int:
        """Number of indices this rank produces per epoch."""
        total = len(self.strategy)
        if self.num_replicas == 1:
            return total
        if self.drop_last:
            return total // self.num_replicas
        return -(-total // self.num_replicas)  # ceil
