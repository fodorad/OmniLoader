"""Worker seeding helpers for reproducible data loading.

:func:`seed_worker` is a ``DataLoader(worker_init_fn=...)`` callback that gives
each worker process a deterministic, distinct RNG state derived from PyTorch's
per-worker seed, so any randomness that falls back to the global NumPy/Python/torch
RNGs (rather than an explicit :class:`torch.Generator`) is reproducible.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 (DataLoader worker_init_fn API)
    """Seed a DataLoader worker's Python, NumPy and torch RNGs deterministically.

    Pass as ``DataLoader(worker_init_fn=seed_worker)``. PyTorch already sets a
    distinct ``torch.initial_seed()`` per worker (derived from the base seed and
    worker id); this propagates it to the other RNGs.

    Args:
        worker_id: The worker index supplied by the DataLoader (unused; the seed
            is taken from ``torch.initial_seed`` which already encodes it).

    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
