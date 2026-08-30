"""Worker seeding helpers for reproducible data loading.

:func:`seed_worker` is a ``DataLoader(worker_init_fn=...)`` callback that gives
each worker process a deterministic, distinct RNG state derived from PyTorch's
per-worker seed, so any randomness that falls back to the global NumPy/Python/torch
RNGs (rather than an explicit :class:`torch.Generator`) is reproducible. It also
resets any HDF5-backed dataset reachable from the worker's dataset, since HDF5
handles are not fork-safe (see :meth:`omniloader.data.datasets.HDF5Dataset.reset_for_worker`).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def _resettable_datasets(dataset: Any, seen: set[int] | None = None) -> list[Any]:
    """Return every dataset reachable from ``dataset`` that defines ``reset_for_worker``.

    Walks ``OmniLoader``-style composition (a ``datasets`` sequence of source
    datasets) as well as plain ``torch.utils.data.Subset``/``ConcatDataset``
    (``dataset``/``datasets`` attributes), so a worker can reset every leaf
    dataset regardless of how many layers of wrapping sit above it.

    Args:
        dataset: The dataset instance handed to the worker (``get_worker_info().dataset``).
        seen: Internal set of already-visited object ids, to guard against cycles.

    Returns:
        The list of leaf datasets that expose a ``reset_for_worker`` method.

    """
    seen = set() if seen is None else seen
    if id(dataset) in seen:
        return []
    seen.add(id(dataset))

    found = [dataset] if hasattr(dataset, "reset_for_worker") else []
    for name in ("datasets", "dataset"):
        child = getattr(dataset, name, None)
        if child is None:
            continue
        children = child if isinstance(child, (list, tuple)) else [child]
        for entry in children:
            found.extend(_resettable_datasets(entry, seen))
    return found


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 (DataLoader worker_init_fn API)
    """Seed a DataLoader worker's RNGs and give it its own HDF5 handles.

    Pass as ``DataLoader(worker_init_fn=seed_worker)``. PyTorch already sets a
    distinct ``torch.initial_seed()`` per worker (derived from the base seed and
    worker id); this propagates it to the other RNGs.

    A ``fork``-started worker (the default on Linux/macOS) also inherits
    whatever HDF5 file handles were already open in the parent process, and
    those handles are not fork-safe — reading them from more than one process
    hangs rather than raising. This clears any such handle so the lazy reopen
    happens inside the worker, as intended by
    :meth:`omniloader.data.datasets.HDF5Dataset._handle`.

    Args:
        worker_id: The worker index supplied by the DataLoader (unused; the seed
            is taken from ``torch.initial_seed`` which already encodes it).

    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    for dataset in _resettable_datasets(info.dataset):
        dataset.reset_for_worker()
