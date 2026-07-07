"""Dataset adapters that yield raw sample dicts for OmniLoader.

The concrete datasets here (an HDF5-backed reader and an in-memory tensor
dataset) implement the loose contract OmniLoader expects: ``__len__`` returns the
number of samples and ``__getitem__`` returns a ``dict`` of named values. The
:class:`~omniloader.schema.spec.DatasetSchema` describing what a dataset provides is
supplied alongside it to :class:`~omniloader.loader.OmniLoader`.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _decode_hdf5_value(dset: h5py.Dataset) -> Any:
    """Read a single HDF5 dataset into a tensor, string, or list of strings.

    Args:
        dset: An open HDF5 dataset object.

    Returns:
        A :class:`torch.Tensor` for numeric data, a ``str`` for scalar strings,
        or a ``list[str]`` for string arrays.

    """
    if dset.dtype.kind in {"S", "O"}:
        value = dset.asstr()[()]
        if isinstance(value, (np.ndarray, list)):
            return [str(v) for v in value]
        return str(value)

    value = dset[()]
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, (bool, np.bool_)):
        return torch.tensor(bool(value))
    if isinstance(value, (int, np.integer)):
        return torch.tensor(int(value))
    if isinstance(value, (float, np.floating)):
        return torch.tensor(float(value))
    return torch.as_tensor(value)


class HDF5Dataset(Dataset):
    """Read samples from a group of an HDF5 file, one sub-group per sample.

    The file is expected to hold a group named ``subset`` (e.g. ``"train"``)
    whose children are per-sample groups, each containing the named value
    datasets (features, targets, masks, metadata). The file handle is opened
    lazily per worker process so the dataset is safe to use with
    ``num_workers > 0``.

    Args:
        h5_path: Path to the ``.h5`` file.
        subset: Name of the top-level group to read (e.g. ``"train"``).
        keys: Optional subset of value names to load per sample. When ``None``,
            every dataset in the sample group is loaded.
        cache_size: If ``> 0``, keep an in-process LRU cache of up to this many
            decoded samples to speed up repeated reads.
        preload: If ``True``, decode every sample into memory up front (fastest
            for small datasets; uses the most memory).

    Raises:
        KeyError: If ``subset`` is not present in the file.

    """

    def __init__(
        self,
        h5_path: str | Path,
        subset: str,
        keys: Sequence[str] | None = None,
        cache_size: int = 0,
        preload: bool = False,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.subset = subset
        self.keys = list(keys) if keys is not None else None
        self.cache_size = cache_size
        self._file: h5py.File | None = None
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        with h5py.File(self.h5_path, "r") as f:
            if subset not in f:
                raise KeyError(f"Subset {subset!r} not found in {self.h5_path}")
            self.sample_ids = sorted(f[subset].keys())
        self._preloaded: list[dict[str, Any]] | None = (
            [self._read(i) for i in range(len(self.sample_ids))] if preload else None
        )

    def _handle(self) -> h5py.File:
        """Return a per-process open file handle, opening it on first use."""
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def _read(self, index: int) -> dict[str, Any]:
        """Decode the sample at ``index`` directly from the HDF5 file."""
        sample_id = self.sample_ids[index]
        group = self._handle()[self.subset][sample_id]
        names = self.keys if self.keys is not None else list(group.keys())
        return {name: _decode_hdf5_value(group[name]) for name in names if name in group}

    def __len__(self) -> int:
        """Number of samples in the subset."""
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load the sample at ``index`` as a dict of named values.

        Args:
            index: Position within the subset.

        Returns:
            A dict mapping each stored key to a tensor or string.

        """
        if self._preloaded is not None:
            return self._preloaded[index]
        if self.cache_size > 0 and index in self._cache:
            self._cache.move_to_end(index)  # mark as most-recently used
            return self._cache[index]
        sample = self._read(index)
        if self.cache_size > 0:
            self._cache[index] = sample
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)  # evict least-recently used
        return sample

    def __getstate__(self) -> dict[str, Any]:
        """Drop the open file handle and cache when pickling (for workers)."""
        state = self.__dict__.copy()
        state["_file"] = None
        state["_cache"] = OrderedDict()
        return state


class DictTensorDataset(Dataset):
    """In-memory dataset over a dict of stacked tensors, keyed by sample index.

    Useful for tests and small experiments: each entry maps a name to a tensor
    whose leading dimension is the sample axis. ``__getitem__`` returns the
    per-sample slice of every tensor as a dict.

    Args:
        tensors: Mapping of value name to a tensor of shape ``(N, ...)`` where
            ``N`` is the number of samples (equal across all tensors).
        metadata: Optional mapping of metadata name to a per-sample sequence
            (e.g. dataset name or sample key) of length ``N``.

    Raises:
        ValueError: If the tensors disagree on the number of samples.

    """

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        metadata: Mapping[str, Sequence[Any]] | None = None,
    ) -> None:
        if not tensors:
            raise ValueError("DictTensorDataset requires at least one tensor")
        lengths = {t.shape[0] for t in tensors.values()}
        if len(lengths) != 1:
            raise ValueError(f"All tensors must share the sample dimension, got {lengths}")
        self.tensors = dict(tensors)
        self.metadata = {k: list(v) for k, v in (metadata or {}).items()}
        self._length = lengths.pop()

    def __len__(self) -> int:
        """Number of samples."""
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the per-sample slice of every tensor plus metadata.

        Args:
            index: Sample position.

        Returns:
            A dict of the sample's values and metadata.

        """
        sample: dict[str, Any] = {name: t[index] for name, t in self.tensors.items()}
        for name, values in self.metadata.items():
            sample[name] = values[index]
        return sample
