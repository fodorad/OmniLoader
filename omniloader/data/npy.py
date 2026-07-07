"""A dataset over a per-sample ``.npy`` folder layout.

:class:`NpyFolderDataset` reads the common feature-extraction layout where each
sample is a directory of ``.npy`` files, one per named value::

    root/
      <sample_key>/
        video.npy
        audio.npy
        valence.npy
        ...

Arrays are memory-mapped (``mmap_mode="r"``) so large features are not copied into
RAM until sliced. Every sample directory is expected to contain the same set of
files; the first sample defines the key set unless ``keys`` is given.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence


class NpyFolderDataset(Dataset):
    """Read samples from a directory of per-sample ``.npy`` sub-folders.

    Args:
        root: Directory whose immediate sub-directories are samples.
        keys: Optional value names (``.npy`` stems) to load. When ``None``, the
            files present in the first sample directory define the key set.
        mmap: Memory-map the arrays instead of reading them fully. Defaults to
            ``True``.

    Raises:
        FileNotFoundError: If ``root`` has no sample sub-directories.

    """

    def __init__(
        self,
        root: str | Path,
        keys: Sequence[str] | None = None,
        mmap: bool = True,
    ) -> None:
        self.root = Path(root)
        self.mmap = mmap
        self.sample_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not self.sample_dirs:
            raise FileNotFoundError(f"No sample sub-directories found under {self.root}")
        self.keys = (
            list(keys)
            if keys is not None
            else sorted(p.stem for p in self.sample_dirs[0].glob("*.npy"))
        )

    def __len__(self) -> int:
        """Number of sample directories."""
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load the sample at ``index`` as a dict of tensors plus its key.

        Args:
            index: Sample position.

        Returns:
            A dict mapping each key to a tensor, plus a ``key`` metadata string
            (the sample directory name).

        """
        sample_dir = self.sample_dirs[index]
        sample: dict[str, Any] = {"key": sample_dir.name}
        for name in self.keys:
            path = sample_dir / f"{name}.npy"
            if not path.exists():
                continue
            array = np.load(path, mmap_mode="r" if self.mmap else None)
            # Copy out of the (read-only) memmap so the tensor owns writable memory.
            sample[name] = torch.from_numpy(np.array(array))
        return sample
