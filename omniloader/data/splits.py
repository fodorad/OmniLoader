"""Reproducible train/val/test index splitting.

:func:`split_indices` partitions ``range(n)`` into named splits by ratio, with an
optional stratification label so each split preserves the class proportions.
:func:`save_split_info` / :func:`load_split_info` persist the result (plus the
seed and ratios) so the exact split can be reused across runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Default split names, matching the HDF5/dataset subset convention.
DEFAULT_NAMES = ("train", "valid", "test")


def _partition(
    indices: list[int], ratios: Sequence[float], names: Sequence[str]
) -> dict[str, list[int]]:
    """Cut an ordered index list into named chunks by (normalized) ratio."""
    total = sum(ratios)
    splits: dict[str, list[int]] = {}
    start = 0
    for i, (name, ratio) in enumerate(zip(names, ratios)):
        # Give the final split the remainder so every index is assigned exactly once.
        end = len(indices) if i == len(names) - 1 else start + round(len(indices) * ratio / total)
        splits[name] = indices[start:end]
        start = end
    return splits


def split_indices(
    n: int,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    stratify: Sequence[int] | None = None,
    names: Sequence[str] = DEFAULT_NAMES,
) -> dict[str, list[int]]:
    """Partition ``range(n)`` into named, reproducible splits.

    Args:
        n: Number of samples to split.
        ratios: Relative sizes per split (need not sum to one).
        seed: Seed for the shuffle.
        stratify: Optional per-index class labels of length ``n``; when given,
            each class is split by ``ratios`` so proportions are preserved.
        names: Split names, aligned with ``ratios``.

    Returns:
        Mapping of split name to a sorted list of indices; the splits are
        disjoint and cover ``range(n)``.

    Raises:
        ValueError: If ``ratios``/``names`` mismatch or ``stratify`` has the
            wrong length.

    """
    if len(ratios) != len(names):
        raise ValueError("ratios and names must have the same length")
    gen = torch.Generator().manual_seed(seed)

    if stratify is None:
        shuffled = torch.randperm(n, generator=gen).tolist()
        splits = _partition(shuffled, ratios, names)
    else:
        if len(stratify) != n:
            raise ValueError(f"stratify must have length {n}, got {len(stratify)}")
        groups: dict[int, list[int]] = {}
        for idx, label in enumerate(stratify):
            groups.setdefault(int(label), []).append(idx)
        splits = {name: [] for name in names}
        for label in sorted(groups):
            members = [groups[label][i] for i in torch.randperm(len(groups[label]), generator=gen)]
            for name, part in _partition(members, ratios, names).items():
                splits[name].extend(part)

    return {name: sorted(idx) for name, idx in splits.items()}


def save_split_info(
    splits: dict[str, list[int]],
    path: str | Path,
    meta: dict[str, Any] | None = None,
) -> None:
    """Save splits (and optional metadata like seed/ratios) to JSON.

    Args:
        splits: The split mapping from :func:`split_indices`.
        path: Destination ``.json`` file.
        meta: Optional extra fields (e.g. ``{"seed": 0, "ratios": [...]}``).

    """
    payload = {
        "splits": splits,
        "counts": {name: len(idx) for name, idx in splits.items()},
        **(meta or {}),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_info(path: str | Path) -> dict[str, list[int]]:
    """Load the ``splits`` mapping saved by :func:`save_split_info`.

    Args:
        path: Path to a JSON file written by :func:`save_split_info`.

    Returns:
        The split-name to index-list mapping.

    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["splits"]
