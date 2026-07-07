"""Compute and persist feature normalization statistics over a dataset.

:func:`compute_feature_stats` estimates per-feature ``(mean, std)`` (streaming),
while :func:`compute_stats` returns the richer set (mean/std/min/max/median/iqr)
needed by every normalizer variant. :func:`compute_dataset_stats` computes that
richer set **separately per source dataset** (grouped by the ``dataset`` metadata
key) for dataset-conditional normalization. :func:`save_stats`/:func:`load_stats`
round-trip either the flat (``{key: {...}}``) or the nested
(``{dataset: {key: {...}}}``) form through JSON. All are modality-agnostic: they
accumulate over the trailing feature axis of both vectors and sequences, counting
only valid (unmasked) positions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def compute_feature_stats(
    samples: Iterable[dict[str, torch.Tensor]],
    keys: Sequence[str],
    eps: float = 1e-6,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Estimate per-feature ``(mean, std)`` for each key over valid positions.

    Args:
        samples: An iterable of unified sample dicts (e.g. an
            :class:`~omniloader.loader.OmniLoader`). Each yields values and
            their ``<name>_mask`` companions.
        keys: The feature keys to compute statistics for.
        eps: Lower bound applied to the standard deviation to avoid zeros.

    Returns:
        Mapping of key to ``(mean, std)`` tensors of shape ``(F,)`` for
        features with a trailing feature axis, or ``()`` for scalar sequences.

    Raises:
        ValueError: If a key never has any valid position across the samples.

    """
    # Running sums per key: count of valid rows, sum and sum-of-squares over them.
    count: dict[str, float] = {k: 0.0 for k in keys}
    total: dict[str, torch.Tensor] = {}
    total_sq: dict[str, torch.Tensor] = {}

    for sample in samples:
        for key in keys:
            if key not in sample:
                continue
            value = sample[key].to(torch.float32)  # (T, F) | (F,) | (T,) | ()
            mask = sample[f"{key}_mask"]
            rows, feat = _valid_rows(value, mask)  # rows: (N, F) | (N,), N valid positions
            if rows.numel() == 0:
                continue
            count[key] += rows.shape[0]
            summed = rows.sum(dim=0)  # (F,) or ()
            summed_sq = (rows * rows).sum(dim=0)  # (F,) or ()
            if key not in total:
                total[key] = torch.zeros(feat, dtype=torch.float32) if feat else torch.zeros(())
                total_sq[key] = torch.zeros_like(total[key])
            total[key] += summed
            total_sq[key] += summed_sq

    stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key in keys:
        if count[key] == 0:
            raise ValueError(f"Key {key!r} has no valid positions to compute statistics")
        n = count[key]
        mean = total[key] / n
        var = torch.clamp(total_sq[key] / n - mean * mean, min=0.0)
        std = torch.sqrt(var).clamp_min(eps)
        stats[key] = (mean, std)
    return stats


def compute_stats(
    samples: Iterable[dict[str, torch.Tensor]],
    keys: Sequence[str],
    eps: float = 1e-6,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute the full statistic set per key, feeding every normalizer variant.

    Unlike :func:`compute_feature_stats`, this collects all valid positions (to
    compute median/IQR), so it uses more memory; run it once and persist with
    :func:`save_stats`.

    Args:
        samples: An iterable of unified sample dicts.
        keys: The feature keys to compute statistics for.
        eps: Lower bound for ``std`` and ``iqr`` to avoid zeros.

    Returns:
        Mapping of key to ``{"mean", "std", "min", "max", "median", "iqr"}``
        tensors of shape ``(F,)`` (or ``()`` for scalar sequences).

    Raises:
        ValueError: If a key never has any valid position across the samples.

    """
    collected: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    for sample in samples:
        for key in keys:
            if key not in sample:
                continue
            rows, _ = _valid_rows(sample[key].to(torch.float32), sample[f"{key}_mask"])
            if rows.numel():
                collected[key].append(rows)

    stats: dict[str, dict[str, torch.Tensor]] = {}
    for key in keys:
        if not collected[key]:
            raise ValueError(f"Key {key!r} has no valid positions to compute statistics")
        stats[key] = _reduce_stats(torch.cat(collected[key], dim=0), eps)
    return stats


def compute_dataset_stats(
    samples: Iterable[dict[str, Any]],
    keys: Sequence[str],
    dataset_key: str = "dataset",
    eps: float = 1e-6,
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Compute the full statistic set per key, **grouped by source dataset**.

    Feeds :class:`~omniloader.transforms.normalize.PerDatasetNormalize`: each source
    dataset (identified by the ``dataset`` metadata key every unified sample carries)
    gets its own statistics, so datasets can be standardized by their own mean/std
    rather than a pooled estimate. As with :func:`compute_stats`, only valid
    (unmasked) positions contribute. Keys a dataset never provides are simply absent
    from that dataset's stats.

    Args:
        samples: An iterable of unified sample dicts (e.g. an
            :class:`~omniloader.loader.OmniLoader`). Each must carry ``dataset_key``.
        keys: The feature keys to compute statistics for.
        dataset_key: Metadata key holding each sample's source-dataset name.
        eps: Lower bound for ``std`` and ``iqr`` to avoid zeros.

    Returns:
        Mapping of dataset name to a stats mapping (``{key: {"mean", "std", ...}}``),
        matching :func:`compute_stats`'s per-key shape.

    Raises:
        ValueError: If a sample lacks ``dataset_key`` (required to group by dataset).

    """
    collected: dict[str, dict[str, list[torch.Tensor]]] = {}
    for sample in samples:
        name = sample.get(dataset_key)
        if name is None:
            raise ValueError(
                f"compute_dataset_stats requires a {dataset_key!r} metadata key on every "
                "sample; none was found. Per-dataset normalization needs dataset provenance."
            )
        bucket = collected.setdefault(str(name), {k: [] for k in keys})
        for key in keys:
            if key not in sample:
                continue
            rows, _ = _valid_rows(sample[key].to(torch.float32), sample[f"{key}_mask"])
            if rows.numel():
                bucket[key].append(rows)

    stats: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for name, buckets in collected.items():
        stats[name] = {
            key: _reduce_stats(torch.cat(rows_list, dim=0), eps)
            for key, rows_list in buckets.items()
            if rows_list
        }
    return stats


def _reduce_stats(rows: torch.Tensor, eps: float) -> dict[str, torch.Tensor]:
    """Reduce stacked valid rows to the full statistic set.

    Args:
        rows: Valid positions of shape ``(N, F)`` or ``(N,)``.
        eps: Lower bound for ``std`` and ``iqr``.

    Returns:
        Mapping ``{"mean", "std", "min", "max", "median", "iqr"}`` of shape ``(F,)``
        (or ``()`` for scalar sequences).

    """
    q = torch.quantile(rows, torch.tensor([0.25, 0.5, 0.75]), dim=0)  # (3, F) or (3,)
    return {
        "mean": rows.mean(dim=0),
        "std": rows.std(dim=0).clamp_min(eps),
        "min": rows.amin(dim=0),
        "max": rows.amax(dim=0),
        "median": q[1],
        "iqr": (q[2] - q[0]).clamp_min(eps),
    }


def save_stats(stats: Mapping[str, Any], path: str | Path) -> None:
    """Save a stats mapping to JSON (tensors are stored as nested lists).

    Handles both the flat form from :func:`compute_stats` (``{key: {field: tensor}}``)
    and the nested per-dataset form from :func:`compute_dataset_stats`
    (``{dataset: {key: {field: tensor}}}``).

    Args:
        stats: A stats mapping.
        path: Destination ``.json`` file.

    """
    Path(path).write_text(json.dumps(_to_serializable(stats), indent=2), encoding="utf-8")


def load_stats(path: str | Path) -> dict[str, Any]:
    """Load a stats mapping saved by :func:`save_stats`, restoring float tensors.

    Restores either the flat or the nested per-dataset form (mirroring
    :func:`save_stats`).

    Args:
        path: Path to a JSON file written by :func:`save_stats`.

    Returns:
        The stats mapping with leaf lists restored to float tensors.

    """
    return _from_serializable(json.loads(Path(path).read_text(encoding="utf-8")))


def _to_serializable(obj: Any) -> Any:
    """Recursively turn tensors into lists, preserving the mapping structure."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {key: _to_serializable(value) for key, value in obj.items()}
    return obj


def _from_serializable(obj: Any) -> Any:
    """Recursively turn leaf lists/numbers back into float tensors."""
    if isinstance(obj, Mapping):
        return {key: _from_serializable(value) for key, value in obj.items()}
    return torch.tensor(obj, dtype=torch.float32)


def _valid_rows(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Flatten a value to its valid positions and report the feature shape.

    Args:
        value: A value of shape ``()``, ``(F,)``, ``(T,)`` or ``(T, F)``.
        mask: The matching mask of shape ``()`` or ``(T,)``.

    Returns:
        Tuple ``(rows, feature_shape)`` where ``rows`` stacks the valid
        positions (shape ``(N, F)`` or ``(N,)``) and ``feature_shape`` is the
        trailing feature shape (``(F,)`` or ``()``).

    """
    if mask.ndim == 0:
        # Vector/scalar: one position, valid iff the single flag is True.
        if not bool(mask):
            return value.new_zeros((0,)), value.shape
        return value.reshape(1, *value.shape), value.shape
    # Sequence: select the valid steps along the sequence axis.
    valid = value[mask.bool()]  # (N, F) or (N,)
    feature_shape = value.shape[1:]  # (F,) or ()
    return valid, feature_shape
