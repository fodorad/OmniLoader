"""Feature normalization transforms.

All normalizers apply an affine map ``(x - offset) / (scale + eps)`` only where
the validity mask is ``True`` — padding and placeholder positions keep their
original value. The stats-based variants take precomputed statistics (see
:func:`~omniloader.transforms.stats.compute_stats`); :class:`InstanceNormalize`
derives them per sample. All run in both training and evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from omniloader.transforms.base import Sample, Transform, broadcast_mask, resolve_keys

if TYPE_CHECKING:
    from omniloader.schema.spec import UnifiedSchema


def _parse_pair(stat: Any, keys: tuple[str, str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse one key's stats into two float tensors from a pair or a mapping.

    Args:
        stat: Either a two-element ``(a, b)`` sequence or a mapping containing
            ``keys``.
        keys: The mapping field names to read (e.g. ``("mean", "std")``).

    Returns:
        The two statistics as float tensors.

    """
    if isinstance(stat, Mapping):
        a, b = stat[keys[0]], stat[keys[1]]
    else:
        a, b = stat
    return torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32)


def _affine(
    sample: Sample, key: str, offset: torch.Tensor, scale: torch.Tensor, eps: float
) -> None:
    """Apply ``(x - offset) / (scale + eps)`` to ``key`` where its mask is valid."""
    value = sample[key].to(torch.float32)  # (..., F) or (...,)
    normalized = (value - offset) / (scale + eps)
    mask_b = broadcast_mask(sample[f"{key}_mask"], value)  # -> value.shape
    sample[key] = torch.where(mask_b, normalized, value)


def _instance_affine(sample: Sample, key: str, eps: float) -> None:
    """Standardize ``key`` by its own valid statistics, in place.

    For a sequence value the mean/std are taken over the valid time steps (per
    feature); for a vector value, over the feature axis. A key that is absent or
    has no valid position is left unchanged.
    """
    if key not in sample:
        return
    value = sample[key].to(torch.float32)
    mask = sample[f"{key}_mask"]
    if mask.ndim == 0:  # vector: normalize over the feature axis
        if not bool(mask):
            return
        offset, scale = value.mean(), value.std()
    else:  # sequence: per-feature stats over valid steps
        valid = value[mask.bool()]  # (N, F) or (N,)
        if valid.numel() == 0:
            return
        offset, scale = valid.mean(dim=0), valid.std(dim=0)
    _affine(sample, key, offset, scale, eps)


class _StatsNormalizer(Transform):
    """Base for stats-driven affine normalizers (offset/scale from two fields)."""

    train_only = False
    _fields: tuple[str, str] = ("mean", "std")

    def __init__(self, stats: Mapping[str, Any], eps: float = 1e-6) -> None:
        self.stats = {key: _parse_pair(stat, self._fields) for key, stat in stats.items()}
        self.eps = eps

    def apply(self, sample: Sample, generator: torch.Generator | None) -> Sample:  # noqa: ARG002
        """Normalize each configured key where its mask is valid."""
        for key, (offset, scale) in self.stats.items():
            if key in sample:
                _affine(sample, key, offset, self._scale(offset, scale), self.eps)
        return sample

    def _scale(self, offset: torch.Tensor, second: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        """Return the divisor given the two parsed statistics."""
        return second


class Normalize(_StatsNormalizer):
    """Standardize to zero mean and unit variance from ``(mean, std)`` stats."""

    _fields = ("mean", "std")


class MinMaxNormalize(_StatsNormalizer):
    """Scale to ``[0, 1]`` from ``(min, max)`` stats (divisor is ``max - min``)."""

    _fields = ("min", "max")

    def _scale(self, offset: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """Divide by the value range ``max - min``."""
        return second - offset


class RobustNormalize(_StatsNormalizer):
    """Standardize by median and IQR from ``(median, iqr)`` stats (outlier-robust)."""

    _fields = ("median", "iqr")


class InstanceNormalize(Transform):
    """Standardize each sample by its own valid statistics (no external stats).

    For a sequence value the mean/std are taken over the valid time steps
    (per feature); for a vector value, over the feature axis.

    Args:
        keys: Feature keys to normalize. When ``None``, every feature in the
            schema is used (requires ``schema``).
        eps: Small constant added to the std.
        schema: Schema used to resolve ``keys`` when it is ``None``.

    """

    train_only = False

    def __init__(
        self,
        keys: Sequence[str] | None = None,
        eps: float = 1e-6,
        schema: UnifiedSchema | None = None,
    ) -> None:
        self.keys = resolve_keys(keys, schema)
        self.eps = eps

    def apply(self, sample: Sample, generator: torch.Generator | None) -> Sample:  # noqa: ARG002
        """Standardize each key using statistics from its own valid positions."""
        for key in self.keys:
            _instance_affine(sample, key, self.eps)
        return sample


class PerDatasetNormalize(Transform):
    """Standardize each sample by **its source dataset's** own ``(mean, std)`` stats.

    The remedy for cross-corpus domain shift when training jointly on several
    datasets that share a feature extractor: each dataset is standardized by its own
    statistics (see :func:`~omniloader.transforms.stats.compute_dataset_stats`),
    selected at run time from the ``dataset`` metadata key every unified sample
    carries. Because per-dataset stats are undefined for a source not seen at
    training time, a ``fallback`` governs unknown ids — crucial for deploying on new
    sources:

    * ``"instance"`` — standardize the sample by its own valid statistics (no external
      stats needed; identical at train and inference);
    * ``"union"`` — apply the pooled ``union_stats`` (a neutral default);
    * ``"identity"`` — leave the sample unchanged.

    Never guess another dataset's stats for an unknown source — that would inject that
    corpus's shift. Runs in both training and evaluation.

    Args:
        stats: Mapping of dataset name to a per-key stats mapping (each an
            ``(mean, std)`` pair or a mapping with ``"mean"``/``"std"``).
        fallback: Policy for a sample whose dataset id is not in ``stats``.
        union_stats: Pooled per-key stats, required when ``fallback="union"``.
        dataset_key: Metadata key holding each sample's source-dataset name.
        eps: Small constant added to the divisor.

    Raises:
        ValueError: If ``fallback`` is unknown, or ``"union"`` without ``union_stats``.

    """

    train_only = False
    _fields: tuple[str, str] = ("mean", "std")

    def __init__(
        self,
        stats: Mapping[str, Mapping[str, Any]],
        fallback: str = "instance",
        union_stats: Mapping[str, Any] | None = None,
        dataset_key: str = "dataset",
        eps: float = 1e-6,
    ) -> None:
        if fallback not in {"instance", "union", "identity"}:
            raise ValueError(
                f"fallback must be 'instance', 'union' or 'identity', got {fallback!r}"
            )
        if fallback == "union" and union_stats is None:
            raise ValueError("fallback='union' requires union_stats")
        self.stats = {
            name: {key: _parse_pair(stat, self._fields) for key, stat in per.items()}
            for name, per in stats.items()
        }
        self.union = (
            {key: _parse_pair(stat, self._fields) for key, stat in union_stats.items()}
            if union_stats is not None
            else None
        )
        self.fallback = fallback
        self.dataset_key = dataset_key
        self.eps = eps
        # Keys eligible for the instance fallback: every key seen across datasets.
        self._all_keys = sorted({key for per in self.stats.values() for key in per})

    def apply(self, sample: Sample, generator: torch.Generator | None) -> Sample:  # noqa: ARG002
        """Normalize ``sample`` by its dataset's stats, or the fallback if unknown."""
        name = sample.get(self.dataset_key)
        per = self.stats.get(str(name)) if name is not None else None
        if per is not None:
            for key, (offset, scale) in per.items():
                if key in sample:
                    _affine(sample, key, offset, scale, self.eps)
            return sample
        # Unknown source: apply the configured fallback (never another dataset's stats).
        if self.fallback == "identity":
            return sample
        if self.fallback == "union":
            assert self.union is not None  # guaranteed by __init__ validation
            for key, (offset, scale) in self.union.items():
                if key in sample:
                    _affine(sample, key, offset, scale, self.eps)
            return sample
        for key in self._all_keys:  # fallback == "instance"
            _instance_affine(sample, key, self.eps)
        return sample
