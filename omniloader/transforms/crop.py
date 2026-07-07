"""Sequence cropping transforms (random window for train, center for eval).

Both crops take a fixed-length window along the sequence axis. A **single shared
offset** is applied across every targeted sequence key, so time-aligned features
and per-step targets stay in sync. Keys shorter than the requested window are
padded back to ``length`` with their placeholder value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from omniloader.transforms.base import Sample, Transform
from omniloader.utils.padding import pad_or_crop_time_dim

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omniloader.schema.spec import UnifiedSchema


def _all_sequence_keys(keys: Sequence[str] | None, schema: UnifiedSchema | None) -> list[str]:
    """Return explicit ``keys`` or every sequence key (features *and* targets).

    Cropping must move features and their aligned per-step targets together, so
    unlike :func:`~omniloader.transforms.base.resolve_sequence_keys` (features
    only) this includes sequence targets.
    """
    if keys is not None:
        return list(keys)
    if schema is None:
        raise ValueError("Either keys or a schema must be provided")
    return [spec.name for spec in schema.specs if spec.is_sequence]


def _valid_length(sample: Sample, keys: list[str]) -> int:
    """Return the largest valid-length across the targeted keys present in ``sample``."""
    lengths = [
        int(sample[f"{k}_mask"].sum()) for k in keys if k in sample and f"{k}_mask" in sample
    ]
    return max(lengths) if lengths else 0


def _apply_window(sample: Sample, keys: list[str], start: int, length: int) -> None:
    """Slice ``[start:start+length]`` from each key and pad back to ``length``."""
    for key in keys:
        if key not in sample:
            continue
        value = sample[key][start : start + length]
        mask = sample[f"{key}_mask"][start : start + length]
        # If the slice is short (key ran out), pad back to the requested length.
        # (T', ...) -> (length, ...); mask (T',) -> (length,)
        sample[key] = pad_or_crop_time_dim(value, length)[0]
        sample[f"{key}_mask"] = pad_or_crop_time_dim(mask.to(torch.float32), length)[0].bool()


class RandomCrop(Transform):
    """Crop a random contiguous window of the sequence (training augmentation).

    Args:
        length: Window length along the sequence axis.
        keys: Sequence keys to crop with a shared offset. When ``None``, every
            sequence key in the schema (features *and* aligned targets) is
            cropped together (requires ``schema``).
        schema: Schema used to resolve ``keys`` when it is ``None``.

    """

    train_only = True

    def __init__(
        self,
        length: int,
        keys: Sequence[str] | None = None,
        schema: UnifiedSchema | None = None,
    ) -> None:
        self.length = length
        self.keys = _all_sequence_keys(keys, schema)

    def apply(self, sample: Sample, generator: torch.Generator | None) -> Sample:
        """Crop all targeted keys at one random offset within the valid region."""
        valid = _valid_length(sample, self.keys)
        max_start = max(0, valid - self.length)
        start = (
            int(torch.randint(0, max_start + 1, (1,), generator=generator).item())
            if max_start > 0
            else 0
        )
        _apply_window(sample, self.keys, start, self.length)
        return sample


class CenterCrop(Transform):
    """Crop the centered window of the sequence (deterministic; used for eval).

    Args:
        length: Window length along the sequence axis.
        keys: Sequence keys to crop. When ``None``, every sequence key in the
            schema is cropped (requires ``schema``).
        schema: Schema used to resolve ``keys`` when it is ``None``.

    """

    train_only = False

    def __init__(
        self,
        length: int,
        keys: Sequence[str] | None = None,
        schema: UnifiedSchema | None = None,
    ) -> None:
        self.length = length
        self.keys = _all_sequence_keys(keys, schema)

    def apply(self, sample: Sample, generator: torch.Generator | None) -> Sample:  # noqa: ARG002
        """Crop all targeted keys at the centered offset within the valid region."""
        valid = _valid_length(sample, self.keys)
        start = max(0, (valid - self.length) // 2)
        _apply_window(sample, self.keys, start, self.length)
        return sample
