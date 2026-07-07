"""Augmentation transforms: feature corruption, dropout and span masking.

These transforms are ``train_only`` — they self-skip during evaluation. Each one
respects the validity mask and is fully reproducible from an optional
:class:`torch.Generator`.

Literature techniques covered: Gaussian noise injection (feature corruption),
whole-feature ("stream"/modality) dropout, and contiguous span masking over
sequences (SpecAugment-style temporal masking).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from omniloader.transforms.base import (
    Sample,
    Transform,
    _rand,
    broadcast_mask,
    resolve_keys,
    resolve_sequence_keys,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omniloader.schema.spec import UnifiedSchema


class GaussianNoise(Transform):
    """Add zero-mean Gaussian noise to feature values (feature corruption).

    Args:
        keys: Feature keys to corrupt. When ``None``, every feature in the
            schema is eligible (requires ``schema``).
        std: Standard deviation of the additive noise.
        p: Per-sample probability of applying noise to each key.
        schema: Schema used to resolve ``keys`` when it is ``None``.

    """

    train_only = True

    def __init__(
        self,
        keys: Sequence[str] | None = None,
        std: float = 0.1,
        p: float = 0.5,
        schema: UnifiedSchema | None = None,
    ) -> None:
        self.keys = resolve_keys(keys, schema)
        self.std = std
        self.p = p

    def apply(self, sample: Sample, generator: torch.Generator | None) -> Sample:
        """Add noise to each eligible key with probability ``p``."""
        for key in self.keys:
            if key not in sample or _rand(generator) >= self.p:
                continue
            value = sample[key].to(torch.float32)  # (..., F) or (...,)
            noise = torch.randn(value.shape, generator=generator) * self.std
            mask_b = broadcast_mask(sample[f"{key}_mask"], value)
            sample[key] = torch.where(mask_b, value + noise, value)
        return sample


class FeatureDropout(Transform):
    """Drop entire features ("stream"/modality dropout).

    With probability ``p`` a feature's value is zeroed and its mask set to all
    ``False``, so a model treats it as absent. This generalises modality dropout
    to any set of feature streams. Optionally guarantees at least one eligible
    feature survives.

    Args:
        keys: Feature keys eligible for dropout. When ``None``, every feature in
            the schema is eligible (requires ``schema``).
        p: Per-sample probability of dropping each eligible feature.
        keep_at_least_one: When ``True``, never drop the last surviving feature.
        schema: Schema used to resolve ``keys`` when it is ``None``.

    """

    train_only = True

    def __init__(
        self,
        keys: Sequence[str] | None = None,
        p: float = 0.2,
        keep_at_least_one: bool = True,
        schema: UnifiedSchema | None = None,
    ) -> None:
        self.keys = resolve_keys(keys, schema)
        self.p = p
        self.keep_at_least_one = keep_at_least_one

    def apply(self, sample: Sample, generator: torch.Generator | None) -> Sample:
        """Zero and invalidate each eligible feature with probability ``p``."""
        # Only consider features that are present and currently have valid data.
        present = [k for k in self.keys if k in sample and bool(sample[f"{k}_mask"].any())]
        drop = [k for k in present if _rand(generator) < self.p]
        if self.keep_at_least_one and drop and len(drop) == len(present):
            drop = drop[1:]  # keep the first eligible feature valid
        for key in drop:
            sample[key] = torch.zeros_like(sample[key])
            sample[f"{key}_mask"] = torch.zeros_like(sample[f"{key}_mask"])
        return sample


class SpanMasking(Transform):
    """Zero contiguous spans of sequence features (SpecAugment-style masking).

    For each configured sequence feature, up to ``num_spans`` contiguous spans of
    length ``span_len`` are zeroed within the valid region. The validity mask is
    left intact — the positions still exist for the model and for targets; only
    their input content is corrupted, encouraging temporal robustness.

    Args:
        keys: Sequence-feature keys to mask. When ``None``, every sequence
            feature in the schema is eligible (requires ``schema``).
        num_spans: Number of spans to mask per key.
        span_len: Length of each span along the sequence axis.
        p: Per-sample probability of masking each eligible key.
        schema: Schema used to resolve sequence ``keys`` when it is ``None``.

    """

    train_only = True

    def __init__(
        self,
        keys: Sequence[str] | None = None,
        num_spans: int = 1,
        span_len: int = 5,
        p: float = 0.5,
        schema: UnifiedSchema | None = None,
    ) -> None:
        self.keys = resolve_sequence_keys(keys, schema)
        self.num_spans = num_spans
        self.span_len = span_len
        self.p = p

    def apply(self, sample: Sample, generator: torch.Generator | None) -> Sample:
        """Zero up to ``num_spans`` spans per eligible sequence feature."""
        for key in self.keys:
            if key not in sample or _rand(generator) >= self.p:
                continue
            mask = sample[f"{key}_mask"]  # (T,)
            valid = torch.nonzero(mask, as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            lo, hi = int(valid[0].item()), int(valid[-1].item()) + 1  # valid region [lo, hi)
            value = sample[key].clone()  # copy so we never mutate shared source tensors
            for _ in range(self.num_spans):
                if hi - lo <= 0:
                    break
                start = lo + int(
                    torch.randint(0, max(1, hi - lo), (1,), generator=generator).item()
                )
                end = min(start + self.span_len, hi)
                value[start:end] = 0  # zero the span along the sequence axis
            sample[key] = value
        return sample


class TimeWarp(Transform):
    """Randomly stretch/squeeze the sequence in time (speed perturbation).

    A single warp rate ``r`` in ``[min_rate, max_rate]`` is shared across the
    targeted keys (so aligned features stay in sync): each sequence is resampled
    to ``round(T * r)`` steps via linear interpolation and resized back to ``T``.
    Only floating-point sequence **features** are warped by default (interpolating
    integer class labels would be meaningless).

    Args:
        min_rate: Minimum warp rate (``< 1`` squeezes, ``> 1`` stretches).
        max_rate: Maximum warp rate.
        keys: Sequence keys to warp with a shared rate. When ``None``, all
            floating-point sequence feature keys in the schema are used.
        p: Per-sample probability of warping.
        schema: Schema used to resolve ``keys`` when it is ``None``.

    """

    train_only = True

    def __init__(
        self,
        min_rate: float = 0.8,
        max_rate: float = 1.25,
        keys: Sequence[str] | None = None,
        p: float = 0.5,
        schema: UnifiedSchema | None = None,
    ) -> None:
        if keys is not None:
            self.keys = list(keys)
        elif schema is None:
            raise ValueError("Either keys or a schema must be provided")
        else:
            self.keys = [
                spec.name
                for spec in schema.features
                if spec.is_sequence and spec.dtype.is_floating_point
            ]
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p = p

    def apply(self, sample: Sample, generator: torch.Generator | None) -> Sample:
        """Resample each targeted key by a single shared random rate."""
        if _rand(generator) >= self.p:
            return sample
        rate = self.min_rate + _rand(generator) * (self.max_rate - self.min_rate)
        for key in self.keys:
            if key not in sample:
                continue
            value = sample[key].to(torch.float32)  # (T, F) or (T,)
            length = value.shape[0]
            new_len = max(1, round(length * rate))
            squeeze = value.ndim == 1
            x = value[None, None] if squeeze else value.transpose(0, 1)[None]  # (1, C, T)
            x = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
            x = F.interpolate(x, size=length, mode="linear", align_corners=False)  # back to T
            warped = x[0, 0] if squeeze else x[0].transpose(0, 1)  # (T,) or (T, F)
            sample[key] = warped.to(sample[key].dtype)
        return sample


class FeatureMasking(Transform):
    """Zero contiguous bands of feature dimensions (channel/frequency masking).

    For each targeted key with a feature axis, ``num_masks`` bands of width up to
    ``max_width`` are zeroed across all time steps — the feature-axis analogue of
    :class:`SpanMasking`.

    Args:
        num_masks: Number of feature-dimension bands to zero per key.
        max_width: Maximum band width along the feature axis.
        keys: Feature keys to mask. When ``None``, every feature in the schema is
            used (requires ``schema``).
        p: Per-sample probability of masking each eligible key.
        schema: Schema used to resolve ``keys`` when it is ``None``.

    """

    train_only = True

    def __init__(
        self,
        num_masks: int = 1,
        max_width: int = 4,
        keys: Sequence[str] | None = None,
        p: float = 0.5,
        schema: UnifiedSchema | None = None,
    ) -> None:
        self.keys = resolve_keys(keys, schema)
        self.num_masks = num_masks
        self.max_width = max_width
        self.p = p

    def apply(self, sample: Sample, generator: torch.Generator | None) -> Sample:
        """Zero up to ``num_masks`` feature bands per eligible key."""
        for key in self.keys:
            if key not in sample or _rand(generator) >= self.p:
                continue
            value = sample[key]
            feat = value.shape[-1] if value.ndim >= 1 else 0
            if feat == 0:
                continue  # scalar value has no feature axis to mask
            value = value.clone()
            for _ in range(self.num_masks):
                width = int(torch.randint(0, self.max_width + 1, (1,), generator=generator).item())
                if width == 0 or width >= feat:
                    continue
                start = int(torch.randint(0, feat - width + 1, (1,), generator=generator).item())
                value[..., start : start + width] = 0  # zero the band across all steps
            sample[key] = value
        return sample
