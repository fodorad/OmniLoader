"""Transform base classes and shared helpers.

A :class:`Transform` operates on an already-unified sample dict (the output of
:class:`~omniloader.schema.unify.SampleUnifier`) and is modality-agnostic — it
reasons only about vectors, sequences and their ``<name>_mask`` companions. Each
transform is:

* **seedable** — an optional :class:`torch.Generator` makes every draw
  reproducible;
* **mask-aware** — values are only altered where the mask is ``True`` (padding
  and placeholders are left untouched);
* **train/eval-aware** — augmentations set ``train_only = True`` so they are
  skipped during evaluation, while normalization runs in both phases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omniloader.schema.spec import UnifiedSchema

#: A unified sample dict mapping keys (and ``<name>_mask``) to tensors/metadata.
Sample = dict[str, Any]


def _rand(generator: torch.Generator | None) -> float:
    """Draw a single uniform ``[0, 1)`` scalar, optionally from a generator."""
    return float(torch.rand((), generator=generator).item())


class Transform(ABC):
    """Base class for per-sample transforms.

    Attributes:
        train_only: When ``True`` the transform is skipped unless ``training``
            is passed as ``True`` to :meth:`__call__`.

    """

    train_only: bool = False

    def __call__(
        self,
        sample: Sample,
        *,
        training: bool = True,
        generator: torch.Generator | None = None,
    ) -> Sample:
        """Apply the transform, honouring the train/eval gate.

        Args:
            sample: The unified sample dict to transform.
            training: Whether the pipeline is in training mode.
            generator: Optional RNG for reproducible randomness.

        Returns:
            The transformed sample (mutated in place and returned).

        """
        if self.train_only and not training:
            return sample
        return self.apply(sample, generator)

    @abstractmethod
    def apply(self, sample: Sample, generator: torch.Generator | None) -> Sample:
        """Transform ``sample`` unconditionally. Subclasses implement this."""


class Compose(Transform):
    """Apply a list of transforms in order.

    Args:
        transforms: The transforms to apply sequentially.

    """

    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms = list(transforms)

    def __call__(
        self,
        sample: Sample,
        *,
        training: bool = True,
        generator: torch.Generator | None = None,
    ) -> Sample:
        """Apply each contained transform in order (each honours its own gate)."""
        for transform in self.transforms:
            sample = transform(sample, training=training, generator=generator)
        return sample

    def apply(self, sample: Sample, generator: torch.Generator | None) -> Sample:
        """Apply every contained transform (train gate handled per-transform)."""
        for transform in self.transforms:
            sample = transform.apply(sample, generator)
        return sample


def broadcast_mask(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Broadcast a validity mask to the value's shape for ``torch.where``.

    Args:
        mask: Mask of shape ``()`` (vector) or ``(T,)`` (sequence).
        value: Value of shape ``()``, ``(F,)``, ``(T,)`` or ``(T, F)``.

    Returns:
        A boolean tensor broadcastable to ``value.shape``.

    """
    if mask.ndim < value.ndim:
        # Sequence value (T, F) with a (T,) mask -> add trailing feature axis.
        mask = mask.reshape(mask.shape + (1,) * (value.ndim - mask.ndim))
    return mask.expand_as(value)


def resolve_keys(keys: Sequence[str] | None, schema: UnifiedSchema | None) -> list[str]:
    """Return explicit ``keys`` or all feature keys from ``schema``."""
    if keys is not None:
        return list(keys)
    if schema is None:
        raise ValueError("Either keys or a schema must be provided")
    return list(schema.feature_keys)


def resolve_sequence_keys(keys: Sequence[str] | None, schema: UnifiedSchema | None) -> list[str]:
    """Return explicit ``keys`` or all sequence feature keys from ``schema``."""
    if keys is not None:
        return list(keys)
    if schema is None:
        raise ValueError("Either keys or a schema must be provided")
    return [spec.name for spec in schema.features if spec.is_sequence]
