"""Turn heterogeneous raw samples into one unified, masked format.

:class:`SampleUnifier` maps any raw sample dict onto the full union of features
and targets described by a :class:`~omniloader.schema.spec.UnifiedSchema`. Keys the
source dataset provides are copied (and, for sequences, padded/cropped to the
declared length); keys it lacks are filled with a placeholder tensor and an
all-``False`` mask. Every value is paired with a ``<name>_mask`` boolean tensor
so downstream code can distinguish real data from padding or placeholders. The
module is modality- and model-agnostic: it only ever reasons about vectors and
sequences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from omniloader.utils.padding import pad_or_crop_time_dim

if TYPE_CHECKING:
    from collections.abc import Mapping

    from omniloader.schema.spec import TensorSpec, UnifiedSchema

#: Metadata keys copied verbatim from the raw sample when present.
METADATA_KEYS = ("dataset", "subset", "key")


def _as_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """Coerce a raw value to a tensor of the given dtype.

    Args:
        value: Raw value from the source sample (tensor, array, or scalar).
        dtype: The target element dtype.

    Returns:
        A tensor cast to ``dtype``.

    """
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(dtype)


class SampleUnifier:
    """Map raw sample dicts onto a shared, fully-masked schema.

    Args:
        schema: The unified schema describing every feature and target.
        pad_features: When ``True`` (default), sequence values longer or shorter
            than their declared ``time_dim`` are cropped or padded via
            :func:`~omniloader.utils.padding.pad_or_crop_time_dim`. When ``False``,
            values are assumed to already match and are copied as-is.

    """

    def __init__(self, schema: UnifiedSchema, pad_features: bool = True) -> None:
        self.schema = schema
        self.pad_features = pad_features

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Unify a single raw sample.

        Args:
            sample: Raw sample dict. May contain any subset of the schema's
                keys plus optional ``<name>_mask`` entries and metadata.

        Returns:
            A dict holding, for every schema key, the value tensor and its
            ``<name>_mask``, plus any metadata found in the source sample.

        """
        out: dict[str, Any] = {}

        for key in METADATA_KEYS:
            if key in sample:
                out[key] = sample[key]

        for spec in self.schema.specs:
            value, mask = self._resolve(spec, sample)
            out[spec.name] = value
            out[f"{spec.name}_mask"] = mask

        return out

    def _resolve(
        self, spec: TensorSpec, sample: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve the value and mask for one spec.

        Args:
            spec: The feature or target spec to resolve.
            sample: The raw sample being unified.

        Returns:
            Tuple of ``(value, mask)`` tensors matching the spec's shapes.

        """
        if spec.name not in sample:
            # Missing key -> placeholder value + all-False mask.
            return spec.placeholder_value(), spec.placeholder_mask()

        value = _as_tensor(sample[spec.name], spec.dtype)
        supplied_mask = sample.get(f"{spec.name}_mask")

        if not spec.is_sequence:
            # Vector/scalar: mask is a single flag (shape ()) for the whole value.
            mask = (
                _as_tensor(supplied_mask, torch.bool)
                if supplied_mask is not None
                else torch.ones(spec.mask_shape, dtype=torch.bool)
            )
            return value, mask

        # Sequence: optionally pad/crop the sequence axis to the declared length.
        if self.pad_features and spec.time_dim is not None:
            # (T, F) or (T,) -> (time_dim, F) or (time_dim,); mask -> (time_dim,)
            value, derived_mask = pad_or_crop_time_dim(value, spec.time_dim, spec.placeholder)
            if supplied_mask is not None:
                # Align a dataset-supplied mask to the same length, then AND it
                # with the padding-derived mask so both constraints apply.
                supplied_mask, _ = pad_or_crop_time_dim(
                    _as_tensor(supplied_mask, torch.float32), spec.time_dim
                )  # (T,) -> (time_dim,)
                mask = supplied_mask.to(torch.bool) & derived_mask
            else:
                mask = derived_mask
        else:
            mask = (
                _as_tensor(supplied_mask, torch.bool)
                if supplied_mask is not None
                else torch.ones(value.shape[0], dtype=torch.bool)  # (T,)
            )
        return value, mask
