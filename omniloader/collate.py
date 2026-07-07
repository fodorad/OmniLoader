"""Collate functions for batching unified samples.

Two collators are provided:

* :func:`unified_collate` — the default: every sample already shares the same
  tensor shapes (the unified schema, with sequences padded to a fixed ``time_dim``),
  so it simply stacks tensors and gathers string metadata into lists.
* :class:`DynamicCollator` — pads each sequence key only to the batch's longest
  sample instead of a fixed ``time_dim``. Use it with
  ``OmniLoader(pad_features=False)`` to avoid wasting memory/compute on padding
  when sequence lengths vary a lot across the batch.

Masks vs. cost: a ``<name>_mask`` guarantees *correctness* (padded steps never
corrupt the model output or the loss), but a padded step still costs compute and
memory (self-attention is ``O(T²)``). Dynamic padding targets that cost — so it is
only useful in the native-length regime. Under the default fixed ``time_dim``
(``pad_features=True``) every batch is already one length and :class:`DynamicCollator`
is a no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from omniloader.utils.padding import pad_or_crop_time_dim

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omniloader.schema.spec import UnifiedSchema


def unified_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of unified samples into a batched dict.

    Args:
        batch: Samples produced by :class:`~omniloader.loader.OmniLoader`.

    Returns:
        A dict where each tensor key holds a stacked tensor of shape
        ``(B, ...)`` and each non-tensor key holds a list of length ``B``.

    Raises:
        ValueError: If the batch is empty.

    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    keys = batch[0].keys()
    out: dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = list(values)
    return out


class DynamicCollator:
    """Collate with per-batch dynamic padding of sequence keys.

    Sequence values (those whose spec sets ``time_dim``) are padded to the
    longest sample in the batch — ``(B, L_batch, F)`` instead of a fixed
    ``(B, time_dim, F)`` — and their ``<name>_mask`` is rebuilt accordingly.
    Vectors, scalars and metadata are batched exactly as
    :func:`unified_collate` does. Pair with ``OmniLoader(pad_features=False)`` so
    samples reach the collator at their native length; under the default fixed
    ``time_dim`` every sample already shares one length, so this collator has
    nothing to shrink and behaves like :func:`unified_collate`.

    Args:
        schema: The unified schema, used to identify sequence keys and their
            placeholder fill values.
        keys: Optional subset of sequence keys to pad dynamically. When ``None``,
            every sequence key in the schema is padded dynamically.

    """

    def __init__(self, schema: UnifiedSchema, keys: Sequence[str] | None = None) -> None:
        seq = {spec.name: spec for spec in schema.specs if spec.is_sequence}
        self.seq_specs = seq if keys is None else {k: seq[k] for k in keys if k in seq}

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Collate ``batch``, dynamically padding the configured sequence keys.

        Args:
            batch: Samples to collate.

        Returns:
            The batched dict.

        Raises:
            ValueError: If the batch is empty.

        """
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        out: dict[str, Any] = {}
        for key in batch[0]:
            # Sequence masks are produced alongside their value key; skip them.
            if key.endswith("_mask") and key[:-5] in self.seq_specs:
                continue
            values = [sample[key] for sample in batch]
            if key in self.seq_specs:
                spec = self.seq_specs[key]
                masks = [sample[f"{key}_mask"] for sample in batch]
                length = max(v.shape[0] for v in values)  # batch max sequence length
                # (T_i, ...) -> (L, ...) padded with the spec placeholder value.
                out[key] = torch.stack(
                    [pad_or_crop_time_dim(v, length, spec.placeholder)[0] for v in values]
                )
                # (T_i,) bool mask -> (L,) padded with False.
                out[f"{key}_mask"] = torch.stack(
                    [pad_or_crop_time_dim(m.to(torch.float32), length)[0].bool() for m in masks]
                )
            elif isinstance(values[0], torch.Tensor):
                out[key] = torch.stack(values, dim=0)
            else:
                out[key] = list(values)
        return out
