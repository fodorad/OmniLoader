"""MixUp / CutMix as a collate-time wrapper.

Cross-sample mixing must happen at the batch level, so this is a *collator* that
wraps a base collate function rather than a per-sample :class:`Transform`. It mixes
only the input **feature** tensors and emits the metadata a training loop needs to
compute the mixed loss itself — the paired (shuffled) targets ``<target>_b``, the
mixing coefficient ``mixup_lambda`` and the permutation ``mixup_index``. No loss
logic lives here, keeping the library strictly data-side.

* ``mode="mixup"`` — convex blend ``lam * x + (1 - lam) * x[perm]``.
* ``mode="cutmix"`` — replace a contiguous time span of each sequence with the
  permuted sample's span; ``mixup_lambda`` is the retained fraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omniloader.schema.spec import UnifiedSchema


class MixupCollator:
    """Wrap a base collator to apply MixUp/CutMix to feature tensors.

    Args:
        base_collate: The collator producing the batched dict (e.g.
            :func:`~omniloader.collate.unified_collate` or
            :class:`~omniloader.collate.DynamicCollator`).
        schema: The unified schema, used to pick feature/target keys.
        alpha: Beta distribution parameter; ``lam ~ Beta(alpha, alpha)``.
        mode: ``"mixup"`` (blend) or ``"cutmix"`` (span replacement).
        p: Per-batch probability of applying mixing (otherwise passthrough).
        seed: Seed for the internal NumPy RNG (deterministic across the run).

    Raises:
        ValueError: If ``mode`` is not ``"mixup"`` or ``"cutmix"``.

    """

    def __init__(
        self,
        base_collate: Any,
        schema: UnifiedSchema,
        alpha: float = 0.2,
        mode: str = "mixup",
        p: float = 1.0,
        seed: int = 0,
    ) -> None:
        if mode not in {"mixup", "cutmix"}:
            raise ValueError(f"mode must be 'mixup' or 'cutmix', got {mode!r}")
        self.base_collate = base_collate
        self.feature_keys = list(schema.feature_keys)
        self.target_keys = list(schema.target_keys)
        self.seq_features = {spec.name for spec in schema.features if spec.is_sequence}
        self.alpha = alpha
        self.mode = mode
        self.p = p
        self._rng = np.random.default_rng(seed)

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Collate then mix; adds ``mixup_lambda``, ``mixup_index`` and ``<t>_b``."""
        out = self.base_collate(batch)
        size = len(batch)
        if size < 2 or self._rng.random() >= self.p:
            # Passthrough with identity metadata so downstream code is uniform.
            out["mixup_lambda"] = 1.0
            out["mixup_index"] = torch.arange(size)
            for key in self.target_keys:
                if key in out:
                    out[f"{key}_b"] = out[key]
            return out

        perm = torch.from_numpy(self._rng.permutation(size))
        # One lam for the whole batch keeps the mixed loss consistent across keys.
        lam = float(self._rng.beta(self.alpha, self.alpha))
        rel_start = float(self._rng.random())  # shared relative cut position (cutmix)

        for key in self.feature_keys:
            value = out[key]
            other = value[perm]
            if self.mode == "mixup" or key not in self.seq_features:
                out[key] = lam * value + (1.0 - lam) * other  # (B, ...) convex blend
            else:  # cutmix: splice a shared time span from the permuted batch
                self._cut_span(value, other, lam, rel_start)
        for key in self.target_keys:
            if key in out:
                out[f"{key}_b"] = out[key][perm]
        out["mixup_lambda"] = lam
        out["mixup_index"] = perm
        return out

    @staticmethod
    def _cut_span(value: torch.Tensor, other: torch.Tensor, keep: float, rel_start: float) -> None:
        """Replace a ``(1 - keep)`` time span of ``value`` with ``other`` in place.

        Args:
            value: Batched sequence feature ``(B, T, ...)``.
            other: The permuted batch to splice in.
            keep: Fraction of ``value`` to retain (the batch's ``mixup_lambda``).
            rel_start: Relative start of the cut window in ``[0, 1)``.

        """
        length = value.shape[1]
        cut = round((1.0 - keep) * length)
        start = int(rel_start * (length - cut))
        value[:, start : start + cut] = other[:, start : start + cut]
