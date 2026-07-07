"""Class-balance helpers for categorical targets.

Two complementary tools, named for what consumes their output, both counting only
valid (unmasked) target positions:

* :func:`class_weights_for_sampler` — one weight **per sample** for the *sampler*
  (:class:`~omniloader.sampling.strategies.SubsampleConfig` ``sample_weights``), so an
  imbalanced dataset is *resampled* toward a flat class histogram.
* :func:`class_histogram` / :func:`class_weights_for_loss` — the exact per-class counts
  and a ``(num_classes,)`` weight vector for a **loss** (e.g. ``CrossEntropyLoss(weight=...)``).
  These are data-side statistics: compute once, persist
  (``omniloader class-weights-for-loss``), and reuse — the loss/model that consumes
  them stays in your training code.

Unlike the per-sample sampler weights, the histogram/loss weights count **every valid
labelled position** (each valid step of a framewise ``(T,)`` target, or the scalar of a
vector target), matching what a per-position loss actually sees.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable


def _sample_class(value: torch.Tensor, mask: torch.Tensor | None) -> int | None:
    """Return the representative class id of a sample, or ``None`` if invalid.

    Args:
        value: A scalar class id ``()`` or a per-step class sequence ``(T,)``.
        mask: The matching validity mask, or ``None``.

    Returns:
        The class id (the value for a scalar, the most common valid step for a
        sequence), or ``None`` when nothing is valid.

    """
    if value.ndim == 0:
        if mask is not None and mask.ndim == 0 and not bool(mask):
            return None
        return int(value)
    # Sequence of class ids: use the most frequent valid step.
    valid = value[mask.bool()] if mask is not None else value
    if valid.numel() == 0:
        return None
    return int(torch.mode(valid).values)


def class_weights_for_sampler(
    samples: Iterable[dict[str, torch.Tensor]],
    target_key: str,
) -> list[float]:
    """Compute per-sample inverse-class-frequency weights for a *sampler*.

    Feed the result to a sampler (e.g.
    :class:`~omniloader.sampling.strategies.SubsampleConfig` ``sample_weights``) to
    *resample* an imbalanced dataset toward a flat class histogram. The loss-side
    counterpart is :func:`class_weights_for_loss`.

    Args:
        samples: An iterable of unified sample dicts (e.g. a source dataset or an
            :class:`~omniloader.loader.OmniLoader`). Iterated once.
        target_key: A categorical (int) target name; its ``<name>_mask`` is used
            to skip invalid positions.

    Returns:
        A list of per-sample weights (``1 / class_count`` for the sample's class,
        ``0`` when the sample has no valid target), aligned with ``samples``.

    """
    classes: list[int | None] = []
    for sample in samples:
        value = sample.get(target_key)
        if value is None:
            classes.append(None)
        else:
            classes.append(_sample_class(value, sample.get(f"{target_key}_mask")))
    counts = Counter(c for c in classes if c is not None)
    return [0.0 if c is None else 1.0 / counts[c] for c in classes]


def _valid_class_ids(value: torch.Tensor, mask: torch.Tensor | None) -> list[int]:
    """Return the valid class ids of a sample (all valid steps, or the scalar).

    Args:
        value: A scalar class id ``()`` or a per-step class sequence ``(T,)``.
        mask: The matching validity mask, or ``None``.

    Returns:
        Every valid class id — one for a scalar, one per valid step for a sequence.

    """
    if value.ndim == 0:
        if mask is not None and mask.ndim == 0 and not bool(mask):
            return []
        return [int(value)]
    valid = value[mask.bool()] if mask is not None else value.reshape(-1)
    return [int(c) for c in valid.tolist()]


def class_histogram(
    samples: Iterable[dict[str, torch.Tensor]],
    target_key: str,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Count valid instances of each class for a categorical target.

    Args:
        samples: An iterable of unified sample dicts (a source dataset or an
            :class:`~omniloader.loader.OmniLoader`). Iterated once.
        target_key: A categorical (int) target name; its ``<name>_mask`` skips
            invalid positions.
        num_classes: Number of classes. When ``None``, inferred as ``max id + 1``
            (so classes with zero instances are only included when given explicitly).

    Returns:
        A ``(num_classes,)`` ``int64`` tensor of per-class counts.

    Raises:
        ValueError: If an observed class id is negative or ``>= num_classes``.

    """
    counts: Counter[int] = Counter()
    for sample in samples:
        value = sample.get(target_key)
        if value is None:
            continue
        counts.update(_valid_class_ids(value, sample.get(f"{target_key}_mask")))
    size = num_classes if num_classes is not None else (max(counts) + 1 if counts else 0)
    hist = torch.zeros(size, dtype=torch.int64)
    for cls, n in counts.items():
        if cls < 0 or cls >= size:
            raise ValueError(f"class id {cls} out of range for num_classes={size}")
        hist[cls] = n
    return hist


def class_weights_for_loss(
    samples: Iterable[dict[str, torch.Tensor]],
    target_key: str,
    num_classes: int | None = None,
    scheme: str = "inverse",
    beta: float = 0.999,
) -> torch.Tensor:
    """Compute per-class weights for a *loss* from a categorical target.

    Intended for ``CrossEntropyLoss(weight=...)`` and similar — it *reweights* the
    loss instead of resampling the data (the sampler-side counterpart is
    :func:`class_weights_for_sampler`). Weights are normalised so they average ``1``
    across classes (the loss keeps its scale). Classes with zero instances get
    weight ``0``.

    Args:
        samples: An iterable of unified sample dicts. Iterated once.
        target_key: A categorical (int) target name.
        num_classes: Number of classes (``None`` infers ``max id + 1``).
        scheme: ``"inverse"`` (inverse frequency ``1 / count``) or ``"effective"``
            (effective number of samples, Cui et al. 2019:
            ``(1 - beta) / (1 - beta ** count)``).
        beta: Re-weighting strength for ``scheme="effective"`` (closer to ``1`` =
            stronger balancing).

    Returns:
        A ``(num_classes,)`` ``float32`` weight tensor.

    Raises:
        ValueError: If ``scheme`` is not ``"inverse"`` or ``"effective"``.

    """
    hist = class_histogram(samples, target_key, num_classes).to(torch.float64)
    zero = torch.zeros_like(hist)
    if scheme == "inverse":
        raw = torch.where(hist > 0, 1.0 / hist.clamp_min(1), zero)
    elif scheme == "effective":
        effective = 1.0 - torch.pow(beta, hist)
        raw = torch.where(hist > 0, (1.0 - beta) / effective.clamp_min(1e-12), zero)
    else:
        raise ValueError(f"scheme must be 'inverse' or 'effective', got {scheme!r}")
    total = raw.sum()
    if total > 0:
        raw = raw * (hist.numel() / total)  # average weight ~1 across classes
    return raw.to(torch.float32)
