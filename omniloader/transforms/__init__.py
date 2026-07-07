"""Configurable, per-sample transforms: normalization and augmentation.

This subpackage groups the transform pipeline used after unification:

* :mod:`~omniloader.transforms.base` — :class:`Transform`, :class:`Compose` and
  shared helpers;
* :mod:`~omniloader.transforms.normalize` — :class:`Normalize`,
  :class:`MinMaxNormalize`, :class:`RobustNormalize`, :class:`InstanceNormalize`;
* :mod:`~omniloader.transforms.augment` — :class:`GaussianNoise`,
  :class:`FeatureDropout`, :class:`SpanMasking`, :class:`TimeWarp`,
  :class:`FeatureMasking`;
* :mod:`~omniloader.transforms.crop` — :class:`RandomCrop`, :class:`CenterCrop`;
* :mod:`~omniloader.transforms.mix` — :class:`MixupCollator` (collate-time);
* :mod:`~omniloader.transforms.stats` — statistics helpers for normalization.

Compose transforms with :class:`Compose`, or build one from a list of config dicts
with :func:`build_transform`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omniloader.transforms.augment import (
    FeatureDropout,
    FeatureMasking,
    GaussianNoise,
    SpanMasking,
    TimeWarp,
)
from omniloader.transforms.base import Compose, Transform
from omniloader.transforms.crop import CenterCrop, RandomCrop
from omniloader.transforms.mix import MixupCollator
from omniloader.transforms.normalize import (
    InstanceNormalize,
    MinMaxNormalize,
    Normalize,
    PerDatasetNormalize,
    RobustNormalize,
)
from omniloader.transforms.stats import (
    compute_dataset_stats,
    compute_feature_stats,
    compute_stats,
    load_stats,
    save_stats,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from omniloader.schema.spec import UnifiedSchema

#: Registry mapping transform names (as used in config) to their classes.
TRANSFORMS: dict[str, type[Transform]] = {
    "normalize": Normalize,
    "min_max_normalize": MinMaxNormalize,
    "robust_normalize": RobustNormalize,
    "instance_normalize": InstanceNormalize,
    "per_dataset_normalize": PerDatasetNormalize,
    "gaussian_noise": GaussianNoise,
    "feature_dropout": FeatureDropout,
    "span_masking": SpanMasking,
    "feature_masking": FeatureMasking,
    "time_warp": TimeWarp,
    "random_crop": RandomCrop,
    "center_crop": CenterCrop,
}

#: Transforms taking precomputed ``stats`` (not a schema) in their config.
_STATS_TRANSFORMS = frozenset(
    {"normalize", "min_max_normalize", "robust_normalize", "per_dataset_normalize"}
)


def build_transform(
    configs: Sequence[Mapping[str, Any]],
    schema: UnifiedSchema | None = None,
) -> Compose | None:
    """Build a :class:`Compose` from a list of transform config dicts.

    Each config dict has a ``name`` (a key of :data:`TRANSFORMS`) and the
    remaining keys are forwarded as keyword arguments. Schema-aware transforms
    receive ``schema`` automatically when they declare no explicit ``keys``;
    stats-based normalizers read their ``stats`` from the config — either inline,
    or from a saved JSON file via ``stats_path`` (and ``union_stats_path`` for
    :class:`~omniloader.transforms.normalize.PerDatasetNormalize`), loaded with
    :func:`~omniloader.transforms.stats.load_stats`. This lets a config reference
    stats produced once by ``omniloader compute-stats`` instead of inlining them.

    Args:
        configs: A list of transform specifications.
        schema: The unified schema, injected into schema-aware transforms.

    Returns:
        A :class:`Compose`, or ``None`` if ``configs`` is empty.

    Raises:
        ValueError: If a config names an unknown transform.

    """
    transforms: list[Transform] = []
    for cfg in configs:
        params = dict(cfg)
        name = params.pop("name")
        if name not in TRANSFORMS:
            raise ValueError(f"Unknown transform {name!r}; expected one of {sorted(TRANSFORMS)}")
        # Load stats from a saved JSON file when referenced by path (inline wins).
        stats_path = params.pop("stats_path", None)
        if stats_path is not None:
            params.setdefault("stats", load_stats(stats_path))
        union_stats_path = params.pop("union_stats_path", None)
        if union_stats_path is not None:
            params.setdefault("union_stats", load_stats(union_stats_path))
        cls = TRANSFORMS[name]
        # Inject the schema for schema-aware transforms unless keys are explicit.
        if name not in _STATS_TRANSFORMS and "keys" not in params:
            params.setdefault("schema", schema)
        transforms.append(cls(**params))
    return Compose(transforms) if transforms else None


__all__ = [
    "TRANSFORMS",
    "CenterCrop",
    "Compose",
    "FeatureDropout",
    "FeatureMasking",
    "GaussianNoise",
    "InstanceNormalize",
    "MinMaxNormalize",
    "MixupCollator",
    "Normalize",
    "PerDatasetNormalize",
    "RandomCrop",
    "RobustNormalize",
    "SpanMasking",
    "TimeWarp",
    "Transform",
    "build_transform",
    "compute_dataset_stats",
    "compute_feature_stats",
    "compute_stats",
    "load_stats",
    "save_stats",
]
