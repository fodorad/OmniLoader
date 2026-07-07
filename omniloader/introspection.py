"""Introspection helpers: describe a dataset mix and validate it against specs.

:func:`describe` summarises a collection of datasets — the union schema, which
dataset provides which key (the *coverage matrix*), the fraction of valid (non-
placeholder) positions per key, and class histograms for integer targets.
:func:`validate` dry-runs each dataset, checking that the raw samples it emits
match the shape and rank their :class:`~omniloader.schema.spec.TensorSpec`
declares — catching schema/data mismatches before training starts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from omniloader.schema.spec import UnifiedSchema
from omniloader.schema.unify import SampleUnifier

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omniloader.loader import SizedDataset
    from omniloader.schema.spec import DatasetSchema


def _dataset_name(dataset: SizedDataset, index: int) -> str:
    """Best-effort name for a dataset: its ``dataset`` metadata or ``dataset_<i>``."""
    if len(dataset):  # type: ignore[arg-type]
        meta = dataset[0].get("dataset")
        if isinstance(meta, str):
            return meta
    return f"dataset_{index}"


@dataclass
class Report:
    """A human-readable summary of a dataset mix.

    Attributes:
        names: Per-dataset names.
        sizes: Per-dataset sample counts.
        union_keys: Every feature/target key in the union schema.
        coverage: ``{dataset_name: {key: provided?}}``.
        valid_fraction: ``{key: mean fraction of valid positions}`` (sampled).
        class_distributions: ``{target_key: {class_id: count}}`` (sampled).

    """

    names: list[str] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    union_keys: list[str] = field(default_factory=list)
    coverage: dict[str, dict[str, bool]] = field(default_factory=dict)
    valid_fraction: dict[str, float] = field(default_factory=dict)
    class_distributions: dict[str, dict[int, int]] = field(default_factory=dict)

    def __str__(self) -> str:
        """Render the report as an aligned text table."""
        lines = ["OmniLoader dataset report", "=" * 25]
        for name, size in zip(self.names, self.sizes):
            lines.append(f"  {name}: {size} samples")
        lines.append("")
        width = max((len(k) for k in self.union_keys), default=3)
        header = "  ".join(n[:6].ljust(6) for n in self.names)
        lines.append(f"coverage  {'key'.ljust(width)}  {header}")
        for key in self.union_keys:
            marks = "  ".join(
                ("  ✓   " if self.coverage[name][key] else "  ·   ")[:6] for name in self.names
            )
            frac = self.valid_fraction.get(key)
            frac_s = f"  valid={frac:.2f}" if frac is not None else ""
            lines.append(f"          {key.ljust(width)}  {marks}{frac_s}")
        for target, dist in self.class_distributions.items():
            counts = ", ".join(f"{cls}:{n}" for cls, n in sorted(dist.items()))
            lines.append(f"classes   {target}: {counts}")
        return "\n".join(lines)


def describe(
    datasets: Sequence[SizedDataset],
    schemas: Sequence[DatasetSchema],
    max_samples: int = 64,
) -> Report:
    """Summarise a dataset mix into a :class:`Report`.

    Args:
        datasets: The source datasets.
        schemas: One schema per dataset.
        max_samples: Cap on samples read per dataset for the valid-fraction and
            class-distribution statistics.

    Returns:
        The populated :class:`Report`.

    """
    schema = UnifiedSchema(list(schemas))
    names = [_dataset_name(d, i) for i, d in enumerate(datasets)]
    sizes = [len(d) for d in datasets]  # type: ignore[arg-type]
    union_keys = [spec.name for spec in schema.specs]
    coverage = {
        name: {key: key in sch.keys for key in union_keys} for name, sch in zip(names, schemas)
    }

    # Unify samples so masks exist; the valid fraction then reflects real data
    # vs padding/placeholders across the whole mix.
    unifier = SampleUnifier(schema)
    valid_hits: dict[str, float] = dict.fromkeys(union_keys, 0.0)
    valid_total: dict[str, float] = dict.fromkeys(union_keys, 0.0)
    int_targets = [spec.name for spec in schema.targets if not spec.dtype.is_floating_point]
    class_dist: dict[str, Counter] = {t: Counter() for t in int_targets}

    for dataset in datasets:
        for i in range(min(len(dataset), max_samples)):  # type: ignore[arg-type]
            sample = unifier(dataset[i])
            for key in union_keys:
                mask = sample[f"{key}_mask"]
                valid_hits[key] += float(mask.sum())
                valid_total[key] += mask.numel()
            for target in int_targets:
                value, mask = sample[target], sample[f"{target}_mask"]
                valid = (
                    value.reshape(-1)[mask.reshape(-1).bool()]
                    if mask.ndim
                    else (value.reshape(-1) if bool(mask) else value.new_empty(0))
                )
                for cls in valid.tolist():
                    class_dist[target][int(cls)] += 1

    valid_fraction = {
        key: valid_hits[key] / valid_total[key] for key in union_keys if valid_total[key] > 0
    }
    return Report(
        names=names,
        sizes=sizes,
        union_keys=union_keys,
        coverage=coverage,
        valid_fraction=valid_fraction,
        class_distributions={t: dict(c) for t, c in class_dist.items()},
    )


def validate(
    datasets: Sequence[SizedDataset],
    schemas: Sequence[DatasetSchema],
    num_samples: int = 4,
    strict: bool = False,
) -> list[str]:
    """Check that each dataset's raw samples match its declared specs.

    For every declared key it verifies the value's rank and trailing feature
    size against the :class:`~omniloader.schema.spec.TensorSpec` (the sequence
    length is free — unification pads it). Dtype is not checked because
    unification casts it.

    Args:
        datasets: The source datasets.
        schemas: One schema per dataset.
        num_samples: Number of samples to probe per dataset.
        strict: When ``True``, raise :class:`ValueError` if any issue is found.

    Returns:
        A list of human-readable issue strings (empty when everything matches).

    Raises:
        ValueError: If ``strict`` and issues were found.

    """
    issues: list[str] = []
    for index, (dataset, schema) in enumerate(zip(datasets, schemas)):
        name = _dataset_name(dataset, index)
        for i in range(min(len(dataset), num_samples)):  # type: ignore[arg-type]
            sample = dataset[i]
            for spec in schema.specs:
                if spec.name not in sample:
                    issues.append(f"{name}[{i}]: missing declared key {spec.name!r}")
                    continue
                value = sample[spec.name]
                expected_ndim = len(spec.value_shape)
                if value.ndim != expected_ndim:
                    issues.append(
                        f"{name}[{i}].{spec.name}: expected {expected_ndim}D, got {value.ndim}D"
                    )
                elif spec.feature_dim is not None and value.shape[-1] != spec.feature_dim:
                    issues.append(
                        f"{name}[{i}].{spec.name}: expected feature_dim {spec.feature_dim}, "
                        f"got {value.shape[-1]}"
                    )
    if strict and issues:
        raise ValueError("Dataset validation failed:\n" + "\n".join(issues))
    return issues
