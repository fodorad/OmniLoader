"""Declarative schema for features and targets across heterogeneous datasets.

The building blocks here let each dataset declare *what* it provides (features and
targets, their shapes, dtypes and placeholder values) without committing to *how*
those values are stored or *what modality* they represent. Everything is described
in purely structural terms:

* a **vector** value has no sequence axis — shape ``()`` (a scalar) or ``(F,)``;
* a **sequence** value has a leading sequence axis of length ``T`` — shape
  ``(T,)`` or ``(T, F)``.

A spec is a sequence exactly when its ``time_dim`` is set; otherwise it is a
vector. No modality (video/audio/text/…) or model concept appears anywhere.
:class:`UnifiedSchema` merges the per-dataset declarations into the single union
schema that :class:`~omniloader.schema.unify.SampleUnifier` uses to bring every raw
sample into a common, masked format.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

#: Mapping of human-readable dtype names to :class:`torch.dtype` values.
_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "int64": torch.int64,
    "int32": torch.int32,
    "bool": torch.bool,
}


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Resolve a dtype given as a string or :class:`torch.dtype`.

    Args:
        dtype: Either a :class:`torch.dtype` or one of the string keys in
            :data:`_DTYPES` (e.g. ``"float32"``).

    Returns:
        The corresponding :class:`torch.dtype`.

    Raises:
        ValueError: If ``dtype`` is a string that is not a known dtype name.

    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in _DTYPES:
        raise ValueError(f"Unknown dtype {dtype!r}; expected one of {sorted(_DTYPES)}")
    return _DTYPES[dtype]


@dataclass(frozen=True)
class TensorSpec:
    """Structural declaration of a single named value (a feature *or* a target).

    The same spec type describes both inputs and supervision targets; its role is
    determined by whether it is placed in :attr:`DatasetSchema.features` or
    :attr:`DatasetSchema.targets`.

    The shape is derived from two independent, optional axes:

    * ``time_dim`` — length ``T`` of the sequence axis. When ``None`` the value
      is a **vector** (no sequence axis); when set the value is a **sequence**.
    * ``feature_dim`` — size ``F`` of the trailing feature axis. When ``None``
      the value is a **scalar** along that axis.

    So the four representable shapes are ``()``, ``(F,)``, ``(T,)`` and
    ``(T, F)`` — covering scalar labels, feature vectors, scalar sequences
    (e.g. one value per step) and feature sequences.

    Args:
        name: Unique key under which the value appears in a sample dict.
        feature_dim: Size of the trailing feature axis ``F``. ``None`` for a
            scalar along that axis.
        time_dim: Length ``T`` of the sequence axis. ``None`` for a vector.
        dtype: Element dtype, as a :class:`torch.dtype` or string name.
        placeholder: Fill value used when a dataset does not provide this key.
            Defaults to ``0`` (use e.g. ``-1`` as an ignore index for class ids).

    """

    name: str
    feature_dim: int | None = None
    time_dim: int | None = None
    dtype: torch.dtype = torch.float32
    placeholder: float = 0.0

    def __post_init__(self) -> None:
        """Validate and normalise field values."""
        object.__setattr__(self, "dtype", resolve_dtype(self.dtype))
        if self.feature_dim is not None and self.feature_dim <= 0:
            raise ValueError(f"Spec {self.name!r} feature_dim must be positive")
        if self.time_dim is not None and self.time_dim <= 0:
            raise ValueError(f"Spec {self.name!r} time_dim must be positive")

    @property
    def is_sequence(self) -> bool:
        """Whether the value has a sequence axis (``time_dim`` is set)."""
        return self.time_dim is not None

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Shape of the placeholder value tensor for a single sample.

        Returns:
            One of ``()``, ``(F,)``, ``(T,)`` or ``(T, F)`` depending on which
            of ``time_dim`` and ``feature_dim`` are set.

        """
        dims: list[int] = []
        if self.time_dim is not None:
            dims.append(self.time_dim)
        if self.feature_dim is not None:
            dims.append(self.feature_dim)
        return tuple(dims)

    @property
    def mask_shape(self) -> tuple[int, ...]:
        """Shape of the boolean validity mask for a single sample.

        Returns:
            ``(T,)`` for a sequence value (one flag per step) and ``()`` for a
            vector value (one flag for the whole value).

        """
        return (self.time_dim,) if self.time_dim is not None else ()

    def placeholder_value(self) -> torch.Tensor:
        """Build the placeholder value tensor filled with :attr:`placeholder`."""
        return torch.full(self.value_shape, self.placeholder, dtype=self.dtype)

    def placeholder_mask(self) -> torch.Tensor:
        """Build the all-``False`` mask marking every position as invalid."""
        return torch.zeros(self.mask_shape, dtype=torch.bool)


@dataclass
class DatasetSchema:
    """The set of features and targets a single dataset provides.

    A value is a *feature* or a *target* purely by which list it is placed in;
    both use the same :class:`TensorSpec`.

    Args:
        features: Input feature specs the dataset supplies.
        targets: Supervision target specs the dataset supplies.

    """

    features: list[TensorSpec] = field(default_factory=list)
    targets: list[TensorSpec] = field(default_factory=list)

    @property
    def specs(self) -> list[TensorSpec]:
        """All feature and target specs concatenated."""
        return [*self.features, *self.targets]

    @property
    def keys(self) -> set[str]:
        """Names of every feature and target the dataset provides."""
        return {spec.name for spec in self.specs}


def _merge_specs(specs: list[TensorSpec], kind: str) -> None:
    """Validate that specs sharing a name are mutually compatible.

    Args:
        specs: Specs collected across datasets for a single ``kind``.
        kind: Human-readable label (``"feature"`` or ``"target"``) for errors.

    Raises:
        ValueError: If two specs share a name but disagree on shape or dtype.

    """
    seen: dict[str, TensorSpec] = {}
    for spec in specs:
        existing = seen.get(spec.name)
        if existing is None:
            seen[spec.name] = spec
            continue
        if existing.value_shape != spec.value_shape or existing.dtype != spec.dtype:
            raise ValueError(f"Incompatible {kind} spec {spec.name!r}: {existing} vs {spec}")


class UnifiedSchema:
    """Union of several :class:`DatasetSchema` objects into one shared schema.

    The unified schema is the deduplicated union of every feature and target
    across the input datasets. Specs that share a name must agree on shape and
    dtype. Order is preserved: features/targets appear in the order they are
    first encountered.

    Args:
        schemas: Per-dataset schemas to merge.

    """

    def __init__(self, schemas: list[DatasetSchema]) -> None:
        features: list[TensorSpec] = []
        targets: list[TensorSpec] = []
        seen: set[str] = set()
        for schema in schemas:
            for spec in schema.features:
                if spec.name not in seen:
                    features.append(spec)
                    seen.add(spec.name)
            for spec in schema.targets:
                if spec.name not in seen:
                    targets.append(spec)
                    seen.add(spec.name)
        _merge_specs([s for sch in schemas for s in sch.features], "feature")
        _merge_specs([s for sch in schemas for s in sch.targets], "target")
        self.features = features
        self.targets = targets

    @property
    def specs(self) -> list[TensorSpec]:
        """All feature and target specs in the unified schema."""
        return [*self.features, *self.targets]

    @property
    def feature_keys(self) -> list[str]:
        """Names of the input features in schema order."""
        return [spec.name for spec in self.features]

    @property
    def target_keys(self) -> list[str]:
        """Names of the supervision targets in schema order."""
        return [spec.name for spec in self.targets]

    @property
    def keys(self) -> set[str]:
        """Names of every feature and target in the unified schema."""
        return {spec.name for spec in self.specs}

    def spec(self, name: str) -> TensorSpec:
        """Return the spec registered under ``name``.

        Args:
            name: Feature or target name.

        Returns:
            The matching :class:`TensorSpec`.

        Raises:
            KeyError: If no spec with that name exists.

        """
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def __len__(self) -> int:
        """Number of features plus targets in the unified schema."""
        return len(self.features) + len(self.targets)
