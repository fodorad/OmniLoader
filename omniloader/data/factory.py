"""Build datasets and schemas from declarative config entries.

This lets datasets be described in JSON/YAML (adapter + constructor args +
schema) so tools like the CLI and :class:`~omniloader.config.OmniConfig` can
assemble a dataset mix without Python glue. Only the registered *data adapters*
are ever constructed — never models.

A dataset entry looks like::

    {
      "adapter": "hdf5",
      "args": {"h5_path": "data/mosei.h5", "subset": "train"},
      "schema": {
        "features": [{"name": "video", "feature_dim": 1024, "time_dim": 300}],
        "targets": [{"name": "sentiment", "placeholder": -5.0}]
      }
    }
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omniloader.data.datasets import HDF5Dataset
from omniloader.data.npy import NpyFolderDataset
from omniloader.schema.spec import DatasetSchema, TensorSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from omniloader.loader import SizedDataset

#: Registry mapping adapter names to dataset classes.
ADAPTERS: dict[str, type] = {
    "hdf5": HDF5Dataset,
    "npy": NpyFolderDataset,
}


def schema_from_dict(data: Mapping[str, Any]) -> DatasetSchema:
    """Build a :class:`DatasetSchema` from a ``{"features": [...], "targets": [...]}`` dict.

    Each spec dict is forwarded to :class:`TensorSpec`/:class:`TensorSpec`; the
    ``dtype`` field may be a string name (e.g. ``"int64"``).

    Args:
        data: The schema mapping.

    Returns:
        The constructed schema.

    """
    return DatasetSchema(
        features=[TensorSpec(**spec) for spec in data.get("features", [])],
        targets=[TensorSpec(**spec) for spec in data.get("targets", [])],
    )


def build_dataset(entry: Mapping[str, Any]) -> tuple[SizedDataset, DatasetSchema]:
    """Construct one ``(dataset, schema)`` pair from a config entry.

    Args:
        entry: A dict with ``adapter`` (a key of :data:`ADAPTERS`), ``args`` (the
            adapter constructor kwargs) and ``schema``.

    Returns:
        The dataset instance and its schema.

    Raises:
        ValueError: If ``adapter`` is not registered.

    """
    adapter = entry["adapter"]
    if adapter not in ADAPTERS:
        raise ValueError(f"Unknown dataset adapter {adapter!r}; expected one of {sorted(ADAPTERS)}")
    dataset = ADAPTERS[adapter](**entry.get("args", {}))
    return dataset, schema_from_dict(entry["schema"])


def build_datasets(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[list[SizedDataset], list[DatasetSchema]]:
    """Construct parallel lists of datasets and schemas from config entries.

    Args:
        entries: A list of dataset entries (see :func:`build_dataset`).

    Returns:
        A ``(datasets, schemas)`` tuple ready for
        :class:`~omniloader.loader.OmniLoader`.

    """
    datasets: list[SizedDataset] = []
    schemas: list[DatasetSchema] = []
    for entry in entries:
        dataset, schema = build_dataset(entry)
        datasets.append(dataset)
        schemas.append(schema)
    return datasets, schemas
