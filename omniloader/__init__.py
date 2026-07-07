"""OmniLoader: a PyTorch meta data loader for disjoint, multi-task datasets.

OmniLoader unifies several datasets with heterogeneous annotations into a single
masked sample scheme so a model can be trained jointly across all of them. It is
modality- and model-agnostic: it reasons only about vectors and sequences. See
:class:`~omniloader.loader.OmniLoader` for the entry point.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from omniloader.collate import DynamicCollator, unified_collate
from omniloader.config import OmniConfig, seed_everything
from omniloader.data.datasets import DictTensorDataset, HDF5Dataset
from omniloader.data.factory import build_datasets
from omniloader.data.npy import NpyFolderDataset
from omniloader.data.splits import load_split_info, save_split_info, split_indices
from omniloader.introspection import Report, describe, validate
from omniloader.loader import OmniLoader
from omniloader.sampling.bucketing import LengthBucketBatchSampler
from omniloader.sampling.sampler import OmniSampler
from omniloader.sampling.strategies import (
    AnnealedTemperatureStrategy,
    FixedWeightStrategy,
    MixingStrategy,
    ProportionalStrategy,
    RoundRobinStrategy,
    SubsampleConfig,
    TemperatureStrategy,
)
from omniloader.sampling.subsamplers import ExhaustionPolicy, IndexPool
from omniloader.sampling.weights import (
    class_histogram,
    class_weights_for_loss,
    class_weights_for_sampler,
)
from omniloader.schema.spec import (
    DatasetSchema,
    TensorSpec,
    UnifiedSchema,
)
from omniloader.schema.unify import SampleUnifier
from omniloader.templates import config_template_path
from omniloader.transforms import (
    CenterCrop,
    Compose,
    FeatureDropout,
    FeatureMasking,
    GaussianNoise,
    InstanceNormalize,
    MinMaxNormalize,
    MixupCollator,
    Normalize,
    PerDatasetNormalize,
    RandomCrop,
    RobustNormalize,
    SpanMasking,
    TimeWarp,
    Transform,
    build_transform,
    compute_dataset_stats,
    compute_stats,
    load_stats,
    save_stats,
)
from omniloader.transforms.stats import compute_feature_stats
from omniloader.utils.padding import (
    fill_gaps_with_repeat,
    pad_or_crop_time_dim,
    repeat_pad_time_dim,
)
from omniloader.utils.seeding import seed_worker

try:
    __version__ = version("omniloader")
except PackageNotFoundError:  # pragma: no cover - only during local, uninstalled use
    __version__ = "1.0.0"

__all__ = [
    "AnnealedTemperatureStrategy",
    "CenterCrop",
    "Compose",
    "DatasetSchema",
    "DictTensorDataset",
    "DynamicCollator",
    "ExhaustionPolicy",
    "FeatureDropout",
    "FeatureMasking",
    "FixedWeightStrategy",
    "GaussianNoise",
    "HDF5Dataset",
    "IndexPool",
    "InstanceNormalize",
    "LengthBucketBatchSampler",
    "MinMaxNormalize",
    "MixingStrategy",
    "MixupCollator",
    "Normalize",
    "NpyFolderDataset",
    "OmniConfig",
    "OmniLoader",
    "OmniSampler",
    "PerDatasetNormalize",
    "ProportionalStrategy",
    "RandomCrop",
    "Report",
    "RobustNormalize",
    "RoundRobinStrategy",
    "SampleUnifier",
    "SpanMasking",
    "SubsampleConfig",
    "TemperatureStrategy",
    "TensorSpec",
    "TimeWarp",
    "Transform",
    "UnifiedSchema",
    "__version__",
    "build_datasets",
    "build_transform",
    "class_histogram",
    "class_weights_for_loss",
    "class_weights_for_sampler",
    "compute_dataset_stats",
    "compute_feature_stats",
    "compute_stats",
    "config_template_path",
    "describe",
    "fill_gaps_with_repeat",
    "load_split_info",
    "load_stats",
    "pad_or_crop_time_dim",
    "repeat_pad_time_dim",
    "save_split_info",
    "save_stats",
    "seed_everything",
    "seed_worker",
    "split_indices",
    "unified_collate",
    "validate",
]
