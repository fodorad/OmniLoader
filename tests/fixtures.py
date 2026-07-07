"""Shared synthetic fixtures for the OmniLoader test-suite.

Everything is built from random tensors (and a tiny temporary HDF5 file) so the
tests are fast and depend on no private datasets. Modality names (``valence``,
``sentiment``, ``emotion``, ``video``, ``audio``) are used purely to *simulate*
realistic multi-task setups; the library itself never assumes them.
"""

from pathlib import Path

import h5py
import numpy as np
import torch

from omniloader import (
    DatasetSchema,
    DictTensorDataset,
    TensorSpec,
)

# Sequence lengths (T) and feature widths (F) for the simulated modalities.
VIDEO_T = 16
VIDEO_F = 32
AUDIO_T = 24
AUDIO_F = 8


def valence_dataset(n: int = 40) -> tuple[DictTensorDataset, DatasetSchema]:
    """A sequence-target dataset: per-step valence over a video feature sequence."""
    tensors = {
        "video": torch.randn(n, VIDEO_T, VIDEO_F),  # (N, T, F)
        "valence": torch.randn(n, VIDEO_T),  # (N, T)
    }
    schema = DatasetSchema(
        features=[TensorSpec("video", feature_dim=VIDEO_F, time_dim=VIDEO_T)],
        targets=[TensorSpec("valence", time_dim=VIDEO_T, placeholder=-5.0)],
    )
    return DictTensorDataset(tensors), schema


def sentiment_dataset(n: int = 200) -> tuple[DictTensorDataset, DatasetSchema]:
    """A vector-target dataset: one sentiment scalar per sample, audio feature."""
    tensors = {
        "audio": torch.randn(n, AUDIO_T, AUDIO_F),  # (N, T, F)
        "sentiment": torch.randn(n),  # (N,)
    }
    schema = DatasetSchema(
        features=[TensorSpec("audio", feature_dim=AUDIO_F, time_dim=AUDIO_T)],
        targets=[TensorSpec("sentiment", placeholder=-5.0)],
    )
    return DictTensorDataset(tensors), schema


def emotion_dataset(n: int = 12) -> tuple[DictTensorDataset, DatasetSchema]:
    """A scalar class-id dataset sharing the video feature."""
    tensors = {
        "video": torch.randn(n, VIDEO_T, VIDEO_F),  # (N, T, F)
        "emotion": torch.randint(0, 7, (n,)),  # (N,)
    }
    schema = DatasetSchema(
        features=[TensorSpec("video", feature_dim=VIDEO_F, time_dim=VIDEO_T)],
        targets=[TensorSpec("emotion", dtype=torch.int64, placeholder=-1)],
    )
    return DictTensorDataset(tensors), schema


class VarLenDataset:
    """A dataset whose sequences have different native lengths per sample.

    Used to exercise dynamic padding and length bucketing. Sample ``i`` has a
    ``feat`` sequence of length ``lengths[i]`` and a scalar ``label``.
    """

    def __init__(self, lengths: list[int], feature_dim: int = 4) -> None:
        self.lengths = lengths
        self.feature_dim = feature_dim

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> dict:
        t = self.lengths[index]
        return {"feat": torch.randn(t, self.feature_dim), "label": torch.tensor(float(index))}


def varlen_dataset(lengths: list[int], time_dim: int = 32) -> tuple[VarLenDataset, DatasetSchema]:
    """A variable-length dataset + schema (sequence feature + scalar target)."""
    ds = VarLenDataset(lengths)
    schema = DatasetSchema(
        features=[TensorSpec("feat", feature_dim=ds.feature_dim, time_dim=time_dim)],
        targets=[TensorSpec("label")],
    )
    return ds, schema


def write_hdf5(path: Path, subset: str = "train", n: int = 5) -> None:
    """Write a tiny HDF5 file with per-sample groups for reader tests."""
    with h5py.File(path, "w") as f:
        group = f.create_group(subset)
        for i in range(n):
            sg = group.create_group(f"sample_{i:03d}")
            sg.create_dataset("video", data=np.random.randn(VIDEO_T, VIDEO_F).astype("float32"))
            sg.create_dataset("valence", data=np.random.randn(VIDEO_T).astype("float32"))
            sg.create_dataset("sentiment", data=np.float32(np.random.randn()))
            sg.create_dataset("emotion", data=np.int64(i % 3))  # categorical target (0-2)
            sg.create_dataset("dataset", data="synthetic")
            sg.create_dataset("key", data=f"clip_{i}")
