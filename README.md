<div align="center">

<img src="https://raw.githubusercontent.com/fodorad/OmniLoader/main/assets/logo/logo-icon_with_name.svg" alt="OmniLoader" width="460">
<br/>
<em>One loader to load them all, one schema to find them,<br/>one batch to bring them all, and in the mask bind them.</em>
<br/><br/>

[![GitHub Release](https://img.shields.io/github/v/release/fodorad/OmniLoader?color=purple)](https://github.com/fodorad/OmniLoader/releases)
[![PyPI](https://img.shields.io/pypi/v/omniloader?color=purple)](https://pypi.org/project/omniloader/)
[![CI](https://github.com/fodorad/OmniLoader/actions/workflows/ci.yml/badge.svg)](https://github.com/fodorad/OmniLoader/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fodorad/OmniLoader/branch/main/graph/badge.svg)](https://codecov.io/gh/fodorad/OmniLoader)
[![Docs](https://github.com/fodorad/OmniLoader/actions/workflows/docs.yml/badge.svg)](https://fodorad.github.io/OmniLoader/)

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](https://github.com/fodorad/OmniLoader/blob/main/LICENSE)

</div>

**A PyTorch meta data loader that unifies disjoint, multi-task datasets so one
model can be trained jointly across all of them.**

**The problem.** Multi-task learning wants one model to learn many related tasks at
once, but the supervision for those tasks is usually scattered across **separate
datasets that annotate different things**: one corpus has a **single label per sample**
(e.g. sentiment), another has a **per-step label sequence** (e.g. valence/arousal), a
third has **categorical class ids** — and each covers only its own subset of the
features. There is no single dataset that carries every task's labels, so a
shared-backbone model cannot simply be pointed at one file. The samples first have to be
brought to **one common, batchable scheme** where every task's slot is always present,
and where the loss can tell a *real* label apart from a *missing* one.

**What OmniLoader does.** It builds the *union* of every feature and target across your
datasets and yields each sample in that shared format:

- keys a dataset **provides** are copied and, for sequences, padded/cropped to a declared length;
- keys a dataset **lacks** are filled with a **placeholder tensor** plus an
  all-`False` **`<name>_mask`**;

so a single batch can freely mix samples from different datasets, each task's head sees a
consistently-shaped tensor, and your loss and model always know which values are real
(train on the masked-in positions, ignore the rest). It's a map-style
`torch.utils.data.Dataset` you wrap in an ordinary `DataLoader` — it unifies and masks,
it does not reinvent batching.

OmniLoader is **modality-, dataset- and model-agnostic** — it knows nothing about video,
audio, text, which specific corpora you loaded, or any model. Everything is described
structurally as **vectors** (shape `()` or `(F,)`) and **sequences** (shape `(T,)` or
`(T, F)`); a value is a sequence exactly when its spec sets `time_dim`.

Beyond this core unification and its utilities (schema declaration, dataset adapters,
splits, introspection), OmniLoader also ships the surrounding machinery a joint training
run needs: **mixing strategies** to balance datasets of very different sizes,
**subsampling** to control within-dataset draws, **class-balance** calculations (sampler
weights and loss weights), and a mask-aware transform pipeline of **normalization** and
**augmentation** techniques — all configurable end-to-end from a single file.

---

## Feature overview

| Category | Feature | What it does |
|---|---|---|
| **Core** | `OmniLoader` | Concatenates disjoint datasets into one masked, unified stream |
| | `SampleUnifier` | Maps a raw sample onto the union schema; fills gaps with placeholder + `<name>_mask` |
| | `unified_collate` | Stacks tensors, lists metadata |
| **Schema** | `TensorSpec` | Declare a value (feature or target): `feature_dim`, `time_dim`, dtype, placeholder |
| | `DatasetSchema` / `UnifiedSchema` | Per-dataset specs; merged + validated union |
| | vector / sequence | Structural, modality-agnostic (`()`, `(F,)`, `(T,)`, `(T,F)`) |
| **Datasets** | `HDF5Dataset` | Per-sample HDF5 groups; worker-safe; `cache_size`/`preload` |
| | `NpyFolderDataset` | Memory-mapped `root/<sample>/<key>.npy` layout |
| | `DictTensorDataset` | In-memory random tensors (tests/experiments) |
| | `split_indices` | Reproducible, optionally stratified train/val/test splits |
| **Mixing** | `ProportionalStrategy` | Sample by true dataset sizes |
| | `TemperatureStrategy` | `size**(1/T)` re-weighting (T=2 → sqrt) |
| | `AnnealedTemperatureStrategy` | Temperature annealed per epoch |
| | `FixedWeightStrategy` | Explicit or uniform per-dataset weights |
| | `RoundRobinStrategy` | Equal, interleaved draws |
| **Subsampling** | `SubsampleConfig` / `IndexPool` | Replacement, FRESH/EXHAUST, `effective_size`, per-sample weights |
| | `class_weights_for_sampler` | Per-sample inverse-frequency weights for the **sampler** |
| **Class balance** | `class_weights_for_loss` / `class_histogram` | Per-class **loss** weights + exact counts for `CrossEntropyLoss(weight=)` (persist via CLI) |
| **Batching** | `DynamicCollator` | Pad sequences to per-batch max, not fixed length (native-length mode only) |
| | `LengthBucketBatchSampler` | Group similar-length sequences to cut padding (needs `pad_features=False`) |
| **Normalization** | `Normalize` / `MinMax` / `Robust` / `Instance` | Standardize features (from stats or per-sample) |
| | `PerDatasetNormalize` | Standardize each sample by **its source dataset's** stats, with an inference fallback for unseen sources |
| | `compute_stats` / `compute_dataset_stats` | Pooled or per-dataset mean/std/min/max/median/iqr |
| | `save_stats` / `load_stats` | JSON persistence (flat or per-dataset) |
| **Augmentation** | `GaussianNoise` | Feature corruption |
| | `FeatureDropout` | Drop whole feature streams (modality dropout) |
| | `SpanMasking` / `FeatureMasking` | Zero time spans / feature bands |
| | `TimeWarp` | Random speed perturbation |
| | `RandomCrop` / `CenterCrop` | Windowed sequence crop (features + targets aligned) |
| | `MixupCollator` | MixUp/CutMix at collate (emits `mixup_lambda` + paired targets) |
| | `Compose` | Chain transforms (mask-aware, seedable, train/eval-gated) |
| **Distributed** | `OmniSampler(num_replicas, rank)` | DDP index sharding |
| | `set_epoch` + `seed_worker` | Reproducible, worker-count-independent augmentation |
| **Introspection** | `describe()` | Coverage matrix, valid fractions, class distributions |
| | `validate()` | Dry-run check of data vs declared specs |
| **Config & CLI** | `OmniConfig` | One JSON/YAML file: seed, strategy, subsample, transforms, collate, bucketing, DDP & DataLoader knobs, datasets |
| | `config.build_dataloader()` | Assemble a ready `DataLoader` end-to-end from the config alone |
| | `build_datasets` | Construct datasets from a declarative `datasets` section |
| | `omniloader` CLI | `describe` / `validate` / `compute-stats` / `class-weights-for-loss` |
| **Integration** | `OmniDataModule` | Optional Lightning module (wires strategy, DDP, seeding) |

---

## Installation

OmniLoader is on **PyPI**, needs **Python 3.12+**, and installs cleanly with
[uv](https://docs.astral.sh/uv/) (recommended) or plain `pip`. Core deps are just
`torch`, `numpy`, `h5py` and `pyyaml`.

```bash
# with uv (recommended)
uv add omniloader
uv add "omniloader[lightning]"   # + the optional PyTorch Lightning DataModule

# or with pip
pip install omniloader
pip install "omniloader[lightning]"
```

### Development

```bash
git clone https://github.com/fodorad/OmniLoader && cd OmniLoader
make dev     # uv editable install with all extras + dev + docs tooling
make check   # ruff + ty + tests(+coverage) + docs build (mirrors CI)
```

---

## Quickstart

```python
import torch
from torch.utils.data import DataLoader
from omniloader import (
    DatasetSchema, DictTensorDataset, TensorSpec,
    OmniLoader, TemperatureStrategy, unified_collate,
)

# Dataset A — a per-step (sequence) target over a length-16 feature sequence.
ds_a = DictTensorDataset({"video": torch.randn(40, 16, 32), "valence": torch.randn(40, 16)})
schema_a = DatasetSchema(
    features=[TensorSpec("video", feature_dim=32, time_dim=16)],   # sequence (T, F)
    targets=[TensorSpec("valence", time_dim=16, placeholder=-5.0)], # sequence (T,)
)

# Dataset B — a single scalar target per sample, a different feature sequence.
ds_b = DictTensorDataset({"audio": torch.randn(200, 24, 8), "sentiment": torch.randn(200)})
schema_b = DatasetSchema(
    features=[TensorSpec("audio", feature_dim=8, time_dim=24)],  # sequence (T, F)
    targets=[TensorSpec("sentiment", placeholder=-5.0)],         # vector scalar ()
)

omni = OmniLoader([ds_a, ds_b], [schema_a, schema_b])
sampler = omni.make_sampler(TemperatureStrategy(omni.dataset_sizes, temperature=2.0))
loader = DataLoader(omni, batch_size=8, sampler=sampler, collate_fn=unified_collate)

batch = next(iter(loader))
# Every batch carries the union schema: video, audio, valence, sentiment + masks.
assert batch["valence"].shape == (8, 16)
assert batch["valence_mask"].shape == (8, 16)   # False for samples from dataset B
assert batch["sentiment"].shape == (8,)
assert batch["sentiment_mask"].shape == (8,)    # False for samples from dataset A
```

> **Single dataset?** OmniLoader works with one dataset too — pass one-element lists.
> The union is then just that dataset's schema, and the multi-dataset machinery (mixing
> strategies, per-dataset normalization) simply lies dormant; you still get masked,
> fixed-shape batches, config-driven loading, transforms and splits.
>
> ```python
> omni = OmniLoader([ds_a], [schema_a])
> loader = DataLoader(omni, batch_size=8, collate_fn=unified_collate)
> ```

---

## Declaring schemas

Each value — feature or target — is described by a single `TensorSpec` (its role
is set by whether it goes in `features=` or `targets=`):

| Field | Meaning |
|-------|---------|
| `name` | key in the sample dict |
| `feature_dim` | trailing feature size `F`, or `None` for a scalar along that axis |
| `time_dim` | sequence length `T`; **set it to make the value a sequence** (padded/cropped to this). `None` → vector |
| `dtype` | `torch.float32`, `torch.int64`, … (class ids use an int dtype) |
| `placeholder` | fill value when a dataset lacks the key (e.g. `-1` ignore-index for classes) |

The four representable shapes are `()`, `(F,)`, `(T,)` and `(T, F)`. The mask
matches the sequence axis: `(T,)` for sequences, scalar `()` for vectors.

---

## Usage

Each subsection below states the problem it solves, then shows the minimal call.

### Unifying disjoint datasets (`OmniLoader`, `SampleUnifier`)

**Problem:** your datasets annotate *different* things (one has `valence`, another
`sentiment`) and have *different* sequence lengths — a model can't consume that
directly. `OmniLoader` builds the **union** of all schemas and yields every sample in
one fixed, masked layout: keys a dataset provides are copied (sequences padded/cropped
to their declared length), keys it lacks are filled with a placeholder and an all-`False`
`<name>_mask`. `SampleUnifier` is the per-sample engine that does this mapping — usually
you let `OmniLoader` drive it, but you can call it directly to see the shape:

```python
from omniloader import SampleUnifier, UnifiedSchema
import torch

schema = UnifiedSchema([schema_a, schema_b])   # merge per-dataset schemas into the union
unify = SampleUnifier(schema)                  # pads/crops sequences + fills missing keys

unified = unify({"video": torch.randn(16, 32)})  # a raw dict that only has 'video'
assert set(unified) >= {"video", "audio", "valence", "sentiment"}  # every union key present
assert not unified["sentiment_mask"]             # absent key -> all-False mask (ignore in loss)
```

`OmniLoader([ds_a, ds_b], [schema_a, schema_b])` applies this across datasets, so a plain
`DataLoader` sees one consistent, batchable stream (see the [Quickstart](#quickstart)).

### Loading features from disk (`HDF5Dataset`)

**Problem:** real feature sets are too big for RAM and must be read lazily, safely from
DataLoader workers. `HDF5Dataset` reads per-sample groups (`file[subset][sample_id][key]`)
straight from disk, with an optional LRU cache and per-process file handles:

```python
from omniloader import HDF5Dataset, DatasetSchema, TensorSpec, OmniLoader

ds = HDF5Dataset("data/mosei.h5", subset="train", cache_size=256)  # worker-safe, lazy
schema = DatasetSchema(
    features=[TensorSpec("video", feature_dim=1024, time_dim=300)],
    targets=[TensorSpec("sentiment", placeholder=-5.0)],
)
omni = OmniLoader([ds], [schema])
```

`NpyFolderDataset` (memory-mapped `root/<sample>/<key>.npy`) is a drop-in alternative.

### Balancing datasets of different sizes

Sampling by raw size lets the largest dataset drown the rest; sampling perfectly
uniformly overfits the tiny ones. The **size-aware, robust default is
`TemperatureStrategy`** — it re-weights each dataset by `size ** (1 / T)`, so `T=2`
gives square-root scaling (a strong, widely-used default), `T=1` is proportional, and
large `T` approaches uniform:

```python
from omniloader import TemperatureStrategy

sampler = omni.make_sampler(TemperatureStrategy(omni.dataset_sizes, temperature=2.0))
```

Two more when you want explicit control rather than a size-derived rule:

```python
from omniloader import FixedWeightStrategy, RoundRobinStrategy

# Explicit mixture — draw dataset A 3x as often as B, whatever their sizes.
sampler = omni.make_sampler(FixedWeightStrategy(omni.dataset_sizes, weights=[3.0, 1.0]))

# Equal contribution, interleaved so consecutive samples come from different datasets.
sampler = omni.make_sampler(RoundRobinStrategy(omni.dataset_sizes))
```

The full set: `ProportionalStrategy` (by true size), `TemperatureStrategy` (sqrt/soft,
the default), `AnnealedTemperatureStrategy` (anneal `T` per epoch — start soft, sharpen
later), `RoundRobinStrategy` (equal interleaved) and `FixedWeightStrategy` (explicit
manual weights).

**Subsampling** then controls how each dataset is drawn *within* an epoch, independent of
the mixing weights (a `SubsampleConfig` per dataset):

- `policy=EXHAUST` walks the whole dataset before any reuse (vs `FRESH`, a fresh random
  subsample each epoch);
- `effective_size=N` caps a dataset's per-epoch contribution;
- `sample_weights` skews within-dataset draws — e.g. `class_weights_for_sampler(ds, "label")`
  flattens a class histogram.

```python
from omniloader import (
    TemperatureStrategy, SubsampleConfig, ExhaustionPolicy, class_weights_for_sampler,
)

sampler = omni.make_sampler(
    TemperatureStrategy(
        omni.dataset_sizes, temperature=2.0,                 # size-aware balancing
        subsample=[
            SubsampleConfig(policy=ExhaustionPolicy.EXHAUST),                   # A: full coverage
            SubsampleConfig(sample_weights=class_weights_for_sampler(ds_b, "label")),  # B: class-balanced
        ],
    )
)
```

### Handling class imbalance — resample *or* reweight the loss

**Problem:** a skewed label distribution lets the majority class dominate. Two standard
remedies, and OmniLoader gives a data-side helper for each. Both are inverse-frequency by
default — the difference is *what* they weight:

| Helper | Granularity | Fed to | Effect |
|--------|-------------|--------|--------|
| `class_weights_for_sampler(ds, "label")` | **per sample** (`1 / count`) | the *sampler* via `SubsampleConfig(sample_weights=…)` | changes *how often* a sample is drawn (**resampling**) |
| `class_weights_for_loss(omni, "emotion")` | **per class** `(num_classes,)` | the *loss* via `CrossEntropyLoss(weight=…)` | changes *how much* each mistake costs (**reweighting**); every sample is still seen once |

So `class_weights_for_loss` **is** inverse frequency too — it's the per-class, loss-side counterpart
of the per-sample sampler weights (resampling shown earlier under *Subsampling*). Its
default `scheme="inverse"` is `1 / count` normalized to average `1`; `scheme="effective"`
uses the *effective number of samples* reweighting (`(1 − β) / (1 − βⁿ)`, Cui et al. 2019),
which tames pure inverse frequency on long-tailed data where a few classes have huge
counts. Inverse- / median-frequency loss weighting is the textbook fix for imbalanced
classification and segmentation; the effective-number scheme is its long-tail refinement.

Compute once and persist, then reuse without recomputation:

```bash
omniloader class-weights-for-loss config.yaml --target emotion -o class_weights_for_loss.json
```

```python
import json, torch
from omniloader import class_weights_for_loss, class_histogram

counts = class_histogram(omni, "emotion")              # exact per-class counts (int64)
w = class_weights_for_loss(omni, "emotion", scheme="inverse")   # (num_classes,), avg 1, absent class -> 0
loss = torch.nn.CrossEntropyLoss(weight=w)

# later runs: load the saved vector instead of recomputing
w = torch.tensor(json.load(open("class_weights_for_loss.json"))["weights"])
```

Both helpers count **only valid (unmasked) positions**; the loss weights count **every
valid labelled step** of a framewise target (matching what a per-step loss sees), while
the per-sample sampler weights use one representative class per sample.

### Normalization

Transforms run per sample after unification: mask-aware, reproducible from the
loader's `seed`. Augmentations self-skip during evaluation.

```python
from omniloader import Normalize, GaussianNoise, Compose, compute_stats

stats = compute_stats(omni, keys=["video", "audio"])   # over valid steps
transform = Compose([Normalize(stats), GaussianNoise(std=0.1, p=0.5, schema=omni.schema)])
omni = OmniLoader([ds_a, ds_b], [schema_a, schema_b], transform=transform, seed=0)
```

Stats are computed **per feature channel over valid (unmasked) positions only** —
padding and placeholders never contribute. Compute them **once on the train split** and
persist to JSON, then reuse them every run — no recomputation:

```bash
omniloader compute-stats config.yaml -o stats.json            # pooled
omniloader compute-stats config.yaml -o ds_stats.json --per-dataset
```

```python
from omniloader import compute_stats, save_stats, load_stats

save_stats(compute_stats(omni, keys=["video", "audio"]), "stats.json")  # persist once
stats = load_stats("stats.json")                                        # reuse later
```

The JSON lives wherever you point it (track it with **DVC** for reproducibility). In a
config, reference it with **`stats_path`** (`{name: normalize, stats_path: stats.json}`)
or **`union_stats_path`** — so normalization is fully config-driven and never recomputed.

#### Normalizing across datasets — pooled vs per-dataset

When you mix several corpora, you can standardize by **pooled (union)** stats or by
**each dataset's own** stats:

- **Pooled** (`compute_stats(omni, …)` → `Normalize`) — one `(mean, std)` per feature
  across all datasets. Simplest; one stat set for inference; keeps genuine cross-dataset
  magnitude differences. Weak when corpora have real domain shift.
- **Per-dataset** (`compute_dataset_stats(omni, …)` → `PerDatasetNormalize`) — each
  dataset centered by its own stats (CMVN / AdaBN / Domain-Specific-BN practice). Removes
  cross-corpus domain shift and dataset-identity leakage. **Its catch is inference:**
  per-dataset stats are undefined for a *new* source, so `PerDatasetNormalize` takes a
  `fallback` for unseen dataset ids — never a guessed dataset's stats.

```python
from omniloader import compute_dataset_stats, PerDatasetNormalize, Compose

# Per-dataset train stats, keyed by the `dataset` metadata every sample carries.
ds_stats = compute_dataset_stats(omni, keys=["video", "audio"])
transform = Compose([
    PerDatasetNormalize(ds_stats, fallback="instance"),  # unseen source → per-sample norm
])
```

**Choosing:** benchmarking within known corpora → per-dataset is best (source id is
always known). Deploying on unseen sources → prefer `fallback="instance"` (per-sample,
identity-free) or estimate the new source's stats offline (AdaBN-style); keep pooled
stats as a `"union"` fallback if you want a neutral default. For already-normalized
pretrained features (DINOv2/WavLM) the biggest win is the *cross-dataset* alignment,
not absolute scale.

### Augmentation — robustness to a missing modality (`FeatureDropout`)

**Problem:** at inference a modality may be absent (no audio track, a dropped camera),
but a model trained on always-complete inputs leans on all of them and degrades. Modality
/ stream **dropout** simulates this during training: with probability `p` a whole feature
stream is zeroed *and* its mask set all-`False`, so the model genuinely learns to cope
without it. Like every augmentation here it is mask-aware, reproducible from the loader's
`seed`, and self-skips during evaluation:

```python
from omniloader import FeatureDropout, OmniLoader

# Drop 'video' or 'audio' 20% of the time each (never both — keep_at_least_one).
transform = FeatureDropout(keys=["video", "audio"], p=0.2, schema=omni.schema)
omni = OmniLoader([ds_a, ds_b], [schema_a, schema_b], transform=transform)
```

Compose it with normalization and other augmenters (`GaussianNoise`, `SpanMasking`,
`FeatureMasking`, `TimeWarp`, `RandomCrop`) via `Compose([...])`.

### Variable-length batching & multi-GPU

This is an **opt-in efficiency path** — it only applies with `pad_features=False`.
By default OmniLoader pads/crops every sequence to its declared `time_dim`, so every
batch is already one length and `DynamicCollator`/`LengthBucketBatchSampler` do
nothing. The `<name>_mask` handles *correctness* (padded steps never corrupt the
output or loss), but a padded step still costs compute and memory (attention is
`O(T²)`). So masks and bucketing are orthogonal: **masks keep you correct, bucketing
keeps you cheap.** When sequence lengths vary a lot, keep native lengths and let each
batch pad only to its own longest sample, grouping similar lengths to shrink that max:

```python
from torch.utils.data import DataLoader
from omniloader import (
    OmniSampler, ProportionalStrategy, DynamicCollator,
    LengthBucketBatchSampler, seed_worker,
)

omni = OmniLoader([ds_a, ds_b], [schema_a, schema_b], pad_features=False)  # native lengths
# num_replicas/rank are inferred from torch.distributed when omitted.
sampler = OmniSampler(ProportionalStrategy(omni.dataset_sizes))
batches = LengthBucketBatchSampler(sampler, omni.sequence_lengths("video"), batch_size=8)
loader = DataLoader(
    omni, batch_sampler=batches,
    collate_fn=DynamicCollator(omni.schema), worker_init_fn=seed_worker,
)
```

### Introspection

```python
print(omni.describe())          # coverage matrix, valid fractions, class distributions
issues = omni.validate()        # [] when every dataset matches its declared specs
```

### Config, CLI & Lightning (`OmniConfig`)

**Problem:** wiring datasets, mixing, transforms, collate, bucketing and DDP by hand is
verbose and hard to reproduce. `OmniConfig` moves the whole run into **one JSON/YAML
file** — seed, batch size, DataLoader knobs (`num_workers`, `pin_memory`,
`persistent_workers`, `prefetch_factor`), the mixing `strategy`, per-dataset `subsample`,
the `transforms` pipeline, the `collate` function, length `bucketing`, distributed
`num_replicas`/`rank`, and the `datasets` themselves. `config.build_dataloader()`
assembles a ready-to-iterate `DataLoader` from it, so experiments are reproducible and
diffable purely from config:

> **Every option in one place.** OmniLoader ships an annotated template listing
> every key (with defaults, notes and which section is optional) — copy it as a
> starting point:
>
> ```python
> from omniloader import config_template_path
> print(config_template_path())   # omniloader/templates/config_template.yaml
> ```

```yaml
# config.yaml — a complete experiment description
seed: 0
batch_size: 32
num_workers: 8
pin_memory: true
persistent_workers: true
pad_features: false            # keep native sequence lengths for dynamic padding
strategy: temperature
strategy_kwargs: {temperature: 2.0}
subsample:
  - null                        # dataset A: untouched
  - {policy: exhaust, effective_size: 5000}
transforms:
  - {name: normalize, stats_path: stats.json}   # or inline: stats: {video: {mean: .., std: ..}}
  - {name: gaussian_noise, keys: [video], std: 0.1, p: 0.5}
collate: mixup                  # unified | dynamic | mixup
collate_kwargs: {base: dynamic, alpha: 0.2, mode: cutmix}
bucketing: {key: video, bucket_multiplier: 50}
datasets:
  - adapter: hdf5
    args: {h5_path: data/mosei.h5, subset: train}
    schema:
      features: [{name: video, feature_dim: 1024, time_dim: 300}]
      targets:  [{name: sentiment, placeholder: -5.0}]
```

```python
from omniloader import OmniConfig

config = OmniConfig.from_file("config.yaml")

# Everything in one call — datasets, strategy, sampler, transforms, collate, bucketing:
train_loader = config.build_dataloader(training=True)
val_loader = config.build_dataloader(training=False)   # sequential, augmentations off

# …or keep the building blocks for manual wiring:
datasets, schemas = config.build_datasets()
strategy = config.build_strategy([len(d) for d in datasets])
transform = config.build_transform(config.build_loader(datasets, schemas).schema)
```

Datasets declared in the config's `datasets` section can be inspected from the shell:

```bash
omniloader describe config.json
omniloader validate config.json
omniloader compute-stats config.json -o stats.json
```

With the `lightning` extra, `OmniDataModule` wires the strategy, distributed
sharding, per-split transforms and seeding from a single config:

```python
from omniloader.integrations.lightning import OmniDataModule

dm = OmniDataModule(config, train=([ds_a, ds_b], [schema_a, schema_b]))
```

> **DDP note.** `OmniSampler` is already distributed-aware (it shards by
> `num_replicas`/`rank`). When training under Lightning DDP, pass
> `Trainer(use_distributed_sampler=False)` so Lightning does not wrap it in a
> second `DistributedSampler`.

Because a run is fully described by its config, OmniLoader slots cleanly into
experiment tooling: `config.to_dict()` is JSON-serialisable for **MLflow**
`log_params` / **DVC** `params.yaml`, and `save_stats`/`save_split_info` emit
JSON artifacts you can version with **DVC** for reproducible normalization and
splits.

---

## Scope

OmniLoader is a **pure data-loading library**. Model- and loss-side concerns
(e.g. multi-task loss balancing) live in your training code — OmniLoader exposes
per-sample `dataset` metadata in every batch so they are easy to wire up there.

---

## License

[MIT](https://github.com/fodorad/OmniLoader/blob/main/LICENSE) © fodorad

---

## Contact

Adam Fodor — [fodorad201@gmail.com](mailto:fodorad201@gmail.com) · [adamfodor.com](https://adamfodor.com)
