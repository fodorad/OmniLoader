# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are managed automatically with
[release-please](https://github.com/googleapis/release-please) from
[Conventional Commits](https://www.conventionalcommits.org/). Future entries will
list the changes relative to the previous release.

## 1.0.0 (2026-07-07)


### Features

* initial public release of OmniLoader ([5814077](https://github.com/fodorad/OmniLoader/commit/58140772992fb70a77865b4bacf5756cbd18da10))

## 1.0.0 (2026-07-05)

Initial public release. OmniLoader is a modality- and model-agnostic PyTorch meta
data loader that unifies datasets with disjoint annotations into one masked sample
scheme for joint multi-task training.

### Unified sample scheme

- **`OmniLoader`** concatenates any number of source datasets and presents them as
  one dataset whose every sample follows a shared `UnifiedSchema`. Keys a source
  lacks are filled with a placeholder tensor plus an all-`False` `<name>_mask`, so a
  single batch may freely mix samples from different datasets.
- **`SampleUnifier`** maps raw sample dicts onto the union schema, padding or cropping
  sequence values to their declared length and ingesting any dataset-supplied
  `<name>_mask` (aligned and AND-ed with the padding mask). Metadata (`dataset`,
  `subset`, `key`) is preserved.
- **`unified_collate`** stacks tensor entries and gathers string metadata into lists.

### Structural schema

- **`TensorSpec`** declares a single value — a feature or target purely by which list
  it is placed in — as a **vector** (`()` or `(F,)`) or a **sequence** (`(T,)` or
  `(T, F)`); it is a sequence exactly when `time_dim` is set. No modality or model
  concept appears anywhere.
- **`DatasetSchema`** / **`UnifiedSchema`** hold per-dataset specs and merge them into
  the validated union (shared keys must agree on shape and dtype).

### Dataset adapters

- **`HDF5Dataset`** reads per-sample groups from HDF5 (worker-safe lazy handle,
  byte-string decoding), with optional `cache_size` (LRU) and `preload`.
- **`NpyFolderDataset`** reads memory-mapped `root/<sample>/<key>.npy` folders.
- **`DictTensorDataset`** provides an in-memory, random-tensor dataset for tests and
  quick experiments.
- **`split_indices`** (+ `save_split_info` / `load_split_info`) produces reproducible,
  optionally stratified train/val/test index splits.

### Mixing strategies

- **`ProportionalStrategy`**, **`TemperatureStrategy`** (`size ** (1 / T)`;
  `temperature=2` recovers square-root scaling), **`AnnealedTemperatureStrategy`**
  (per-epoch annealing), **`FixedWeightStrategy`** (explicit or uniform weights), and
  **`RoundRobinStrategy`** (equal, interleaved draws) — each producing a flat list of
  global indices per epoch.

### Subsampling

- **`SubsampleConfig`** / **`IndexPool`** support with/without-replacement draws, two
  exhaustion policies (`FRESH` re-randomises every epoch, `EXHAUST` covers the whole
  dataset before any reuse), per-dataset `effective_size`, and per-sample
  `sample_weights` — all reproducible from a seed and epoch.
- **`class_weights_for_sampler`** derives per-sample inverse-frequency `sample_weights`
  from a target (for the sampler).

### Class balance

- **`class_weights_for_loss`** and **`class_histogram`** compute per-class **loss** weights and
  exact counts over every valid labelled position of a categorical target (schemes:
  inverse frequency or effective-number-of-samples, Cui et al. 2019). Persist and reuse
  via ``omniloader class-weights-for-loss CONFIG --target KEY -o weights.json`` for
  `CrossEntropyLoss(weight=...)` — the loss/model usage stays in your training code.

### Variable-length batching

- **`DynamicCollator`** pads each sequence key to the batch's longest sample instead of
  a fixed `time_dim` (pair with `OmniLoader(pad_features=False)`).
- **`LengthBucketBatchSampler`** (+ `OmniLoader.sequence_lengths`) groups similar-length
  sequences to minimise padding waste in the native-length regime.

### Normalization

- **`Normalize`**, **`MinMaxNormalize`**, **`RobustNormalize`** (median/IQR) and
  **`InstanceNormalize`** (per-sample, no external stats).
- **`PerDatasetNormalize`** standardizes each sample by **its source dataset's** own
  statistics (selected from the `dataset` metadata key) — the CMVN/AdaBN remedy for
  cross-corpus domain shift — with a `fallback` (`instance` / `union` / `identity`) for
  dataset ids unseen at training time, so inference on new sources is handled
  deliberately.
- **`compute_stats`** (pooled) and **`compute_dataset_stats`** (per source dataset)
  return mean/std/min/max/median/iqr over valid positions; `save_stats` / `load_stats`
  round-trip either the flat or nested per-dataset form through JSON. The streaming
  `compute_feature_stats` remains for `(mean, std)` only. Config transforms may
  reference a saved stats file via `stats_path` / `union_stats_path` (loaded with
  `load_stats`) instead of inlining the numbers, so normalization is fully config-driven.

### Augmentation

- **`GaussianNoise`** (feature corruption), **`FeatureDropout`** (stream/modality
  dropout, keeps at least one stream), **`SpanMasking`** (SpecAugment-style time spans),
  **`FeatureMasking`** (feature bands), **`TimeWarp`** (random speed perturbation),
  **`RandomCrop`** / **`CenterCrop`** (shared offset keeps features and targets aligned),
  and **`MixupCollator`** (MixUp/CutMix at collate, emitting `mixup_lambda` and paired
  targets for the model to combine — no loss logic in the library).
- **`Compose`** chains transforms; every transform is mask-aware, seedable and
  self-skips during evaluation.

### Distributed & reproducibility

- **`OmniSampler`** turns a strategy into a per-epoch index stream and shards it across
  `rank` / `num_replicas` (inferred from `torch.distributed`) for multi-GPU runs.
- Per-sample augmentation is seeded from `(seed, epoch, index)` via `set_epoch`, and
  **`seed_worker`** seeds DataLoader workers — augmentation is deterministic and
  independent of worker count.

### Introspection

- **`describe`** (coverage matrix, valid fractions, class distributions) and
  **`validate`** (dry-run spec check), plus `OmniLoader.describe()` / `.validate()`.

### Configuration, CLI & Lightning

- **`OmniConfig`** loads from a dict, JSON or YAML and describes an entire run — seed,
  DataLoader knobs, mixing strategy, per-dataset subsampling, transform pipeline,
  collate function, length bucketing, distributed settings and the datasets themselves.
  **`build_dataloader`** assembles a ready `DataLoader` end-to-end; `build_loader`,
  `build_strategy`, `build_transform`, `build_collate`, `build_subsample` and
  `build_datasets` expose the pieces. `seed_everything` seeds Python, NumPy and PyTorch.
- **`omniloader` CLI**: `describe` / `validate` / `compute-stats` / `class-weights-for-loss`
  over a config file.
- **Annotated config template** shipped as package data (`config_template.yaml`),
  documenting every option with defaults; locate it with `config_template_path`.
- **`OmniDataModule`** (optional `omniloader[lightning]` extra) wires train/val/test
  splits, the mixing strategy, distributed sharding and seeding; the core stays
  pure-torch.
