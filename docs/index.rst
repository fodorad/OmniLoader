OmniLoader
==========

**OmniLoader** is a PyTorch meta data loader that unifies several datasets with
disjoint annotations into a single, masked sample scheme so a model can be
trained jointly across all of them.

**The problem.** Multi-task learning wants one model to learn many related tasks at
once, but the supervision for those tasks is usually scattered across *separate
datasets that annotate different things*: one corpus has a single label per sample
(e.g. sentiment), another a per-step label sequence (e.g. valence/arousal), a third
categorical class ids — and each covers only its own subset of the features. No single
dataset carries every task's labels, so a shared-backbone model cannot simply be
pointed at one file. The samples first have to be brought to **one common, batchable
scheme** where every task's slot is always present and the loss can tell a *real* label
apart from a *missing* one.

**What OmniLoader does.** It builds the *union* of every feature and target across your
datasets. Every sample it yields carries the full union schema: keys a dataset provides
are copied (and, for sequences, padded/cropped to a declared length), while keys it
lacks are filled with a placeholder tensor plus an all-``False`` ``<name>_mask``. So a
single batch can freely mix samples from different datasets, each task's head sees a
consistently-shaped tensor, and your loss and model always know which values are real.
It is a map-style :class:`torch.utils.data.Dataset` you wrap in an ordinary
``DataLoader`` — it unifies and masks, it does not reinvent batching.

OmniLoader is **modality-, dataset- and model-agnostic**: it knows nothing about video,
audio, text, which specific corpora you loaded, or any model. It reasons only about
**vectors** (shape ``()`` or ``(F,)``) and **sequences** (shape ``(T,)`` or
``(T, F)``); a value is a sequence exactly when its spec sets ``time_dim``.

Beyond this core unification and its utilities (schema declaration, dataset adapters,
splits, introspection), OmniLoader also ships the surrounding machinery a joint
training run needs: **mixing strategies** to balance datasets of very different sizes,
**subsampling** to control within-dataset draws, **class-balance** calculations
(sampler weights and loss weights), and a mask-aware transform pipeline of
**normalization** and **augmentation** techniques — all configurable end-to-end from a
single file.

.. toctree::
   :maxdepth: 2
   :caption: Contents

Feature overview
----------------

.. list-table::
   :header-rows: 1
   :widths: 16 26 58

   * - Category
     - Feature
     - What it does
   * - **Core**
     - :class:`~omniloader.loader.OmniLoader`
     - Concatenates disjoint datasets into one masked, unified stream.
   * -
     - :class:`~omniloader.schema.unify.SampleUnifier`
     - Maps a raw sample onto the union schema; fills gaps with placeholder + ``<name>_mask``.
   * -
     - :func:`~omniloader.collate.unified_collate`
     - Stacks tensors, lists metadata.
   * - **Schema**
     - :class:`~omniloader.schema.spec.TensorSpec`
     - Declare a value (feature or target): ``feature_dim``, ``time_dim``, dtype, placeholder.
   * -
     - :class:`~omniloader.schema.spec.DatasetSchema` / :class:`~omniloader.schema.spec.UnifiedSchema`
     - Per-dataset specs; merged + validated union.
   * -
     - vector / sequence
     - Structural, modality-agnostic (``()``, ``(F,)``, ``(T,)``, ``(T, F)``).
   * - **Datasets**
     - :class:`~omniloader.data.datasets.HDF5Dataset`
     - Per-sample HDF5 groups; worker-safe; ``cache_size`` / ``preload``.
   * -
     - :class:`~omniloader.data.npy.NpyFolderDataset`
     - Memory-mapped ``root/<sample>/<key>.npy`` layout.
   * -
     - :class:`~omniloader.data.datasets.DictTensorDataset`
     - In-memory random tensors (tests/experiments).
   * -
     - :func:`~omniloader.data.splits.split_indices`
     - Reproducible, optionally stratified train/val/test splits.
   * - **Mixing**
     - :class:`~omniloader.sampling.strategies.ProportionalStrategy`
     - Sample by true dataset sizes.
   * -
     - :class:`~omniloader.sampling.strategies.TemperatureStrategy`
     - ``size ** (1 / T)`` re-weighting (T=2 → sqrt).
   * -
     - :class:`~omniloader.sampling.strategies.AnnealedTemperatureStrategy`
     - Temperature annealed per epoch.
   * -
     - :class:`~omniloader.sampling.strategies.FixedWeightStrategy`
     - Explicit or uniform per-dataset weights.
   * -
     - :class:`~omniloader.sampling.strategies.RoundRobinStrategy`
     - Equal, interleaved draws.
   * - **Subsampling**
     - :class:`~omniloader.sampling.strategies.SubsampleConfig` / :class:`~omniloader.sampling.subsamplers.IndexPool`
     - Replacement, FRESH/EXHAUST, ``effective_size``, per-sample weights.
   * -
     - :func:`~omniloader.sampling.weights.class_weights_for_sampler`
     - Per-sample inverse-frequency weights for the sampler.
   * - **Class balance**
     - :func:`~omniloader.sampling.weights.class_weights_for_loss` / :func:`~omniloader.sampling.weights.class_histogram`
     - Per-class loss weights + exact counts for ``CrossEntropyLoss(weight=)`` (persist via the CLI).
   * - **Batching**
     - :class:`~omniloader.collate.DynamicCollator`
     - Pad sequences to per-batch max, not fixed length (native-length mode only).
   * -
     - :class:`~omniloader.sampling.bucketing.LengthBucketBatchSampler`
     - Group similar-length sequences to cut padding (needs ``pad_features=False``).
   * - **Normalization**
     - :class:`~omniloader.transforms.normalize.Normalize` / ``MinMax`` / ``Robust`` / ``Instance``
     - Standardize features (from stats or per-sample).
   * -
     - :class:`~omniloader.transforms.normalize.PerDatasetNormalize`
     - Standardize each sample by its source dataset's stats, with an inference fallback for unseen sources.
   * -
     - :func:`~omniloader.transforms.stats.compute_stats` / :func:`~omniloader.transforms.stats.compute_dataset_stats`
     - Pooled or per-dataset mean/std/min/max/median/iqr.
   * -
     - :func:`~omniloader.transforms.stats.save_stats` / :func:`~omniloader.transforms.stats.load_stats`
     - JSON persistence (flat or per-dataset).
   * - **Augmentation**
     - :class:`~omniloader.transforms.augment.GaussianNoise`
     - Feature corruption.
   * -
     - :class:`~omniloader.transforms.augment.FeatureDropout`
     - Drop whole feature streams (modality dropout).
   * -
     - :class:`~omniloader.transforms.augment.SpanMasking` / :class:`~omniloader.transforms.augment.FeatureMasking`
     - Zero time spans / feature bands.
   * -
     - :class:`~omniloader.transforms.augment.TimeWarp`
     - Random speed perturbation.
   * -
     - :class:`~omniloader.transforms.crop.RandomCrop` / :class:`~omniloader.transforms.crop.CenterCrop`
     - Windowed sequence crop (features + targets aligned).
   * -
     - :class:`~omniloader.transforms.mix.MixupCollator`
     - MixUp/CutMix at collate (emits ``mixup_lambda`` + paired targets).
   * -
     - :class:`~omniloader.transforms.base.Compose`
     - Chain transforms (mask-aware, seedable, train/eval-gated).
   * - **Distributed**
     - :class:`~omniloader.sampling.sampler.OmniSampler`
     - DDP index sharding (``num_replicas`` / ``rank``).
   * -
     - ``set_epoch`` + :func:`~omniloader.utils.seeding.seed_worker`
     - Reproducible, worker-count-independent augmentation.
   * - **Introspection**
     - :func:`~omniloader.introspection.describe`
     - Coverage matrix, valid fractions, class distributions.
   * -
     - :func:`~omniloader.introspection.validate`
     - Dry-run check of data vs declared specs.
   * - **Config & CLI**
     - :class:`~omniloader.config.OmniConfig`
     - One JSON/YAML file: seed, strategy, subsample, transforms, collate, bucketing, DDP & DataLoader knobs, datasets.
   * -
     - :meth:`~omniloader.config.OmniConfig.build_dataloader`
     - Assemble a ready ``DataLoader`` end-to-end from the config alone.
   * -
     - :func:`~omniloader.data.factory.build_datasets`
     - Construct datasets from a declarative ``datasets`` section.
   * -
     - :func:`~omniloader.templates.config_template_path`
     - Locate the shipped, fully annotated ``config_template.yaml`` (every option + defaults).
   * -
     - ``omniloader`` CLI
     - ``describe`` / ``validate`` / ``compute-stats`` / ``class-weights-for-loss``.
   * - **Integration**
     - :class:`~omniloader.integrations.lightning.OmniDataModule`
     - Optional Lightning module (wires strategy, DDP, seeding).

Quickstart
----------

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   from omniloader import (
       DatasetSchema, DictTensorDataset, TensorSpec,
       OmniLoader, TemperatureStrategy, unified_collate,
   )

   # Dataset A: a sequence target (per-step) over a feature sequence.
   ds_a = DictTensorDataset({"video": torch.randn(40, 16, 32), "valence": torch.randn(40, 16)})
   schema_a = DatasetSchema(
       features=[TensorSpec("video", feature_dim=32, time_dim=16)],  # sequence (T, F)
       targets=[TensorSpec("valence", time_dim=16, placeholder=-5.0)],  # sequence (T,)
   )

   # Dataset B: a scalar target (one per sample), different feature sequence.
   ds_b = DictTensorDataset({"audio": torch.randn(200, 24, 8), "sentiment": torch.randn(200)})
   schema_b = DatasetSchema(
       features=[TensorSpec("audio", feature_dim=8, time_dim=24)],  # sequence (T, F)
       targets=[TensorSpec("sentiment", placeholder=-5.0)],  # vector scalar ()
   )

   omni = OmniLoader([ds_a, ds_b], [schema_a, schema_b])
   sampler = omni.make_sampler(TemperatureStrategy(omni.dataset_sizes, temperature=2.0))
   loader = DataLoader(omni, batch_size=8, sampler=sampler, collate_fn=unified_collate)

   for batch in loader:
       # Every batch carries video, audio, valence, sentiment + their masks.
       print(batch["valence"].shape, batch["valence_mask"].shape)
       break

A single dataset works too — pass one-element lists (``OmniLoader([ds_a], [schema_a])``).
The union is then just that dataset's schema and the multi-dataset machinery (mixing
strategies, per-dataset normalization) simply lies dormant.

Config-driven runs
------------------

A single JSON/YAML file can describe an entire experiment — seed, DataLoader
knobs, mixing strategy, per-dataset subsampling, the transform pipeline, the
collate function, length bucketing, distributed settings and the datasets
themselves. :meth:`~omniloader.config.OmniConfig.build_dataloader` assembles a
ready ``DataLoader`` from it:

.. code-block:: python

   from omniloader import OmniConfig

   config = OmniConfig.from_file("config.yaml")
   train_loader = config.build_dataloader(training=True)
   val_loader = config.build_dataloader(training=False)  # sequential, augmentations off

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
