"""Command-line interface for inspecting a dataset mix from a config file.

Subcommands (all take a JSON/YAML config whose ``datasets`` section declares the
adapters and schemas — see :mod:`omniloader.data.factory`):

* ``omniloader describe CONFIG`` — print the coverage/statistics report.
* ``omniloader validate CONFIG`` — check datasets against their specs (exit 1 on issues).
* ``omniloader compute-stats CONFIG -o STATS`` — compute and save normalization stats.
* ``omniloader class-weights-for-loss CONFIG --target KEY -o WEIGHTS`` — compute and
  save per-class loss weights (and counts) for a categorical target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from omniloader.config import OmniConfig
from omniloader.introspection import describe, validate
from omniloader.loader import OmniLoader
from omniloader.sampling.weights import class_histogram, class_weights_for_loss
from omniloader.transforms.stats import compute_dataset_stats, compute_stats, save_stats

if TYPE_CHECKING:
    from collections.abc import Sequence


def _load(config_path: str) -> tuple[list, list]:
    """Load a config and build its declared datasets and schemas."""
    config = OmniConfig.from_file(config_path)
    datasets, schemas = config.build_datasets()
    if not datasets:
        raise SystemExit(f"Config {config_path!r} declares no datasets")
    return datasets, schemas


def _describe(args: argparse.Namespace) -> int:
    """Print the dataset report."""
    datasets, schemas = _load(args.config)
    print(describe(datasets, schemas, max_samples=args.max_samples))
    return 0


def _validate(args: argparse.Namespace) -> int:
    """Validate datasets against their specs; exit non-zero if issues are found."""
    datasets, schemas = _load(args.config)
    issues = validate(datasets, schemas, num_samples=args.num_samples)
    if issues:
        print("\n".join(issues))
        return 1
    print("OK: all datasets match their declared specs.")
    return 0


def _compute_stats(args: argparse.Namespace) -> int:
    """Compute normalization statistics over the loader and save them to JSON.

    With ``--per-dataset`` the stats are grouped by source dataset (for
    :class:`~omniloader.transforms.normalize.PerDatasetNormalize`); otherwise they
    are pooled over the union.
    """
    datasets, schemas = _load(args.config)
    loader = OmniLoader(datasets, schemas)
    keys = args.keys or loader.schema.feature_keys
    samples = (loader[i] for i in range(len(loader)))
    if args.per_dataset:
        stats = compute_dataset_stats(samples, keys)
        print(f"Saved per-dataset stats for {sorted(stats)} to {args.output}")
    else:
        stats = compute_stats(samples, keys)
        print(f"Saved stats for {sorted(stats)} to {args.output}")
    save_stats(stats, args.output)
    return 0


def _class_weights_for_loss(args: argparse.Namespace) -> int:
    """Compute per-class counts and loss weights for a target and save them to JSON."""
    datasets, schemas = _load(args.config)
    loader = OmniLoader(datasets, schemas)
    samples = [loader[i] for i in range(len(loader))]  # single pass, reused for both
    hist = class_histogram(samples, args.target, num_classes=args.num_classes)
    weights = class_weights_for_loss(
        samples, args.target, num_classes=args.num_classes, scheme=args.scheme, beta=args.beta
    )
    payload = {
        "target": args.target,
        "scheme": args.scheme,
        "num_classes": int(weights.numel()),
        "counts": hist.tolist(),
        "weights": weights.tolist(),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Saved class weights for {args.target!r} ({payload['num_classes']} classes) "
        f"to {args.output}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``omniloader`` CLI."""
    parser = argparse.ArgumentParser(prog="omniloader", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_desc = sub.add_parser("describe", help="print a dataset coverage/statistics report")
    p_desc.add_argument("config", help="path to a JSON/YAML config")
    p_desc.add_argument("--max-samples", type=int, default=64, dest="max_samples")
    p_desc.set_defaults(func=_describe)

    p_val = sub.add_parser("validate", help="check datasets against their declared specs")
    p_val.add_argument("config", help="path to a JSON/YAML config")
    p_val.add_argument("--num-samples", type=int, default=4, dest="num_samples")
    p_val.set_defaults(func=_validate)

    p_stats = sub.add_parser("compute-stats", help="compute and save normalization stats")
    p_stats.add_argument("config", help="path to a JSON/YAML config")
    p_stats.add_argument("-o", "--output", required=True, help="destination stats JSON")
    p_stats.add_argument("--keys", nargs="*", help="feature keys (default: all features)")
    p_stats.add_argument(
        "--per-dataset",
        action="store_true",
        dest="per_dataset",
        help="group stats by source dataset (for PerDatasetNormalize)",
    )
    p_stats.set_defaults(func=_compute_stats)

    p_cw = sub.add_parser("class-weights-for-loss", help="compute and save per-class loss weights")
    p_cw.add_argument("config", help="path to a JSON/YAML config")
    p_cw.add_argument("--target", required=True, help="categorical target key")
    p_cw.add_argument("-o", "--output", required=True, help="destination weights JSON")
    p_cw.add_argument("--num-classes", type=int, default=None, dest="num_classes")
    p_cw.add_argument("--scheme", choices=["inverse", "effective"], default="inverse")
    p_cw.add_argument("--beta", type=float, default=0.999, help="beta for scheme=effective")
    p_cw.set_defaults(func=_class_weights_for_loss)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``omniloader`` console script.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        The process exit code.

    """
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
