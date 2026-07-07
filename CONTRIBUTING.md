# Contributing to OmniLoader

Thank you for your interest in contributing! This guide covers everything you need to get started.

---

## Quick start

```bash
git clone https://github.com/fodorad/OmniLoader
cd OmniLoader
uv pip install -e ".[all,dev,docs]"
pre-commit install   # optional: runs ruff automatically before every commit
```

> OmniLoader's core installs with just `torch` + `numpy` + `h5py` + `pyyaml`.
> The `lightning` extra is only needed for the optional `OmniDataModule`.

---

## Architecture

OmniLoader is a small library grouped into subpackages by concern, with a clear
dependency direction (upper layers call down only):

```
integrations/lightning.py                 optional PyTorch Lightning glue
   │
loader.py, config.py, collate.py          concatenate + unify + sample (user-facing surface)
   │
sampling/    strategies.py, subsamplers.py, sampler.py     (mixing / balancing)
transforms/  base.py, normalize.py, augment.py, stats.py   (normalization / augmentation)
   │
schema/  spec.py, unify.py                 the data model (specs + unification)
data/    datasets.py                       dataset adapters / IO
utils/   padding.py                        leaf helpers
```

Keep new mixing strategies in `sampling/strategies.py` (subclass `MixingStrategy`),
new transforms in `transforms/augment.py` or `transforms/normalize.py`, new dataset
adapters in `data/datasets.py`, and schema primitives in `schema/spec.py`.

---

## Development workflow

1. **Fork** the repository and create a branch from `main`.
2. **Make your changes** — keep them focused and minimal.
3. **Write or update tests** in `tests/` (the tree mirrors `omniloader/`).
4. **Run checks locally** before pushing:

   ```bash
   make fix    # auto-format and fix lint issues
   make check  # lint + type-check + tests + docs build (mirrors CI)
   ```

5. **Open a Pull Request** against `main` and fill in the template.

---

## Commit message convention

OmniLoader follows **Conventional Commits** so the version history is readable and
the correct version bump is signalled automatically (via release-please).

| Prefix | Meaning | Version bump |
|--------|---------|--------------|
| `fix:` | Bug fix, regression, hotfix | **Patch** (x.y.Z) |
| `feat:` | New feature | **Minor** (x.Y.0) |
| `feat!:` or `BREAKING CHANGE:` | API change that breaks existing usage | **Major** (X.0.0) |
| `docs:` | Documentation only | No bump |
| `test:` | Tests only | No bump |
| `refactor:` | Code refactor with no behaviour change | No bump |
| `chore:` | Build, CI, dependency updates | No bump |

---

## Release process

Releases are automated with **release-please**. Merging Conventional Commits to
`main` opens/maintains a release PR; merging that PR tags the version and creates
a GitHub Release. Docs deploy on every push to `main`.

---

## Code style

- **Formatter / linter**: [ruff](https://docs.astral.sh/ruff/) — run `make fix`.
- **Type checker**: [ty](https://github.com/astral-sh/ty) — run `make type-check`.
- **Line length**: 100 characters.
- **Python version**: 3.12+.
- **Docstrings**: Google style (rendered by Sphinx autoapi + napoleon).
- **Type hints**: required on public function signatures.

---

## Tests

```bash
make test              # run all tests with coverage
coverage html          # open coverage_html/index.html to browse
```

Tests live in `tests/` and mirror the `omniloader/` package structure. They use
**synthetic random tensors** (and a tiny temporary HDF5 file) generated in
`setUp` — no private datasets required. We use the standard-library `unittest`
framework and avoid mocks.

---

## License

By contributing you agree that your work will be released under the [MIT License](LICENSE).
