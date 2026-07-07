"""Packaged configuration templates shipped with OmniLoader.

The annotated ``config_template.yaml`` lists every option
:class:`~omniloader.config.OmniConfig` understands, with defaults and inline
guidance. Locate it programmatically with :func:`config_template_path` (e.g. to
copy it as a starting point for a new experiment).
"""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

#: File name of the annotated config template within this package.
CONFIG_TEMPLATE = "config_template.yaml"


def config_template_path() -> Path:
    """Return the filesystem path to the annotated ``config_template.yaml``.

    The template ships as package data, so this resolves it wherever OmniLoader
    is installed (source tree or wheel).

    Returns:
        The path to the annotated config template.

    """
    with as_file(files(__package__).joinpath(CONFIG_TEMPLATE)) as path:
        return Path(path)


__all__ = ["CONFIG_TEMPLATE", "config_template_path"]
