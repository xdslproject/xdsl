"""
marimo documentation package.

Copied from https://github.com/marimo-team/marimo/blob/f25319284d439e8d809fab421f27e9226cb00a65/docs/__init__.py#L3

Currently mkdocs-marimo doesn't ship the best way to embed notebooks in mkdocs, they use
this mechanism instead, which we should replace with mkdocs-marimo when they update the
package.
"""

from . import blocks

__all__ = ["blocks"]
