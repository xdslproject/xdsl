#!/usr/bin/env python3
"""Benchmark the time to import xDSL classes.

These are are simple example which can be used as a template for more
complex/helpful benchmarks. See "Writing benchmarks" in the asv docs for more
information.
"""

from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parent


def time_import_xdsl_opt() -> None:
    """Import benchmark using the default asv mechanism."""
    from xdsl.xdsl_opt_main import xDSLOptMain  # noqa: F401


def timeraw_import_xdsl_opt() -> str:
    """Import benchmark using the `raw` asv mechanism."""
    return """
    from xdsl.xdsl_opt_main import xDSLOptMain
    """


if __name__ == "__main__":
    from bench_utils import profile  # type: ignore

    BENCHMARKS = {
        "time_import_xdsl_opt": time_import_xdsl_opt,
    }
    profile(BENCHMARKS)
