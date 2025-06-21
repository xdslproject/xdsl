"""
This file has benchmarks used in CI to catch regressions.

This is in addition to the performance benchmarking in `../../benchmarks`, which track
the speed of functions over time.
"""

import pytest

from xdsl.dialects.arith import ConstantOp


@pytest.mark.benchmark
def test_const_verify():
    ConstantOp.from_int_and_width(1, 32).verify()
