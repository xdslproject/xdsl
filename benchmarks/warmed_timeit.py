#!/usr/bin/env python3
"""A precision timer for the implementations of methods on objects.

Derived from the timeit module, this timer aims to accurately time method
implementations which may modify mutable state, even when these implementations
are very short, in the order of the timing instrumentation itself.
"""

import gc
import math
import time
from collections.abc import Callable
from statistics import mean, median, stdev
from typing import Any

PERF_COUNTER_RESOLUTION = time.get_clock_info("perf_counter").resolution


def warmed_timeit(
    func: Callable[[], Any],
    setup: Callable[[], Any] | None = None,
    repeats: int = 1000,
    warmups: int = 3,
) -> tuple[float, float, float]:
    """
    Time the contents of a class method with setup and warmup.

    Derived from <https://github.com/python/cpython/blob/3.13/Lib/timeit.py>.
    """

    class EmptyBenchmarkClass:
        """A class with an empty method call as a baseline for timing."""

        def empty(self) -> None:
            """An empty method call."""
            pass

    benchmark_class_empty = EmptyBenchmarkClass().empty

    # Inspired by timeit, we disable garbage collection for less noise in
    # measurements
    gcold = gc.isenabled()
    gc.disable()

    # Pre-populate the arrays to avoid costs of re-sizing them
    times = [0.0 for _ in range(repeats)]
    offset = [0.0 for _ in range(repeats)]

    # Warm up before measuring
    for _ in range(warmups):
        if setup is not None:
            setup()
        func()

    # If the function is close to the resolution of the timer, group it into
    # batches to aim for measurement periods of at least 25x the resolution.
    if setup is not None:
        setup()
    batch_size_func_start = time.perf_counter()
    func()
    batch_size_func_end = time.perf_counter()
    single_func_time = batch_size_func_end - batch_size_func_start
    if single_func_time > PERF_COUNTER_RESOLUTION * 25:
        batch_size = 1
    else:
        batch_size = math.ceil(
            (PERF_COUNTER_RESOLUTION * 25 * 1000000000)
            / (single_func_time * 1000000000)
        )

    for i in range(repeats):
        # Calculate the base cost of method invocation and timing overhead, so
        # we can offset our final measurements by it. Setup functions are invoked
        # in separate clauses despite code duplication to avoid overhead
        loop = range(batch_size)
        if setup is not None:
            offset_start = time.perf_counter()
            for _ in loop:
                setup()
                benchmark_class_empty()
            offset_end = time.perf_counter()
        else:
            offset_start = time.perf_counter()
            for _ in loop:
                benchmark_class_empty()
            offset_end = time.perf_counter()
        offset[i] = (offset_end - offset_start) / batch_size

        # Time the actual function we want to measure
        loop = range(batch_size)
        if setup is not None:
            func_start = time.perf_counter()
            for _ in loop:
                setup()
                func()
            func_end = time.perf_counter()
        else:
            func_start = time.perf_counter()
            for _ in loop:
                func()
            func_end = time.perf_counter()
        times[i] = (func_end - func_start) / batch_size

    # Re-enable the garbage collector if it was initially on
    if gcold:
        gc.enable()

    # Return the mean, median, and standard deviations of the measured times.
    # The mean offset is subtracted from the median time for the best
    # approximation given the data we can record
    return (
        mean(times) - mean(offset),
        median(times) - mean(offset),
        stdev(times) + stdev(offset),
    )
