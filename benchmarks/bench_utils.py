#!/usr/bin/env python3
"""Utilities for profiling ASV benchmarks with a variety of tools."""

import cProfile
import gc
import subprocess
import time
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, NamedTuple, cast

from benchmarks.bytecode.visualise import print_bytecode

DEFAULT_OUTPUT_DIRECTORY = Path(__file__).parent / "profiles"
PROFILERS = (
    "run",
    "timeit",
    "snakeviz",
    "viztracer",
    "flameprof",
    "pyinstrument",
    "dis",
)


class Benchmark(NamedTuple):
    """A wrapper for a benchmark function with optional setup funtion."""

    body: Callable[[], Any]
    setup: Callable[[], Any] | None = None


def warmed_timeit(
    func: Callable[[], Any],
    setup: Callable[[], Any] | None = None,
    number: int = 100,
    warmup: int = 3,
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
    times = [0.0 for _ in range(number)]
    offset = [0.0 for _ in range(number)]

    # Run the interleaved setup and function calls to warm up
    for _ in range(warmup):
        if setup is not None:
            setup()
        func()

    for i in range(number):
        # Optionally run setup code (for example resetting mutable state) before
        # each measurement iteration
        if setup is not None:
            setup()
        # Calculate the base cost of method invocation and timing overhead, so
        # we can offset our final measurements by it
        offset_start = time.perf_counter()
        benchmark_class_empty()
        offset_end = time.perf_counter()
        offset[i] = offset_end - offset_start
        # Time the actual function we want to measure
        func_start = time.perf_counter()
        func()
        func_end = time.perf_counter()
        times[i] = func_end - func_start

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


def parse_arguments(benchmark_names: list[str]) -> ArgumentParser:
    """Parse the arguments for the profiler tool."""
    parser = ArgumentParser()

    parser.add_argument(
        "test",
        choices=benchmark_names + ["all"],
        help="the name of the benchmark to run, `all` to run all benchmarks",
    )
    parser.add_argument(
        "profiler",
        choices=PROFILERS,
        nargs="?",
        default=PROFILERS[0],
        help="the profiler to use",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="the directory into which to write out the profile files",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="don't show the profiler's UI"
    )

    return parser


def get_benchmark_runs(
    args: Namespace,
    benchmarks: dict[str, Benchmark],
) -> list[tuple[str, Benchmark]]:
    """Get the benchmark to profile."""
    if args.test == "all":
        return list(benchmarks.items())

    name = args.test
    test = benchmarks[name]
    return [(name, test)]


def run_benchmark(
    args: Namespace,
    benchmarks: dict[str, Benchmark],
    warmup: bool = False,
) -> None:
    """Directly run a benchmark."""
    benchmark_runs = get_benchmark_runs(args, benchmarks)
    for name, (test, setup) in benchmark_runs:
        if warmup:
            if setup is not None:
                setup()
            test()
        if setup is not None:
            setup()
        start_time = time.time()
        test()
        print(f"Test {name} ran in: {time.time() - start_time:.5f}s")


def timeit_benchmark(
    args: Namespace,
    benchmarks: dict[str, Benchmark],
) -> None:
    """Use a custom function based on timeit to run a benchmark."""
    benchmark_runs = get_benchmark_runs(args, benchmarks)
    for name, (test, setup) in benchmark_runs:
        me, _, std = warmed_timeit(test, setup=setup)
        print(f"Test {name} ran in: {me:.3g} ± {std:.3g}s")


def cprofile_benchmark(
    args: Namespace,
    benchmarks: dict[str, Benchmark],
    warmup: bool = False,
) -> Path:
    """Use cProfile to profile a benchmark."""
    benchmark_runs = get_benchmark_runs(args, benchmarks)
    if len(benchmark_runs) != 1:
        raise ValueError("Cannot profile multiple benchmarks together")
    name, (test, setup) = benchmark_runs[0]
    output_prof = args.output / f"{name}.prof"
    if warmup:
        if setup is not None:
            setup()
        test()
    if setup is not None:
        setup()
    profiler = cProfile.Profile()
    profiler.enable()
    test()
    profiler.disable()
    profiler.dump_stats(str(output_prof))
    return output_prof


def viztracer_benchmark(
    args: Namespace,
    benchmarks: dict[str, Benchmark],
    warmup: bool = True,
) -> Path:
    """Use VizTracer to profile a benchmark."""
    from viztracer import VizTracer  # pyright: ignore[reportMissingTypeStubs]

    benchmark_runs = get_benchmark_runs(args, benchmarks)
    if len(benchmark_runs) != 1:
        raise ValueError("Cannot profile multiple benchmarks together")
    name, (test, setup) = benchmark_runs[0]
    output_prof = args.output / f"{name}.json"
    if warmup:
        if setup is not None:
            setup()
        test()
    if setup is not None:
        setup()
    with VizTracer(output_file=str(output_prof)):
        test()
    return output_prof


def pyinstrument_benchmark(
    args: Namespace,
    benchmarks: dict[str, Benchmark],
    warmup: bool = True,
) -> Path:
    """Use pyinstrument to profile a benchmark."""
    from pyinstrument import Profiler

    benchmark_runs = get_benchmark_runs(args, benchmarks)
    if len(benchmark_runs) != 1:
        raise ValueError("Cannot profile multiple benchmarks together")
    name, (test, setup) = benchmark_runs[0]
    output_prof = args.output / f"{name}.html"
    if warmup:
        if setup is not None:
            setup()
        test()
    if setup is not None:
        setup()
    profiler = Profiler(interval=1e-9)
    profiler.start()
    test()
    profiler.stop()
    profiler.write_html(output_prof)
    return output_prof


def dis_benchmark(
    args: Namespace,
    benchmarks: dict[str, Benchmark],
):
    """Use dis to disassemble a benchmark."""
    benchmark_runs = get_benchmark_runs(args, benchmarks)
    if len(benchmark_runs) != 1:
        raise ValueError("Cannot disassemble multiple benchmarks together")
    _, (test, setup) = benchmark_runs[0]
    if setup is not None:
        setup()
    print_bytecode(test)


def show(
    args: Namespace,
    output_prof: Path,
    tool: str,
    options: tuple[str, ...] | None = None,
) -> None:
    """Show the profile using the specified tool."""
    if args.quiet:
        return
    if options is None:
        options = cast(tuple[str], ())
    command = ["uv", "run", tool, output_prof, *options]
    subprocess.run(command, check=True)  # noqa: S603


def profile(
    benchmarks: dict[str, Benchmark],
    argv: list[str] | None = None,
) -> None:
    """Run the selected profiler."""
    if not benchmarks:
        raise ValueError("At least one benchmark must be provided to profile!")

    args = parse_arguments(list(benchmarks.keys())).parse_args(args=argv)

    match args.profiler:
        case "run":
            run_benchmark(args, benchmarks)
        case "timeit":
            timeit_benchmark(args, benchmarks)
        case "snakeviz":
            output_prof = cprofile_benchmark(args, benchmarks)
            show(args, output_prof, tool="snakeviz")
        case "flameprof":
            output_prof = cprofile_benchmark(args, benchmarks)
            show(args, output_prof, tool="flameprof")
        case "viztracer":
            output_prof = viztracer_benchmark(args, benchmarks)
            show(args, output_prof, tool="vizviewer")
        case "pyinstrument":
            output_prof = pyinstrument_benchmark(args, benchmarks)
            show(args, output_prof, tool="open")
        case "dis":
            dis_benchmark(args, benchmarks)
        case _:
            raise ValueError("Invalid command!")
