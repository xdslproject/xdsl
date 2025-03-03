#!/usr/bin/env python3
"""Utilities for profiling ASV benchmarks with a variety of tools."""

import cProfile
import subprocess
import time
import timeit
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, cast

DEFAULT_OUTPUT_DIRECTORY = Path(__file__).parent / "profiles"
PROFILERS = ("run", "timeit", "snakeviz", "viztracer", "flameprof")


def warmed_timeit(
    func: Callable[[], Any], number: int = 100, warmup: int = 3
) -> tuple[float, float, float]:
    """Time the contents of a class method with warmup."""

    class EmptyBenchmarkClass:
        """A benchmark class for the empty function."""

        def empty(self) -> None:
            """An empty function call."""
            pass

    benchmark_class = EmptyBenchmarkClass()
    timeit.timeit(func, number=warmup)
    times = timeit.repeat(func, repeat=number, number=1)
    offset = timeit.repeat(benchmark_class.empty, repeat=number, number=1)
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
        help="the name of the benchmark to run",
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
    args: Namespace, benchmarks: dict[str, Callable[[], Any]]
) -> list[tuple[str, Callable[[], None]]]:
    """Get the benchmark to profile."""
    if args.test == "all":
        return [(name, test) for name, test in benchmarks.items()]
    return [(args.test, benchmarks[args.test])]


def run_benchmark(args: Namespace, benchmarks: dict[str, Callable[[], Any]]) -> None:
    """Directly run a benchmark."""
    benchmark_runs = get_benchmark_runs(args, benchmarks)
    for name, test in benchmark_runs:
        start_time = time.time()
        test()
        print(f"Test {name} ran in: {time.time() - start_time:.5f}s")


def timeit_benchmark(args: Namespace, benchmarks: dict[str, Callable[[], Any]]) -> None:
    """Use timeit to run a benchmark."""
    benchmark_runs = get_benchmark_runs(args, benchmarks)
    for name, test in benchmark_runs:
        me, _, std = warmed_timeit(test)
        print(f"Test {name} ran in: {me:.3g} ± {std:.3g}s")


def cprofile_benchmark(
    args: Namespace, benchmarks: dict[str, Callable[[], Any]], warmup: bool = False
) -> Path:
    """Use cProfile to profile a benchmark."""
    benchmark_runs = get_benchmark_runs(args, benchmarks)
    if len(benchmark_runs) != 1:
        raise ValueError("Cannot profile multiple benchmarks together")
    name, test = benchmark_runs[0]
    output_prof = args.output / f"{name}.prof"
    if warmup:
        test()
    profiler = cProfile.Profile()
    profiler.enable()
    test()
    profiler.disable()
    profiler.dump_stats(str(output_prof))
    return output_prof


def viztracer_benchmark(
    args: Namespace, benchmarks: dict[str, Callable[[], Any]], warmup: bool = False
) -> Path:
    """Use VizTracer to profile a benchmark."""
    from viztracer import VizTracer  # pyright: ignore[reportMissingTypeStubs]

    benchmark_runs = get_benchmark_runs(args, benchmarks)
    if len(benchmark_runs) != 1:
        raise ValueError("Cannot profile multiple benchmarks together")
    name, test = benchmark_runs[0]
    output_prof = args.output / f"{name}.json"
    if warmup:
        test()
    with VizTracer(output_file=str(output_prof)):
        test()
    return output_prof


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
    benchmarks: dict[str, Callable[[], Any]], argv: list[str] | None = None
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
        case _:
            raise ValueError("Invalid command!")
