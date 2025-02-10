#!/usr/bin/env python3
"""Utilities for profiling ASV benchmarks with a variety of tools."""

import cProfile
import subprocess
import timeit
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

DEFAULT_OUTPUT_DIRECTORY = Path(__file__).parent / "profiles"
PROFILERS = (
    "timeit",
    "snakeviz",
    "viztracer",
    "flameprof"
)


def parse_arguments(benchmark_names: list[str]) -> ArgumentParser:
    """Parse the arguments for the profiler tool."""
    parser = ArgumentParser()

    parser.add_argument(
        "test",
        choices=benchmark_names,
        help="the name of the benchmark to run"
    )
    parser.add_argument(
        "profiler",
        choices=PROFILERS,
        help="the profiler to use"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="the directory into which to write out the profile files",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="don't show the profiler's UI"
    )

    return parser


def get_benchmark(
    args: Namespace, benchmarks: dict[str, Callable[[], Any]]
) -> tuple[str, Callable[[], None]]:
    """Get the benchmark to profile."""
    return (args.test, benchmarks[args.test])


def timeit_benchmark(
    args: Namespace, benchmarks: dict[str, Callable[[], Any]], number: int = 1
) -> None:
    """Use timeit to run a benchmark."""
    name, test = get_benchmark(args, benchmarks)
    print(f"Test {name} ran in: {timeit.timeit(test, number=number):.5f}s")


def cprofile_benchmark(
    args: Namespace, benchmarks: dict[str, Callable[[], Any]]
) -> Path:
    """Use cProfile to profile a benchmark."""
    name, _ = get_benchmark(args, benchmarks)
    output_prof = args.output / f"{name}.prof"
    cProfile.run(f"{name}()", str(output_prof))
    return output_prof


def viztracer_benchmark(
    args: Namespace, benchmarks: dict[str, Callable[[], Any]]
) -> Path:
    """Use VizTracer to profile a benchmark."""
    from viztracer import VizTracer  # pyright: ignore[reportMissingTypeStubs]

    name, test = get_benchmark(args, benchmarks)
    output_prof = args.output / f"{name}.json"
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
