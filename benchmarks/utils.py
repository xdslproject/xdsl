#!/usr/bin/env python3
"""Utilities for profiling ASV benchmarks with a variety of tools."""

from pathlib import Path
from argparse import ArgumentParser, Namespace
import timeit

import cProfile
from collections.abc import Callable
from typing import Any, Iterable, cast
import subprocess
from viztracer import VizTracer

DEFAULT_OUTPUT_DIRECTORY = Path(__file__).parent / "profiles"


def parse_arguments() -> ArgumentParser:
    """Parse the arguments for the profiler tool."""
    parser = ArgumentParser()

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="the directory into which to write out the profile files",
    )
    parser.add_argument(
        "-t",
        "--test",
        help="the name of the test to run",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="show the profiler's UI"
    )

    sub_parsers = parser.add_subparsers(dest="command", required=True)
    sub_parsers.add_parser(
        "timeit", help="use the timeit"
    )
    sub_parsers.add_parser(
        "snakeviz", help="use the SnakeViz profiler"
    )
    sub_parsers.add_parser(
        "viztracer", help="use the VizTracer profiler"
    )
    sub_parsers.add_parser(
        "flameprof", help="use the flameprof profiler"
    )

    return parser


def get_benchmarks(args: Namespace, benchmarks: dict[str, Callable[[], Any]]) -> Iterable[tuple[str, Callable[[], None]]]:
    """Get the filtered set of benchmarks items to profile."""
    if args.test is not None and args.test in benchmarks:
        benchmarks = {args.test: benchmarks[args.test]}
    return benchmarks.items()


def timeit_benchmark(args: Namespace, benchmarks: dict[str, Callable[[], Any]], number: int = 1) -> None:
    """Use timeit to run a benchmark."""
    for name, test in get_benchmarks(args, benchmarks):
        print(f"Test {name} ran in: {timeit.timeit(test, number=number):.5f}s")


def cprofile_benchmark(args: Namespace, benchmarks: dict[str, Callable[[], Any]]) -> list[Path]:
    """Use cProfile to profile a benchmark."""
    output_profs: list[Path] = []
    for name, _ in get_benchmarks(args, benchmarks):
        output_profs.append(output_prof := args.output / f"{name}.prof")
        cProfile.run(f"{name}()", str(output_prof))
    return output_profs


def viztracer_benchmark(args: Namespace, benchmarks: dict[str, Callable[[], Any]]) -> list[Path]:
    """Use VizTracer to profile a benchmark."""
    output_profs: list[Path] = []
    for name, test in get_benchmarks(args, benchmarks):
        output_profs.append(output_prof := args.output / f"{name}.json")
        with VizTracer(output_file=str(output_prof)):
            test()
    return output_profs


def show(args: Namespace, output_profs: list[Path], tool: str, options: tuple[str, ...] | None = None) -> None:
    """Show the profile using the specified tool."""
    if args.test is None and args.show:
        raise ValueError("Cannot show UI for more than one benchmark")
    assert len(output_profs) == 1
    if options is None:
        options = cast(tuple[str], ())
    command = ["uv", "run", tool, str(output_profs[0]), *options]
    print(command)
    subprocess.run(command, check=True)  # noqa: S603


def profile(benchmarks: dict[str, Callable[[], Any]], argv: list[str] | None = None) -> None:
    """Run the selected profiler."""
    if not benchmarks:
        raise ValueError("At least one benchmark must be provided to profile!")

    args = parse_arguments().parse_args(args=argv)

    match args.command:
        case "timeit":
            timeit_benchmark(args, benchmarks)
        case "snakeviz":
            output_profs = cprofile_benchmark(args, benchmarks)
            show(args, output_profs, tool="snakeviz")
        case "flameprof":
            output_profs = cprofile_benchmark(args, benchmarks)
            show(args, output_profs, tool="flameprof")
        case "viztracer":
            output_profs = viztracer_benchmark(args, benchmarks)
            show(args, output_profs, tool="vizviewer")
        case _:
            raise ValueError("Invalid command!")
