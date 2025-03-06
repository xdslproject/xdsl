"""Visualisations for benchmark data."""

import time

from benchmarks.bench_utils import warmed_timeit
from benchmarks.lexer import Lexer
from benchmarks.parser import Parser
from benchmarks.printer import Printer
from benchmarks.rewriting import PatternRewriter
from benchmarks.verifier import Verifier
from benchmarks.workloads import WorkloadBuilder
from xdsl.context import Context
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.parser import Parser as XdslParser
from xdsl.transforms.canonicalize import CanonicalizePass

LEXER = Lexer()
PARSER = Parser()
PATTERN_REWRITER = PatternRewriter()
VERIFIER = Verifier()
PRINTER = Printer()


def draw_comparison_chart() -> None:
    """Compare the pipeline phase times for a workload."""

    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "grid.alpha": 0.7,
            "grid.linestyle": "--",
            "figure.dpi": 100,
            "font.family": "Menlo",
        }
    )
    plt.title("Pipeline phase times for constant folding 100 items")

    phase_functions = {
        "Lexing": LEXER.time_constant_100,
        "Lexing + Parsing": PARSER.time_constant_100,
        "Rewriting": PATTERN_REWRITER.time_constant_folding_100,
        "Verifying": VERIFIER.time_constant_100,
        "Printing": PRINTER.time_constant_100,
    }

    raw_phase_times = {
        name: warmed_timeit(func) for name, func in phase_functions.items()
    }
    # print(raw_phase_times)
    phase_means = {
        "Lexing": raw_phase_times["Lexing"][0],
        "Parsing": raw_phase_times["Lexing + Parsing"][0]
        - raw_phase_times["Lexing"][0],
        "Rewriting": raw_phase_times["Rewriting"][0],
        "Verifying": raw_phase_times["Verifying"][0],
        "Printing": raw_phase_times["Printing"][0],
    }
    phase_medians = {
        "Lexing": raw_phase_times["Lexing"][1],
        "Parsing": raw_phase_times["Lexing + Parsing"][1]
        - raw_phase_times["Lexing"][1],
        "Rewriting": raw_phase_times["Rewriting"][1],
        "Verifying": raw_phase_times["Verifying"][1],
        "Printing": raw_phase_times["Printing"][1],
    }
    phase_errors = {
        "Lexing": raw_phase_times["Lexing"][2],
        "Parsing": raw_phase_times["Lexing + Parsing"][2]
        + raw_phase_times["Lexing"][2],
        "Rewriting": raw_phase_times["Rewriting"][2],
        "Verifying": raw_phase_times["Verifying"][2],
        "Printing": raw_phase_times["Printing"][2],
    }

    for name in phase_means:
        if name in phase_errors:
            print(
                f"{name}: {phase_means[name]:.3g} ± {phase_errors[name]:.3g}s (median {phase_medians[name]:.3g})"
            )
    print(f"Total: {sum(phase_means.values()):.3g} ± {sum(phase_errors.values()):.3g}s")

    plt.bar(
        phase_means.keys(),
        phase_means.values(),
        yerr=[phase_errors[name] for name in phase_means],
        error_kw={"capsize": 5},
        label="mean",
    )
    plt.scatter(
        phase_medians.keys(),
        phase_medians.values(),
        label="median",
    )
    plt.xlabel("Workload", fontweight="bold")
    plt.ylabel("Time [s]", fontweight="bold")
    plt.grid(axis="y")
    plt.legend()
    plt.savefig(f"/Users/edjg/Desktop/ubenchmarks/out{int(time.time())}.pdf", dpi=300)
    plt.show()


def draw_scaling_plot() -> None:
    """Plot pattern rewrite scaling."""

    def parse_module(contents: str) -> ModuleOp:
        """Parse a MLIR file as a module."""
        parser = XdslParser(CTX, contents)
        return parser.parse_module()

    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "grid.alpha": 0.7,
            "grid.linestyle": "--",
            "figure.dpi": 100,
            "font.family": "Menlo",
        }
    )
    plt.title("Pattern rewrite scaling")

    CTX = Context(allow_unregistered=True)
    CTX.load_dialect(Arith)
    CTX.load_dialect(Builtin)

    SIZES = (
        10,
        100,
        1000,
        10000,
    )
    constant_fold_workloads = {
        size: parse_module(WorkloadBuilder.constant_folding(size)) for size in SIZES
    }
    constant_fold_functions = {
        str(size): lambda: CanonicalizePass().apply(
            CTX, constant_fold_workloads[size].clone()
        )
        for size in SIZES
    }

    print("Starting")
    warmed_timeit(list(constant_fold_functions.values())[3])
    constant_fold_times = {
        name: warmed_timeit(func) for name, func in constant_fold_functions.items()
    }
    # print(constant_fold_times)
    constant_fold_means = {
        name: times[0] for name, times in constant_fold_times.items()
    }
    constant_fold_errors = {
        name: times[2] for name, times in constant_fold_times.items()
    }

    for name in constant_fold_means:
        if name in constant_fold_errors:
            print(
                f"{name}: {constant_fold_means[name]:.3g} ± {constant_fold_errors[name]:.3g}s"
            )

    plt.errorbar(
        constant_fold_means.keys(),
        constant_fold_means.values(),
        yerr=[constant_fold_errors[name] for name in constant_fold_errors],
        capsize=5,
        label="Constant folding",
    )
    plt.xlabel("Workload size [operations]", fontweight="bold")
    plt.ylabel("Time [s]", fontweight="bold")
    plt.grid(axis="y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # draw_comparison_chart()
    draw_scaling_plot()
