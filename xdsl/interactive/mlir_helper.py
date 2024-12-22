from collections.abc import Callable
from typing import NamedTuple

from xdsl.context import MLContext
from xdsl.dialects import builtin, get_all_dialects
from xdsl.ir import Dialect
from xdsl.passes import ModulePass, PipelinePass
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.utils.parse_pipeline import PipelinePassSpec, parse_pipeline
from xdsl.parser import Parser
from xdsl.dialects import get_all_dialects


import os
import subprocess

def iterate_and_extract_options() -> list[str]:
    """
    Fetch MLIR passes by calling `mlir-opt --help` and extract options starting with `--`.

    Returns:
        list: A list of strings containing the extracted options.
    """
    options = []

    # Step 1: Call `mlir-opt --help` and capture the output
    try:
        result = subprocess.run(
            ["mlir-opt", "--help"],
            capture_output=True,
            text=True,
            check=True
        )
        help_output = result.stdout
    except FileNotFoundError:
        print("Error: 'mlir-opt' not found. Make sure it is installed and added to your PATH.")
        return []
    except subprocess.CalledProcessError as e:
        print(f"Error while running 'mlir-opt --help': {e}")
        return []

    # Step 2: Parse the options from the output
    for line in help_output.splitlines():
        line = line.strip()
        if '--' in line:
            start_index = line.find('--')
            option = line[start_index:].split()[0]  # Extract the word directly after `--`
            options.append(option)

    return options

def get_mlir_pass_list(
) -> tuple[str, ...]:
    """
    Function that returns the condensed pass list for a given ModuleOp, i.e. the passes that
    change the ModuleOp.
    """

    return tuple(pass_name for pass_name in iterate_and_extract_options())


def get_new_registered_context(
    all_dialects: tuple[tuple[str, Callable[[], Dialect]], ...],
) -> MLContext:
    """
    Generates a new MLContext, registers it and returns it.
    """
    ctx = MLContext()
    for dialect_name, dialect_factory in all_dialects:
        ctx.register_dialect(dialect_name, dialect_factory)
    return ctx


def generate_mlir_pass_spec(args : tuple[str, ...]) -> PipelinePassSpec | None:
    return MLIROptPass("mlir-opt", arguments=args)


def apply_mlir_pass_with_args_to_module(
    module: builtin.ModuleOp,
    ctx: MLContext,
    module_pass: ModulePass
) -> builtin.ModuleOp:
    """
    Function that takes a ModuleOp, an MLContext and a pass_pipeline (consisting of a type[ModulePass] and PipelinePassSpec), applies the pass(es) to the ModuleOp and returns the new ModuleOp.
    """
    module_pass.apply(ctx, module)
    return module
