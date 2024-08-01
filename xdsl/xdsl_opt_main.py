import argparse
import sys
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from importlib.metadata import version
from io import StringIO
from itertools import accumulate
from typing import IO

from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass, PipelinePass
from xdsl.printer import Printer
from xdsl.tools.command_line_tool import CommandLineTool, get_all_passes
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.parse_pipeline import parse_pipeline


class xDSLOptMain(CommandLineTool):
    available_passes: dict[str, Callable[[], type[ModulePass]]]
    """
    A mapping from pass names to functions that apply the pass to a ModuleOp.
    """

    available_targets: dict[str, Callable[[ModuleOp, IO[str]], None]]
    """
    A mapping from target names to functions that serialize a ModuleOp into a
    stream.
    """

    pipeline: PipelinePass
    """ The pass-pipeline to be applied. """

    def __init__(
        self,
        description: str = "xDSL modular optimizer driver",
        args: Sequence[str] | None = None,
    ):
        self.available_frontends = {}
        self.available_passes = {}
        self.available_targets = {}

        self.ctx = MLContext()
        self.register_all_dialects()
        self.register_all_frontends()
        self.register_all_passes()
        self.register_all_targets()

        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        self.register_all_arguments(arg_parser)
        self.args = arg_parser.parse_args(args=args)

        self.ctx.allow_unregistered = self.args.allow_unregistered_dialect

        self.setup_pipeline()

    def run(self):
        """
        Executes the different steps.
        """
        chunks, file_extension = self.prepare_input()
        output_stream = self.prepare_output()
        try:
            for i, (chunk, offset) in enumerate(chunks):
                try:
                    if i > 0:
                        output_stream.write("// -----\n")
                    module = self.parse_chunk(chunk, file_extension, offset)

                    if module is not None:
                        if self.apply_passes(module):
                            output_stream.write(self.output_resulting_program(module))
                    output_stream.flush()
                finally:
                    chunk.close()
        finally:
            if output_stream is not sys.stdout:
                output_stream.close()

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        """
        Registers all the command line arguments that are used by this tool.

        Add other/additional arguments by overloading this function.
        """
        super().register_all_arguments(arg_parser)

        targets = [name for name in self.available_targets]
        arg_parser.add_argument(
            "-t",
            "--target",
            type=str,
            required=False,
            choices=targets,
            help="target",
            default="mlir",
        )

        arg_parser.add_argument(
            "-o", "--output-file", type=str, required=False, help="path to output file"
        )

        pass_names = ",".join([name for name in self.available_passes])
        arg_parser.add_argument(
            "-p",
            "--passes",
            required=False,
            help="Delimited list of passes." f" Available passes are: {pass_names}",
            type=str,
            default="",
        )

        arg_parser.add_argument(
            "--print-between-passes",
            default=False,
            action="store_true",
            help="Print the IR between each pass",
        )

        arg_parser.add_argument(
            "--verify-diagnostics",
            default=False,
            action="store_true",
            help="Prints the content of a triggered "
            "verifier exception and exits with code 0",
        )

        arg_parser.add_argument(
            "--parsing-diagnostics",
            default=False,
            action="store_true",
            help="Prints the content of a triggered "
            "parsing exception and exits with code 0",
        )

        arg_parser.add_argument(
            "--split-input-file",
            default=False,
            action="store_true",
            help="Split the input file into pieces and process each chunk independently by "
            " using `// -----`",
        )

        arg_parser.add_argument(
            "--print-op-generic",
            default=False,
            action="store_true",
            help="Print operations with the generic format",
        )

        arg_parser.add_argument(
            "--print-no-properties",
            default=False,
            action="store_true",
            help="Print properties as if they were attributes for retrocompatibility.",
        )

        arg_parser.add_argument(
            "--print-debuginfo",
            default=False,
            action="store_true",
            help="Print operations with debug info annotation, such as location.",
        )

        arg_parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"xdsl-opt built from xdsl version {version('xdsl')}\n",
        )

    def register_pass(
        self, pass_name: str, pass_factory: Callable[[], type[ModulePass]]
    ):
        self.available_passes[pass_name] = pass_factory

    def register_all_passes(self):
        """
        Register all passes that can be used.

        Add other/additional passes by overloading this function.
        """
        for pass_name, pass_factory in get_all_passes().items():
            self.register_pass(pass_name, pass_factory)

    def register_all_targets(self):
        """
        Register all targets that can be used.

        Add other/additional targets by overloading this function.
        """

        def _output_mlir(prog: ModuleOp, output: IO[str]):
            printer = Printer(
                stream=output,
                print_generic_format=self.args.print_op_generic,
                print_properties_as_attributes=self.args.print_no_properties,
                print_debuginfo=self.args.print_debuginfo,
            )
            printer.print_op(prog)
            print("\n", file=output)

        def _output_riscv_asm(prog: ModuleOp, output: IO[str]):
            from xdsl.dialects.riscv import print_assembly

            print_assembly(prog, output)

        def _output_x86_asm(prog: ModuleOp, output: IO[str]):
            from xdsl.dialects.x86.ops import print_assembly

            print_assembly(prog, output)

        def _output_wat(prog: ModuleOp, output: IO[str]):
            from xdsl.dialects.wasm import WasmModule
            from xdsl.dialects.wasm.wat import WatPrinter

            for op in prog.walk():
                if isinstance(op, WasmModule):
                    printer = WatPrinter(output)
                    op.print_wat(printer)
                    print("", file=output)

        def _emulate_riscv(prog: ModuleOp, output: IO[str]):
            # import only if running riscv emulation
            try:
                from xdsl.interpreters.riscv_emulator import run_riscv
            except ImportError:
                print("Please install optional dependencies to run riscv emulation")
                return

            from xdsl.dialects.riscv import riscv_code

            code = riscv_code(prog)
            with redirect_stdout(output):
                run_riscv(code, unlimited_regs=True, verbosity=0)

        def _print_to_csl(prog: ModuleOp, output: IO[str]):
            from xdsl.backend.csl.print_csl import print_to_csl

            print_to_csl(prog, output)

        self.available_targets["mlir"] = _output_mlir
        self.available_targets["riscv-asm"] = _output_riscv_asm
        self.available_targets["x86-asm"] = _output_x86_asm
        self.available_targets["riscemu"] = _emulate_riscv
        self.available_targets["wat"] = _output_wat
        self.available_targets["csl"] = _print_to_csl

    def setup_pipeline(self):
        """
        Creates a pipeline that consists of all the passes specified.

        Fails, if not all passes are registered.
        """

        def callback(
            previous_pass: ModulePass, module: ModuleOp, next_pass: ModulePass
        ) -> None:
            if not self.args.disable_verify:
                module.verify()
            if self.args.print_between_passes:
                print(f"IR after {previous_pass.name}:")
                printer = Printer(stream=sys.stdout)
                printer.print_op(module)
                print("\n\n\n")

        self.pipeline = PipelinePass(
            tuple(
                pass_type.from_pass_spec(spec)
                for pass_type, spec in PipelinePass.build_pipeline_tuples(
                    self.available_passes, parse_pipeline(self.args.passes)
                )
            ),
            callback,
        )

    def prepare_input(self) -> tuple[list[tuple[IO[str], int]], str]:
        """
        Prepare input by eventually splitting it in chunks. If not set, the parser
        registered for this file extension is used.
        """

        # when using the split input flag, program is split into multiple chunks
        # it's used for split input file

        chunks: list[tuple[IO[str], int]] = []
        f, file_extension = self.get_input_stream()
        chunks = [(f, 0)]
        if self.args.split_input_file:
            chunks_str = [chunk for chunk in f.read().split("// -----")]
            chunks_off = accumulate(
                [0, *[chunk.count("\n") for chunk in chunks_str[:-1]]]
            )
            chunks = [
                (StringIO(chunk), off)
                for chunk, off in zip(chunks_str, chunks_off, strict=True)
            ]
            f.close()
        if self.args.frontend:
            file_extension = self.args.frontend

        if file_extension not in self.available_frontends:
            for chunk, _ in chunks:
                chunk.close()
            raise Exception(f"Unrecognized file extension '{file_extension}'")

        return chunks, file_extension

    def prepare_output(self) -> IO[str]:
        if self.args.output_file is None:
            return sys.stdout
        else:
            return open(self.args.output_file, "w")

    def apply_passes(self, prog: ModuleOp) -> bool:
        """Apply passes in order."""
        try:
            assert isinstance(prog, ModuleOp)
            if not self.args.disable_verify:
                prog.verify()
            self.pipeline.apply(self.ctx, prog)
            if not self.args.disable_verify:
                prog.verify()
        except DiagnosticException as e:
            if self.args.verify_diagnostics:
                print(e)
                return False
            else:
                raise e
        return True

    def output_resulting_program(self, prog: ModuleOp) -> str:
        """Get the resulting program."""
        output = StringIO()
        if self.args.target not in self.available_targets:
            raise Exception(f"Unknown target {self.args.target}")

        self.available_targets[self.args.target](prog, output)
        return output.getvalue()
