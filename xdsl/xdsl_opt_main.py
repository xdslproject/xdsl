import argparse
import sys
from io import StringIO
from typing import IO, Callable, Dict, List, Sequence, Type

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import print_assembly, riscv_code
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.printer import Printer
from xdsl.tools.command_line_tool import CommandLineTool, get_all_passes
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.parse_pipeline import parse_pipeline


class xDSLOptMain(CommandLineTool):
    available_passes: Dict[str, Type[ModulePass]]
    """
    A mapping from pass names to functions that apply the pass to a ModuleOp.
    """

    available_targets: Dict[str, Callable[[ModuleOp, IO[str]], None]]
    """
    A mapping from target names to functions that serialize a ModuleOp into a
    stream.
    """

    pipeline: List[ModulePass]
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
            for i, chunk in enumerate(chunks):
                try:
                    if i > 0:
                        output_stream.write("// -----\n")
                    module = self.parse_chunk(chunk, file_extension)
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
            "--print-debuginfo",
            default=False,
            action="store_true",
            help="Print operations with debug info annotation, such as location.",
        )

    def register_pass(self, opPass: Type[ModulePass]):
        self.available_passes[opPass.name] = opPass

    def register_all_passes(self):
        """
        Register all passes that can be used.

        Add other/additional passes by overloading this function.
        """
        for pass_ in get_all_passes():
            self.register_pass(pass_)

    def register_all_targets(self):
        """
        Register all targets that can be used.

        Add other/additional targets by overloading this function.
        """

        def _output_mlir(prog: ModuleOp, output: IO[str]):
            printer = Printer(
                stream=output,
                print_generic_format=self.args.print_op_generic,
                print_debuginfo=self.args.print_debuginfo,
            )
            printer.print_op(prog)
            print("\n", file=output)

        def _output_riscv_asm(prog: ModuleOp, output: IO[str]):
            print_assembly(prog, output)

        def _emulate_riscv(prog: ModuleOp, output: IO[str]):
            # import only if running riscv emulation
            try:
                from xdsl.interpreters.riscv_emulator import RV_Debug, run_riscv
            except ImportError:
                print("Please install optional dependencies to run riscv emulation")
                return

            code = riscv_code(prog)
            RV_Debug.stream = output
            run_riscv(code, unlimited_regs=True, verbosity=0)

        self.available_targets["mlir"] = _output_mlir
        self.available_targets["riscv-asm"] = _output_riscv_asm
        self.available_targets["riscemu"] = _emulate_riscv

    def setup_pipeline(self):
        """
        Creates a pipeline that consists of all the passes specified.

        Fails, if not all passes are registered.
        """
        pipeline = list(parse_pipeline(self.args.passes))

        for p in pipeline:
            if p.name not in self.available_passes:
                raise Exception(f"Unrecognized pass: {p.name}")

        self.pipeline = [
            self.available_passes[p.name].from_pass_spec(p) for p in pipeline
        ]

    def prepare_input(self) -> tuple[List[IO[str]], str]:
        """
        Prepare input by eventually splitting it in chunks. If not set, the parser
        registered for this file extension is used.
        """

        # when using the split input flag, program is split into multiple chunks
        # it's used for split input file

        chunks: List[IO[str]] = []
        f, file_extension = self.get_input_stream()
        chunks = [f]
        if self.args.split_input_file:
            chunks = [StringIO(chunk) for chunk in f.read().split("// -----")]
            f.close()
        if self.args.frontend:
            file_extension = self.args.frontend

        if file_extension not in self.available_frontends:
            for chunk in chunks:
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
            for p in self.pipeline:
                p.apply(self.ctx, prog)
                assert isinstance(prog, ModuleOp)
                if not self.args.disable_verify:
                    prog.verify()
                if self.args.print_between_passes:
                    print(f"IR after {p.name}:")
                    printer = Printer(stream=sys.stdout)
                    printer.print_op(prog)
                    print("\n\n\n")
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
