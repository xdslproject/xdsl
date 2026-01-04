import argparse
import sys
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from importlib.metadata import version
from io import StringIO
from itertools import accumulate
from typing import IO, Any

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Attribute
from xdsl.passes import ModulePass, PassPipeline
from xdsl.printer import Printer
from xdsl.syntax_printer import SyntaxPrinter
from xdsl.tools.command_line_tool import CommandLineTool
from xdsl.universe import Universe
from xdsl.utils.diagnostic import Diagnostic
from xdsl.utils.exceptions import DiagnosticException, ParseError, ShrinkException
from xdsl.utils.lexer import Span


def _empty_post_init(self: Attribute):
    """
    An empty 'post_init' function.
    Used to replace the default 'post_init' on 'Attribute' when verification is disabled.
    """
    pass


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

    pipeline: PassPipeline
    """ The pass-pipeline to be applied. """

    def __init__(
        self,
        description: str = "xDSL modular optimizer driver",
        args: Sequence[str] | None = None,
    ):
        self.available_frontends = {}
        self.available_passes = {}
        self.available_targets = {}

        self.ctx = Context()
        self.register_all_dialects()
        self.register_all_frontends()
        self.register_all_passes()
        self.register_all_targets()

        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        self.register_all_arguments(arg_parser)
        self.args = arg_parser.parse_args(args=args)

        self.ctx.allow_unregistered = self.args.allow_unregistered_dialect

        if self.args.disable_verify:
            Attribute.__post_init__ = _empty_post_init

        if self.args.syntax_highlight:
            Diagnostic.colored = True

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
                except ParseError as e:
                    s = e.span
                    e.span = Span(s.start, s.end, s.input, offset)
                    if self.args.parsing_diagnostics:
                        print(e)
                    else:
                        raise
                except DiagnosticException as e:
                    if self.args.verify_diagnostics:
                        print(e)
                        # __notes__ only in Python 3.11 and above
                        if hasattr(e, "__notes__"):
                            for e in getattr(e, "__notes__"):
                                print(e)
                    else:
                        raise
                finally:
                    chunk.close()
        except ShrinkException:
            assert self.args.shrink
            print("Success, can shrink")
            # Exit with value 0 to let shrinkray know that it can shrink
            exit(0)
        finally:
            if output_stream is not sys.stdout:
                output_stream.close()
        if self.args.shrink:
            print("Failure, can't shrink")
            # Exit with non-0 value to let shrinkray know that it cannot shrink
            exit(1)

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
            help=f"Delimited list of passes. Available passes are: {pass_names}",
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
            help="Split the input file into pieces and process each chunk "
            "independently by using `// -----`",
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
            "--syntax-highlight",
            default=False,
            action="store_true",
            help="Enable printing with syntax highlighting on the terminal.",
        )

        arg_parser.add_argument(
            "-v",
            "--version",
            action=VersionAction,
        )

        arg_parser.add_argument(
            "--shrink",
            default=False,
            action="store_true",
            help="Return success on exit if ShrinkException was raised.",
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
        multiverse = Universe.get_multiverse()
        for pass_name, pass_factory in multiverse.all_passes.items():
            self.register_pass(pass_name, pass_factory)

    def register_all_targets(self):
        """
        Register all targets that can be used.

        Add other/additional targets by overloading this function.
        """

        def _output_arm_asm(prog: ModuleOp, output: IO[str]):
            from xdsl.dialects.arm import print_assembly

            print_assembly(prog, output)

        def _output_mlir(prog: ModuleOp, output: IO[str]):
            cls = SyntaxPrinter if self.args.syntax_highlight else Printer
            printer = cls(
                stream=output,
                print_generic_format=self.args.print_op_generic,
                print_properties_as_attributes=self.args.print_no_properties,
                print_debuginfo=self.args.print_debuginfo,
            )
            printer.print_op(prog)
            printer.print_metadata(self.ctx.loaded_dialects)
            print("\n", file=output)

        def _output_riscv_asm(prog: ModuleOp, output: IO[str]):
            from xdsl.dialects.riscv import print_assembly

            print_assembly(prog, output)

        def _output_x86_asm(prog: ModuleOp, output: IO[str]):
            from xdsl.dialects.x86.ops import print_assembly

            print_assembly(prog, output)

        def _output_wat(prog: ModuleOp, output: IO[str]):
            from xdsl.dialects.wasm import WasmModuleOp
            from xdsl.dialects.wasm.wat import WatPrinter

            for op in prog.walk():
                if isinstance(op, WasmModuleOp):
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

        def _output_csl(prog: ModuleOp, output: IO[str]):
            from xdsl.backend.csl.print_csl import print_to_csl

            print_to_csl(prog, output)

        def _output_wgsl(prog: ModuleOp, output: IO[str]):
            from xdsl.backend.wgsl.wgsl_printer import WGSLPrinter
            from xdsl.dialects import gpu

            for op in prog.ops:
                if isinstance(op, gpu.ModuleOp):
                    printer = WGSLPrinter(stream=output)
                    printer.print(op)

        def _output_air(prog: ModuleOp, output: IO[str]):
            from xdsl.backend.mps.print_mps import print_to_mps

            print_to_mps(prog, output)

        def _output_llvm(prog: ModuleOp, output: IO[str]):
            from xdsl.backend.llvm.convert import convert_module

            llvm_module = convert_module(prog)
            print(llvm_module, file=output)

        self.available_targets["arm-asm"] = _output_arm_asm
        self.available_targets["csl"] = _output_csl
        self.available_targets["mlir"] = _output_mlir
        self.available_targets["riscemu"] = _emulate_riscv
        self.available_targets["riscv-asm"] = _output_riscv_asm
        self.available_targets["wat"] = _output_wat
        self.available_targets["wgsl"] = _output_wgsl
        self.available_targets["mps"] = _output_air
        self.available_targets["x86-asm"] = _output_x86_asm
        self.available_targets["llvm"] = _output_llvm

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

        self.pipeline = PassPipeline.parse_spec(
            self.available_passes,
            self.args.passes,
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
            raise ValueError(f"Unrecognized file extension '{file_extension}'")

        return chunks, file_extension

    def prepare_output(self) -> IO[str]:
        if self.args.output_file is None:
            return sys.stdout
        else:
            return open(self.args.output_file, "w")

    def apply_passes(self, prog: ModuleOp) -> bool:
        """Apply passes in order."""
        if not self.args.disable_verify:
            prog.verify()
        self.pipeline.apply(self.ctx, prog)
        if not self.args.disable_verify:
            prog.verify()
        return True

    def output_resulting_program(self, prog: ModuleOp) -> str:
        """Get the resulting program."""
        output = StringIO()
        self.available_targets[self.args.target](prog, output)
        return output.getvalue()


class VersionAction(argparse.Action):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(nargs=0, *args, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        print(f"xdsl-opt built from xdsl version {version('xdsl')}\n")
        parser.exit()
