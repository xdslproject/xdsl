import argparse
import sys
import os
from io import IOBase, StringIO

from xdsl.ir import MLContext
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.func import Func
from xdsl.dialects.scf import Scf
from xdsl.dialects.arith import Arith
from xdsl.dialects.affine import Affine
from xdsl.dialects.memref import MemRef
from xdsl.dialects.builtin import ModuleOp, Builtin
from xdsl.dialects.cmath import CMath
from xdsl.dialects.cf import Cf
from xdsl.diagnostic import DiagnosticException
from xdsl.dialects.llvm import LLVM
from xdsl.dialects.irdl import IRDL
from xdsl.irdl_mlir_printer import IRDLPrinter

from typing import Dict, Callable, List


class xDSLOptMain:
    ctx: MLContext
    args: argparse.Namespace
    """
    The argument parsers namespace which holds the parsed commandline
    attributes.
    """

    available_frontends: Dict[str, Callable[[IOBase], ModuleOp]] = {}
    """
    A mapping from file extension to a frontend that can handle this
    file type.
    """

    available_passes: Dict[str, Callable[[MLContext, ModuleOp], None]] = {}
    """
    A mapping from pass names to functions that apply the pass to a  ModuleOp.
    """

    available_targets: Dict[str, Callable[[ModuleOp, IOBase], None]] = {}
    """
    A mapping from target names to functions that serialize a ModuleOp into a
    stream.
    """

    pipeline: List[tuple[str, Callable[[ModuleOp], None]]]
    """ The pass-pipeline to be applied. """

    def __init__(self, description: str = 'xDSL modular optimizer driver'):
        self.ctx = MLContext()
        self.register_all_dialects()
        self.register_all_frontends()
        self.register_all_passes()
        self.register_all_targets()

        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        self.register_all_arguments(arg_parser)
        self.args = arg_parser.parse_args()

        self.setup_pipeline()

    def run(self):
        """
        Executes the different steps.
        """
        module = self.parse_input()
        if not self.args.verify_diagnostics:
            self.apply_passes(module)
        else:
            try:
                self.apply_passes(module)
            except DiagnosticException as e:
                print(e)
                exit(0)

        contents = self.output_resulting_program(module)
        self.print_to_output_stream(contents)

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        """
        Registers all the command line arguments that are used by this tool.

        Add other/additional arguments by overloading this function.
        """
        arg_parser.add_argument("input_file",
                                type=str,
                                nargs="?",
                                help="path to input file")

        targets = [name for name in self.available_targets]
        arg_parser.add_argument("-t",
                                "--target",
                                type=str,
                                required=False,
                                choices=targets,
                                help="target",
                                default="xdsl")

        frontends = [name for name in self.available_frontends]
        arg_parser.add_argument(
            "-f",
            "--frontend",
            type=str,
            required=False,
            choices=frontends,
            help="Frontend to be used for the input. If not set, "
            "the xdsl frontend or the one for the file extension "
            "is used.")

        arg_parser.add_argument("--disable-verify",
                                default=False,
                                action='store_true')
        arg_parser.add_argument("-o",
                                "--output-file",
                                type=str,
                                required=False,
                                help="path to output file")

        pass_names = ",".join([name for name in self.available_passes])
        arg_parser.add_argument("-p",
                                "--passes",
                                required=False,
                                help="Delimited list of passes."
                                f" Available passes are: {pass_names}",
                                type=str,
                                default="")

        arg_parser.add_argument("--print-between-passes",
                                default=False,
                                action='store_true',
                                help="Print the IR between each pass")

        arg_parser.add_argument("--verify-diagnostics",
                                default=False,
                                action='store_true',
                                help="Prints the content of a triggered "
                                "exception and exits with code 0")

        arg_parser.add_argument(
            "--use-mlir-bindings",
            default=False,
            action='store_true',
            help="Use the MLIR bindings for printing MLIR. "
            "This requires the MLIR Python bindings to be installed.")

        arg_parser.add_argument(
            "--allow-unregistered-ops",
            default=False,
            action='store_true',
            help="Allow the parsing of unregistered operations.")

    def register_all_dialects(self):
        """
        Register all dialects that can be used.

        Add other/additional dialects by overloading this function.
        """
        _ = Builtin(self.ctx)
        _ = Func(self.ctx)
        _ = Arith(self.ctx)
        _ = MemRef(self.ctx)
        _ = Affine(self.ctx)
        _ = Scf(self.ctx)
        _ = Cf(self.ctx)
        _ = CMath(self.ctx)
        _ = IRDL(self.ctx)
        _ = LLVM(self.ctx)

    def register_all_frontends(self):
        """
        Register all frontends that can be used.

        Add other/additional frontends by overloading this function.
        """

        def parse_xdsl(f: IOBase):
            input_str = f.read()
            parser = Parser(
                self.ctx,
                input_str,
                allow_unregistered_ops=self.args.allow_unregistered_ops)
            module = parser.parse_op()
            if not (isinstance(module, ModuleOp)):
                raise Exception(
                    "Expected module or program as toplevel operation")
            return module

        def parse_mlir(f: IOBase):
            input_str = f.read()
            parser = Parser(
                self.ctx,
                input_str,
                source=Parser.Source.MLIR,
                allow_unregistered_ops=self.args.allow_unregistered_ops)
            module = parser.parse_op()
            if not (isinstance(module, ModuleOp)):
                raise Exception(
                    "Expected module or program as toplevel operation")
            return module

        self.available_frontends['xdsl'] = parse_xdsl
        self.available_frontends['mlir'] = parse_mlir

    def register_all_passes(self):
        """
        Register all passes that can be used.

        Add other/additional passes by overloading this function.
        """
        pass

    def register_all_targets(self):
        """
        Register all targets that can be used.

        Add other/additional targets by overloading this function.
        """

        def _output_xdsl(prog: ModuleOp, output: IOBase):
            printer = Printer(stream=output)
            printer.print_op(prog)

        def _output_mlir(prog: ModuleOp, output: IOBase):
            if self.args.use_mlir_bindings:
                from xdsl.mlir_converter import MLIRConverter
                converter = MLIRConverter(self.ctx)
                mlir_module = converter.convert_module(prog)
                print(mlir_module, file=output)
            else:
                printer = Printer(stream=output, target=Printer.Target.MLIR)
                printer.print_op(prog)

        def _output_irdl(prog: ModuleOp, output: IOBase):
            irdl_to_mlir = IRDLPrinter(stream=output)
            irdl_to_mlir.print_module(prog)

        self.available_targets['xdsl'] = _output_xdsl
        self.available_targets['irdl'] = _output_irdl
        self.available_targets['mlir'] = _output_mlir

    def setup_pipeline(self):
        """
        Creates a pipeline that consists of all the passes specified.

        Failes, if not all passes are registered.
        """
        pipeline = [
            str(item) for item in self.args.passes.split(',') if len(item) > 0
        ]

        for p in pipeline:
            if p not in self.available_passes:
                raise Exception(f"Unrecognized pass: {p}")

        self.pipeline = [(p, lambda op, p=p: self.available_passes[p]
                          (self.ctx, op)) for p in pipeline]

    def parse_input(self) -> ModuleOp:
        """
        Parse the input file by invoking the parser specified by the `parser`
        argument. If not set, the parser registered for this file extension
        is used.
        """
        if self.args.input_file is None:
            f = sys.stdin
            file_extension = 'xdsl'
        else:
            f = open(self.args.input_file)
            _, file_extension = os.path.splitext(self.args.input_file)
            file_extension = file_extension.replace(".", "")

        if self.args.frontend is not None:
            file_extension = self.args.frontend

        if file_extension not in self.available_frontends:
            raise Exception(f"Unrecognized file extension '{file_extension}'")

        return self.available_frontends[file_extension](f)

    def apply_passes(self, prog: ModuleOp):
        """Apply passes in order."""
        assert isinstance(prog, ModuleOp)
        if not self.args.disable_verify:
            prog.verify()
        for pass_name, p in self.pipeline:
            p(prog)
            assert isinstance(prog, ModuleOp)
            if not self.args.disable_verify:
                prog.verify()
            if self.args.print_between_passes:
                print(f"IR after {pass_name}:")
                printer = Printer(stream=sys.stdout)
                printer.print_op(prog)
                print("\n\n")

    def output_resulting_program(self, prog: ModuleOp) -> str:
        """Get the resulting program."""
        output = StringIO()
        if self.args.target not in self.available_targets:
            raise Exception(f"Unknown target {self.args.target}")

        self.available_targets[self.args.target](prog, output)
        return output.getvalue()

    def print_to_output_stream(self, contents: str):
        """Print the contents in the expected stream."""
        if self.args.output_file is None:
            print(contents)
        else:
            output_stream = open(self.args.output_file, 'w')
            output_stream.write(contents)
