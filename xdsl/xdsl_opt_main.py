import argparse
import sys
import os

from io import StringIO
from xdsl.frontend.symref import Symref

from xdsl.ir import Dialect, MLContext
from xdsl.parser import Parser, ParseError
from xdsl.passes import ModulePass
from xdsl.printer import Printer
from xdsl.dialects.func import Func
from xdsl.dialects.scf import Scf
from xdsl.dialects.affine import Affine
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import ModuleOp, Builtin
from xdsl.dialects.cmath import CMath
from xdsl.dialects.cf import Cf
from xdsl.dialects.vector import Vector
from xdsl.dialects.memref import MemRef
from xdsl.dialects.llvm import LLVM
from xdsl.dialects.mpi import MPI
from xdsl.dialects.gpu import GPU
from xdsl.dialects.pdl import PDL
from xdsl.dialects.test import Test
from xdsl.dialects.stencil import Stencil
from xdsl.dialects.riscv_func import RISCV_Func
from xdsl.dialects.irdl import IRDL
from xdsl.dialects.riscv import RISCV, print_assembly, riscv_code
from xdsl.dialects.snitch import Snitch
from xdsl.dialects.snitch_runtime import SnitchRuntime
from xdsl.dialects.print import Print

from xdsl.dialects.experimental.math import Math
from xdsl.dialects.experimental.fir import FIR
from xdsl.dialects.experimental.dmp import DMP

from xdsl.frontend.passes.desymref import DesymrefyPass
from xdsl.transforms.dead_code_elimination import DeadCodeElimination
from xdsl.transforms.experimental.stencil_storage_materialization import (
    StencilStorageMaterializationPass,
)
from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation
from xdsl.transforms.lower_riscv_func import LowerRISCVFunc
from xdsl.transforms.cf_to_riscv import CfToRISCV
from xdsl.transforms.lower_mpi import LowerMPIPass
from xdsl.transforms.lower_snitch import LowerSnitchPass
from xdsl.transforms.lower_snitch_runtime import LowerSnitchRuntimePass
from xdsl.transforms.experimental.convert_stencil_to_ll_mlir import (
    ConvertStencilToLLMLIRPass,
)
from xdsl.transforms.experimental.stencil_shape_inference import (
    StencilShapeInferencePass,
)
from xdsl.transforms.experimental.dmp.stencil_global_to_local import (
    GlobalStencilToLocalStencil2DHorizontal,
    LowerHaloToMPI,
)
from xdsl.transforms.experimental.dmp.scatter_gather import (
    DmpScatterGatherTrivialLowering,
)
from xdsl.transforms.print_to_println import PrintToPrintf

from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.parse_pipeline import parse_pipeline

from typing import IO, Dict, Callable, List, Sequence, Type


def get_all_dialects() -> list[Dialect]:
    """Return the list of all available dialects."""
    return [
        Affine,
        Arith,
        Builtin,
        Cf,
        CMath,
        DMP,
        FIR,
        Func,
        GPU,
        IRDL,
        LLVM,
        Math,
        MemRef,
        MPI,
        PDL,
        Print,
        RISCV,
        RISCV_Func,
        Scf,
        Snitch,
        SnitchRuntime,
        Stencil,
        Symref,
        Test,
        Vector,
    ]


def get_all_passes() -> list[type[ModulePass]]:
    """Return the list of all available passes."""
    return [
        CfToRISCV,
        ConvertStencilToLLMLIRPass,
        DeadCodeElimination,
        DesymrefyPass,
        DmpScatterGatherTrivialLowering,
        GlobalStencilToLocalStencil2DHorizontal,
        LowerHaloToMPI,
        LowerMPIPass,
        LowerRISCVFunc,
        LowerSnitchPass,
        LowerSnitchRuntimePass,
        PrintToPrintf,
        RISCVRegisterAllocation,
        StencilShapeInferencePass,
        StencilStorageMaterializationPass,
    ]


class xDSLOptMain:
    ctx: MLContext
    args: argparse.Namespace
    """
    The argument parsers namespace which holds the parsed commandline
    attributes.
    """

    available_frontends: Dict[str, Callable[[IO[str]], ModuleOp]]
    """
    A mapping from file extension to a frontend that can handle this
    file type.
    """

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
        arg_parser.add_argument(
            "input_file", type=str, nargs="?", help="path to input file"
        )

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

        frontends = [name for name in self.available_frontends]
        arg_parser.add_argument(
            "-f",
            "--frontend",
            type=str,
            required=False,
            choices=frontends,
            help="Frontend to be used for the input. If not set, "
            "the xdsl frontend or the one for the file extension "
            "is used.",
        )

        arg_parser.add_argument("--disable-verify", default=False, action="store_true")
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
            "--allow-unregistered-dialect",
            default=False,
            action="store_true",
            help="Allow the parsing of unregistered dialects.",
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

    def register_all_dialects(self):
        """
        Register all dialects that can be used.

        Add other/additional dialects by overloading this function.
        """
        for dialect in get_all_dialects():
            self.ctx.register_dialect(dialect)

    def register_all_frontends(self):
        """
        Register all frontends that can be used.

        Add other/additional frontends by overloading this function.
        """

        def parse_mlir(io: IO[str]):
            return Parser(
                self.ctx,
                io.read(),
                self.get_input_name(),
                self.args.allow_unregistered_dialect,
            ).parse_module()

        self.available_frontends["mlir"] = parse_mlir

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
                stream=output, print_generic_format=self.args.print_op_generic
            )
            printer.print_op(prog)
            print("\n", file=output)

        def _output_riscv_asm(prog: ModuleOp, output: IO[str]):
            print_assembly(prog, output)

        def _emulate_riscv(prog: ModuleOp, output: IO[str]):
            # import only if running riscv emulation
            try:
                from xdsl.interpreters.riscv_emulator import run_riscv, RV_Debug
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
        if self.args.input_file is None:
            f = sys.stdin
            file_extension = "mlir"
        else:
            f = open(self.args.input_file)
            _, file_extension = os.path.splitext(self.args.input_file)
            file_extension = file_extension.replace(".", "")

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

    def parse_chunk(self, chunk: IO[str], file_extension: str) -> ModuleOp | None:
        """
        Parse the input file by invoking the parser specified by the `parser`
        argument. If not set, the parser registered for this file extension
        is used.
        """

        try:
            return self.available_frontends[file_extension](chunk)
        except ParseError as e:
            if self.args.parsing_diagnostics:
                print(e)
            else:
                raise e
        finally:
            chunk.close()

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

    def get_input_name(self):
        return self.args.input_file or "stdin"
