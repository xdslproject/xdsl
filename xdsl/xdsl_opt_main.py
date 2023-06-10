import argparse
import sys
import os

from io import StringIO

from xdsl.ir import Dialect, MLContext
from xdsl.parser import Parser, ParseError
from xdsl.passes import ModulePass
from xdsl.printer import Printer

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import print_assembly, riscv_code

from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.parse_pipeline import parse_pipeline

from typing import IO, Dict, Callable, List, Sequence


def get_all_dialects() -> list[tuple[str, Callable[[], Dialect]]]:
    """Return the list of all available dialects."""

    def get_affine():
        from xdsl.dialects.affine import Affine

        return Affine

    def get_arith():
        from xdsl.dialects.arith import Arith

        return Arith

    def get_builtin():
        from xdsl.dialects.builtin import Builtin

        return Builtin

    def get_cf():
        from xdsl.dialects.cf import Cf

        return Cf

    def get_cmath():
        from xdsl.dialects.cmath import CMath

        return CMath

    def get_dmp():
        from xdsl.dialects.experimental.dmp import DMP

        return DMP

    def get_fir():
        from xdsl.dialects.experimental.fir import FIR

        return FIR

    def get_func():
        from xdsl.dialects.func import Func

        return Func

    def get_gpu():
        from xdsl.dialects.gpu import GPU

        return GPU

    def get_irdl():
        from xdsl.dialects.irdl import IRDL

        return IRDL

    def get_llvm():
        from xdsl.dialects.llvm import LLVM

        return LLVM

    def get_math():
        from xdsl.dialects.experimental.math import Math

        return Math

    def get_memref():
        from xdsl.dialects.memref import MemRef

        return MemRef

    def get_mpi():
        from xdsl.dialects.mpi import MPI

        return MPI

    def get_pdl():
        from xdsl.dialects.pdl import PDL

        return PDL

    def get_riscv():
        from xdsl.dialects.riscv import RISCV

        return RISCV

    def get_riscv_func():
        from xdsl.dialects.riscv_func import RISCV_Func

        return RISCV_Func

    def get_scf():
        from xdsl.dialects.scf import Scf

        return Scf

    def get_snitch():
        from xdsl.dialects.snitch import Snitch

        return Snitch

    def get_snitch_runtime():
        from xdsl.dialects.snitch_runtime import SnitchRuntime

        return SnitchRuntime

    def get_stencil():
        from xdsl.dialects.stencil import Stencil

        return Stencil

    def get_symref():
        from xdsl.frontend.symref import Symref

        return Symref

    def get_test():
        from xdsl.dialects.test import Test

        return Test

    def get_vector():
        from xdsl.dialects.vector import Vector

        return Vector

    return [
        ("affine", get_affine),
        ("arith", get_arith),
        ("builtin", get_builtin),
        ("cf", get_cf),
        ("cmath", get_cmath),
        ("dmp", get_dmp),
        ("fir", get_fir),
        ("func", get_func),
        ("gpu", get_gpu),
        ("irdl", get_irdl),
        ("llvm", get_llvm),
        ("math", get_math),
        ("memref", get_memref),
        ("mpi", get_mpi),
        ("pdl", get_pdl),
        ("riscv", get_riscv),
        ("riscv_func", get_riscv_func),
        ("scf", get_scf),
        ("snitch", get_snitch),
        ("snrt", get_snitch_runtime),
        ("stencil", get_stencil),
        ("symref", get_symref),
        ("test", get_test),
        ("vector", get_vector),
    ]


def get_all_passes() -> list[tuple[str, Callable[[], type[ModulePass]]]]:
    """Return the list of all available passes."""

    def get_convert_stencil_to_llmlir():
        from xdsl.transforms.experimental.ConvertStencilToLLMLIR import (
            ConvertStencilToLLMLIRPass,
        )

        return ConvertStencilToLLMLIRPass

    def get_dead_code_elimination():
        from xdsl.transforms.dead_code_elimination import DeadCodeElimination

        return DeadCodeElimination

    def get_desymrefy():
        from xdsl.frontend.passes.desymref import DesymrefyPass

        return DesymrefyPass

    def get_dmp_scatter_gather():
        from xdsl.transforms.experimental.dmp.scatter_gather import (
            DmpScatterGatherTrivialLowering,
        )

        return DmpScatterGatherTrivialLowering

    def get_dmp_stencil_global_to_local():
        from xdsl.transforms.experimental.dmp.stencil_global_to_local import (
            GlobalStencilToLocalStencil2DHorizontal,
        )

        return GlobalStencilToLocalStencil2DHorizontal

    def get_lower_halo_to_mpi():
        from xdsl.transforms.experimental.dmp.stencil_global_to_local import (
            LowerHaloToMPI,
        )

        return LowerHaloToMPI

    def get_lower_mpi():
        from xdsl.transforms.lower_mpi import LowerMPIPass

        return LowerMPIPass

    def get_lower_riscv_func():
        from xdsl.transforms.lower_riscv_func import LowerRISCVFunc

        return LowerRISCVFunc

    def get_lower_snitch():
        from xdsl.transforms.lower_snitch import LowerSnitchPass

        return LowerSnitchPass

    def get_lower_snitch_runtime():
        from xdsl.transforms.lower_snitch_runtime import LowerSnitchRuntimePass

        return LowerSnitchRuntimePass

    def get_riscv_register_allocation():
        from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation

        return RISCVRegisterAllocation

    def get_stencil_shape_inference():
        from xdsl.transforms.experimental.StencilShapeInference import (
            StencilShapeInferencePass,
        )

        return StencilShapeInferencePass

    return [
        ("convert-stencil-to-ll-mlir", get_convert_stencil_to_llmlir),
        ("dce", get_dead_code_elimination),
        ("frontend-desymrefy", get_desymrefy),
        ("dmp-setup-and-teardown", get_dmp_scatter_gather),
        ("dmp-decompose-2d", get_dmp_stencil_global_to_local),
        ("dmp-to-mpi", get_lower_halo_to_mpi),
        ("lower-mpi", get_lower_mpi),
        ("lower-riscv-func", get_lower_riscv_func),
        ("lower-snitch", get_lower_snitch),
        ("lower-snrt-to-func", get_lower_snitch_runtime),
        ("riscv-allocate-registers", get_riscv_register_allocation),
        ("stencil-shape-inference", get_stencil_shape_inference),
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

    available_passes: Dict[str, Callable[[], type[ModulePass]]]
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
        for dialect_name, dialect_factory in get_all_dialects():
            self.ctx.register_dialect(dialect_name, dialect_factory)

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

    def register_pass(self, pass_name: str, opPass: Callable[[], type[ModulePass]]):
        self.available_passes[pass_name] = opPass

    def register_all_passes(self):
        """
        Register all passes that can be used.

        Add other/additional passes by overloading this function.
        """
        for pass_name, pass_ in get_all_passes():
            self.register_pass(pass_name, pass_)

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
            self.available_passes[p.name]().from_pass_spec(p) for p in pipeline
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
