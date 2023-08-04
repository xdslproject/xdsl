import argparse
import os
import sys
from typing import IO, Callable

from xdsl.backend.riscv.lowering import scf_to_riscv_scf
from xdsl.backend.riscv.lowering.lower_func_riscv_func import LowerFuncToRiscvFunc
from xdsl.backend.riscv.lowering.optimisation_riscv import OptimiseRiscvPass
from xdsl.backend.riscv.lowering.riscv_arith_lowering import RISCVLowerArith
from xdsl.dialects.affine import Affine
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.cf import Cf
from xdsl.dialects.cmath import CMath
from xdsl.dialects.experimental.dmp import DMP
from xdsl.dialects.experimental.fir import FIR
from xdsl.dialects.experimental.math import Math
from xdsl.dialects.func import Func
from xdsl.dialects.gpu import GPU
from xdsl.dialects.irdl.irdl import IRDL
from xdsl.dialects.linalg import Linalg
from xdsl.dialects.llvm import LLVM
from xdsl.dialects.memref import MemRef
from xdsl.dialects.mpi import MPI
from xdsl.dialects.pdl import PDL
from xdsl.dialects.printf import Printf
from xdsl.dialects.riscv import RISCV
from xdsl.dialects.riscv_func import RISCV_Func
from xdsl.dialects.riscv_scf import RISCV_Scf
from xdsl.dialects.scf import Scf
from xdsl.dialects.snitch import Snitch
from xdsl.dialects.snitch_runtime import SnitchRuntime
from xdsl.dialects.stencil import Stencil
from xdsl.dialects.test import Test
from xdsl.dialects.vector import Vector
from xdsl.frontend.passes.desymref import DesymrefyPass
from xdsl.frontend.symref import Symref
from xdsl.ir import Dialect, MLContext
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.transforms import (
    canonicalize_dmp,
    dead_code_elimination,
    lower_mpi,
    lower_riscv_func,
    lower_snitch,
    lower_snitch_runtime,
    mlir_opt,
    printf_to_llvm,
    reconcile_unrealized_casts,
    riscv_register_allocation,
)
from xdsl.transforms.experimental import (
    convert_stencil_to_ll_mlir,
    stencil_shape_inference,
    stencil_storage_materialization,
)
from xdsl.transforms.experimental.dmp import stencil_global_to_local
from xdsl.utils.exceptions import ParseError


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
        Linalg,
        IRDL,
        LLVM,
        Math,
        MemRef,
        MPI,
        PDL,
        Printf,
        RISCV,
        RISCV_Func,
        RISCV_Scf,
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
        canonicalize_dmp.CanonicalizeDmpPass,
        convert_stencil_to_ll_mlir.ConvertStencilToLLMLIRPass,
        dead_code_elimination.DeadCodeElimination,
        DesymrefyPass,
        stencil_global_to_local.GlobalStencilToLocalStencil2DHorizontal,
        stencil_global_to_local.LowerHaloToMPI,
        lower_mpi.LowerMPIPass,
        lower_riscv_func.LowerRISCVFunc,
        lower_snitch.LowerSnitchPass,
        lower_snitch_runtime.LowerSnitchRuntimePass,
        mlir_opt.MLIROptPass,
        printf_to_llvm.PrintfToLLVM,
        riscv_register_allocation.RISCVRegisterAllocation,
        RISCVLowerArith,
        LowerFuncToRiscvFunc,
        OptimiseRiscvPass,
        scf_to_riscv_scf.ScfToRiscvPass,
        stencil_shape_inference.StencilShapeInferencePass,
        stencil_storage_materialization.StencilStorageMaterializationPass,
        reconcile_unrealized_casts.ReconcileUnrealizedCastsPass,
    ]


class CommandLineTool:
    ctx: MLContext
    args: argparse.Namespace
    """
    The argument parsers namespace which holds the parsed commandline
    attributes.
    """

    available_frontends: dict[str, Callable[[IO[str]], ModuleOp]]
    """
    A mapping from file extension to a frontend that can handle this
    file type.
    """

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        arg_parser.add_argument(
            "input_file", type=str, nargs="?", help="path to input file"
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
            "--allow-unregistered-dialect",
            default=False,
            action="store_true",
            help="Allow the parsing of unregistered dialects.",
        )

        arg_parser.add_argument(
            "--no-implicit-module",
            default=False,
            action="store_true",
            help="Disable implicit addition of a top-level module op during parsing.",
        )

    def get_input_stream(self) -> tuple[IO[str], str]:
        """
        Get the input stream to parse from, along with the file extension.
        """
        if self.args.input_file is None:
            f = sys.stdin
            file_extension = "mlir"
        else:
            f = open(self.args.input_file)
            _, file_extension = os.path.splitext(self.args.input_file)
            file_extension = file_extension.replace(".", "")
        return f, file_extension

    def get_input_name(self):
        return self.args.input_file or "stdin"

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
            ).parse_module(not self.args.no_implicit_module)

        self.available_frontends["mlir"] = parse_mlir

    def parse_chunk(self, chunk: IO[str], file_extension: str) -> ModuleOp | None:
        """
        Parse the input file by invoking the parser specified by the `parser`
        argument. If not set, the parser registered for this file extension
        is used.
        """

        try:
            return self.available_frontends[file_extension](chunk)
        except ParseError as e:
            if "parsing_diagnostics" in self.args and self.args.parsing_diagnostics:
                print(e.with_context())
            else:
                raise Exception("Failed to parse:\n" + e.with_context()) from e
        finally:
            chunk.close()
