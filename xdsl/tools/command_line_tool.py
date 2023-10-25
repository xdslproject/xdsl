import argparse
import os
import sys
from collections.abc import Callable
from typing import IO

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Dialect, MLContext
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import ParseError


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

    def get_comb():
        from xdsl.dialects.comb import Comb

        return Comb

    def get_dmp():
        from xdsl.dialects.experimental.dmp import DMP

        return DMP

    def get_fir():
        from xdsl.dialects.experimental.fir import FIR

        return FIR

    def get_fsm():
        from xdsl.dialects.fsm import FSM

        return FSM

    def get_func():
        from xdsl.dialects.func import Func

        return Func

    def get_gpu():
        from xdsl.dialects.gpu import GPU

        return GPU

    def get_hls():
        from xdsl.dialects.experimental.hls import HLS

        return HLS

    def get_irdl():
        from xdsl.dialects.irdl import IRDL

        return IRDL

    def get_linalg():
        from xdsl.dialects.linalg import Linalg

        return Linalg

    def get_llvm():
        from xdsl.dialects.llvm import LLVM

        return LLVM

    def get_ltl():
        from xdsl.dialects.ltl import LTL

        return LTL

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

    def get_printf():
        from xdsl.dialects.printf import Printf

        return Printf

    def get_riscv():
        from xdsl.dialects.riscv import RISCV

        return RISCV

    def get_riscv_func():
        from xdsl.dialects.riscv_func import RISCV_Func

        return RISCV_Func

    def get_riscv_scf():
        from xdsl.dialects.riscv_scf import RISCV_Scf

        return RISCV_Scf

    def get_riscv_snitch():
        from xdsl.dialects.riscv_snitch import RISCV_Snitch

        return RISCV_Snitch

    def get_scf():
        from xdsl.dialects.scf import Scf

        return Scf

    def get_seq():
        from xdsl.dialects.seq import Seq

        return Seq

    def get_snitch():
        from xdsl.dialects.snitch import Snitch

        return Snitch

    def get_snitch_runtime():
        from xdsl.dialects.snitch_runtime import SnitchRuntime

        return SnitchRuntime

    def get_stencil():
        from xdsl.dialects.stencil import Stencil

        return Stencil

    def get_stream():
        from xdsl.dialects.stream import Stream

        return Stream

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
        ("comb", get_comb),
        ("cf", get_cf),
        ("cmath", get_cmath),
        ("dmp", get_dmp),
        ("fir", get_fir),
        ("fsm", get_fsm),
        ("func", get_func),
        ("gpu", get_gpu),
        ("hls", get_hls),
        ("irdl", get_irdl),
        ("linalg", get_linalg),
        ("llvm", get_llvm),
        ("ltl", get_ltl),
        ("math", get_math),
        ("memref", get_memref),
        ("mpi", get_mpi),
        ("pdl", get_pdl),
        ("printf", get_printf),
        ("riscv", get_riscv),
        ("riscv_func", get_riscv_func),
        ("riscv_scf", get_riscv_scf),
        ("riscv_snitch", get_riscv_snitch),
        ("scf", get_scf),
        ("seq", get_seq),
        ("snitch", get_snitch),
        ("snrt", get_snitch_runtime),
        ("stencil", get_stencil),
        ("stream", get_stream),
        ("symref", get_symref),
        ("test", get_test),
        ("vector", get_vector),
    ]


def get_all_passes() -> list[tuple[str, Callable[[], type[ModulePass]]]]:
    """Return the list of all available passes."""

    def get_canonicalize():
        from xdsl.transforms.canonicalize import CanonicalizePass

        return CanonicalizePass

    def get_canonicalize_dmp_pass():
        from xdsl.transforms.canonicalize_dmp import CanonicalizeDmpPass

        return CanonicalizeDmpPass

    def get_convert_stencil_to_llmlir_pass():
        from xdsl.transforms.experimental.convert_stencil_to_ll_mlir import (
            ConvertStencilToLLMLIRPass,
        )

        return ConvertStencilToLLMLIRPass

    def get_dead_code_elimination():
        from xdsl.transforms.dead_code_elimination import DeadCodeElimination

        return DeadCodeElimination

    def get_desymrefy_pass():
        from xdsl.frontend.passes.desymref import DesymrefyPass

        return DesymrefyPass

    def get_distribute_stencil_pass():
        from xdsl.transforms.experimental.dmp.stencil_global_to_local import (
            DistributeStencilPass,
        )

        return DistributeStencilPass

    def get_lower_halo_to_mpi():
        from xdsl.transforms.experimental.dmp.stencil_global_to_local import (
            LowerHaloToMPI,
        )

        return LowerHaloToMPI

    def get_lower_affine_pass():
        from xdsl.transforms.lower_affine import LowerAffinePass

        return LowerAffinePass

    def get_lower_mpi_pass():
        from xdsl.transforms.lower_mpi import LowerMPIPass

        return LowerMPIPass

    def get_lower_riscv_func():
        from xdsl.transforms.lower_riscv_func import LowerRISCVFunc

        return LowerRISCVFunc

    def get_lower_snitch_pass():
        from xdsl.transforms.lower_snitch import LowerSnitchPass

        return LowerSnitchPass

    def get_mlir_opt_pass():
        from xdsl.transforms.mlir_opt import MLIROptPass

        return MLIROptPass

    def get_printf_to_llvm():
        from xdsl.transforms.printf_to_llvm import PrintfToLLVM

        return PrintfToLLVM

    def get_printf_to_putchar_pass():
        from xdsl.transforms.printf_to_putchar import PrintfToPutcharPass

        return PrintfToPutcharPass

    def get_riscv_reduce_register_pressure_pass():
        from xdsl.backend.riscv.lowering.reduce_register_pressure import (
            RiscvReduceRegisterPressurePass,
        )

        return RiscvReduceRegisterPressurePass

    def get_riscv_register_allocation():
        from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation

        return RISCVRegisterAllocation

    def get_riscv_scf_loop_range_folding_pass():
        from xdsl.transforms.riscv_scf_loop_range_folding import (
            RiscvScfLoopRangeFoldingPass,
        )

        return RiscvScfLoopRangeFoldingPass

    def get_convert_arith_to_riscv_pass():
        from xdsl.backend.riscv.lowering.convert_arith_to_riscv import (
            ConvertArithToRiscvPass,
        )

        return ConvertArithToRiscvPass

    def get_convert_func_to_riscv_func_pass():
        from xdsl.backend.riscv.lowering.convert_func_to_riscv_func import (
            ConvertFuncToRiscvFuncPass,
        )

        return ConvertFuncToRiscvFuncPass

    def get_convert_memref_to_riscv_pass():
        from xdsl.backend.riscv.lowering.convert_memref_to_riscv import (
            ConvertMemrefToRiscvPass,
        )

        return ConvertMemrefToRiscvPass

    def get_convert_scf_to_riscv_pass():
        from xdsl.backend.riscv.lowering.convert_scf_to_riscv_scf import (
            ConvertScfToRiscvPass,
        )

        return ConvertScfToRiscvPass

    def get_lower_scf_for_to_labels():
        from xdsl.backend.riscv.riscv_scf_to_asm import LowerScfForToLabels

        return LowerScfForToLabels

    def get_stencil_shape_inference_pass():
        from xdsl.transforms.experimental.stencil_shape_inference import (
            StencilShapeInferencePass,
        )

        return StencilShapeInferencePass

    def get_stencil_storage_materialization_pass():
        from xdsl.transforms.experimental.stencil_storage_materialization import (
            StencilStorageMaterializationPass,
        )

        return StencilStorageMaterializationPass

    def get_reconcile_unrealized_casts_pass():
        from xdsl.transforms.reconcile_unrealized_casts import (
            ReconcileUnrealizedCastsPass,
        )

        return ReconcileUnrealizedCastsPass

    def get_hls_convert_stencil_to_llmlir_pass():
        from xdsl.transforms.experimental.hls_convert_stencil_to_ll_mlir import (
            HLSConvertStencilToLLMLIRPass,
        )

        return HLSConvertStencilToLLMLIRPass

    def get_lower_hls_pass():
        from xdsl.transforms.experimental.lower_hls import LowerHLSPass

        return LowerHLSPass

    def get_replace_incompatible_fpga():
        from xdsl.transforms.experimental.replace_incompatible_fpga import (
            ReplaceIncompatibleFPGA,
        )

        return ReplaceIncompatibleFPGA

    return [
        ("canonicalize", get_canonicalize),
        ("canonicalize-dmp", get_canonicalize_dmp_pass),
        ("convert-stencil-to-ll-mlir", get_convert_stencil_to_llmlir_pass),
        ("dce", get_dead_code_elimination),
        ("frontend-desymrefy", get_desymrefy_pass),
        ("distribute-stencil", get_distribute_stencil_pass),
        ("dmp-to-mpi", get_lower_halo_to_mpi),
        ("lower-affine", get_lower_affine_pass),
        ("lower-mpi", get_lower_mpi_pass),
        ("lower-riscv-func", get_lower_riscv_func),
        ("lower-snitch", get_lower_snitch_pass),
        ("mlir-opt", get_mlir_opt_pass),
        ("printf-to-llvm", get_printf_to_llvm),
        ("printf-to-putchar", get_printf_to_putchar_pass),
        ("riscv-reduce-register-pressure", get_riscv_reduce_register_pressure_pass),
        ("riscv-allocate-registers", get_riscv_register_allocation),
        ("riscv-scf-loop-range-folding", get_riscv_scf_loop_range_folding_pass),
        ("convert-arith-to-riscv", get_convert_arith_to_riscv_pass),
        ("convert-func-to-riscv-func", get_convert_func_to_riscv_func_pass),
        ("convert-memref-to-riscv", get_convert_memref_to_riscv_pass),
        ("convert-scf-to-riscv-scf", get_convert_scf_to_riscv_pass),
        ("lower-riscv-scf-to-labels", get_lower_scf_for_to_labels),
        ("stencil-shape-inference", get_stencil_shape_inference_pass),
        ("stencil-storage-materialization", get_stencil_storage_materialization_pass),
        ("reconcile-unrealized-casts", get_reconcile_unrealized_casts_pass),
        ("hls-convert-stencil-to-ll-mlir", get_hls_convert_stencil_to_llmlir_pass),
        ("lower-hls", get_lower_hls_pass),
        ("replace-incompatible-fpga", get_replace_incompatible_fpga),
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
        for name, dialect_factory in get_all_dialects():
            self.ctx.register_dialect(name, dialect_factory)

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
