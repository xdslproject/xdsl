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


def get_all_dialects() -> dict[str, Callable[[], Dialect]]:
    """Returns all available dialects."""

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

    def get_hw():
        from xdsl.dialects.hw import HW

        return HW

    def get_linalg():
        from xdsl.dialects.linalg import Linalg

        return Linalg

    def get_irdl():
        from xdsl.dialects.irdl.irdl import IRDL

        return IRDL

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

    def get_omp():
        from xdsl.dialects.omp import OMP

        return OMP

    def get_onnx():
        from xdsl.dialects.onnx import ONNX

        return ONNX

    def get_pdl():
        from xdsl.dialects.pdl import PDL

        return PDL

    def get_printf():
        from xdsl.dialects.printf import Printf

        return Printf

    def get_riscv_debug():
        from xdsl.dialects.riscv_debug import RISCV_Debug

        return RISCV_Debug

    def get_riscv():
        from xdsl.dialects.riscv import RISCV

        return RISCV

    def get_riscv_func():
        from xdsl.dialects.riscv_func import RISCV_Func

        return RISCV_Func

    def get_riscv_scf():
        from xdsl.dialects.riscv_scf import RISCV_Scf

        return RISCV_Scf

    def get_riscv_cf():
        from xdsl.dialects.riscv_cf import RISCV_Cf

        return RISCV_Cf

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

    def get_snitch_stream():
        from xdsl.dialects.snitch_stream import SnitchStream

        return SnitchStream

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

    return {
        "affine": get_affine,
        "arith": get_arith,
        "builtin": get_builtin,
        "cf": get_cf,
        "cmath": get_cmath,
        "comb": get_comb,
        "dmp": get_dmp,
        "fir": get_fir,
        "fsm": get_fsm,
        "func": get_func,
        "gpu": get_gpu,
        "hls": get_hls,
        "hw": get_hw,
        "linalg": get_linalg,
        "irdl": get_irdl,
        "llvm": get_llvm,
        "ltl": get_ltl,
        "math": get_math,
        "memref": get_memref,
        "mpi": get_mpi,
        "omp": get_omp,
        "onnx": get_onnx,
        "pdl": get_pdl,
        "printf": get_printf,
        "riscv": get_riscv,
        "riscv_debug": get_riscv_debug,
        "riscv_func": get_riscv_func,
        "riscv_scf": get_riscv_scf,
        "riscv_cf": get_riscv_cf,
        "riscv_snitch": get_riscv_snitch,
        "scf": get_scf,
        "seq": get_seq,
        "snitch": get_snitch,
        "snrt": get_snitch_runtime,
        "snitch_stream": get_snitch_stream,
        "stencil": get_stencil,
        "stream": get_stream,
        "symref": get_symref,
        "test": get_test,
        "vector": get_vector,
    }


def get_all_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    """Return the list of all available passes."""

    def get_arith_add_fastmath():
        from xdsl.transforms import arith_add_fastmath

        return arith_add_fastmath.AddArithFastMathFlagsPass

    def get_canonicalize():
        from xdsl.transforms import canonicalize

        return canonicalize.CanonicalizePass

    def get_canonicalize_dmp():
        from xdsl.transforms import canonicalize_dmp

        return canonicalize_dmp.CanonicalizeDmpPass

    def get_convert_riscv_scf_for_to_frep():
        from xdsl.transforms import convert_riscv_scf_for_to_frep

        return convert_riscv_scf_for_to_frep.ConvertRiscvScfForToFrepPass

    def get_convert_scf_to_openmp():
        from xdsl.transforms import convert_scf_to_openmp

        return convert_scf_to_openmp.ConvertScfToOpenMPPass

    def get_convert_snitch_stream_to_snitch():
        from xdsl.backend.riscv.lowering import convert_snitch_stream_to_snitch

        return convert_snitch_stream_to_snitch.ConvertSnitchStreamToSnitch

    def get_constant_fold_interp():
        from xdsl.transforms import constant_fold_interp

        return constant_fold_interp.ConstantFoldInterpPass

    def get_convert_stencil_to_ll_mlir():
        from xdsl.transforms.experimental import convert_stencil_to_ll_mlir

        return convert_stencil_to_ll_mlir.ConvertStencilToLLMLIRPass

    def get_convert_riscv_scf_to_riscv_cf():
        from xdsl.backend.riscv.lowering import (
            convert_riscv_scf_to_riscv_cf,
        )

        return convert_riscv_scf_to_riscv_cf.ConvertRiscvScfToRiscvCfPass

    def get_dce():
        from xdsl.transforms import dead_code_elimination

        return dead_code_elimination.DeadCodeElimination

    def get_desymrefy():
        from xdsl.frontend.passes.desymref import DesymrefyPass

        return DesymrefyPass

    def get_gpu_map_parallel_loops():
        from xdsl.transforms import gpu_map_parallel_loops

        return gpu_map_parallel_loops.GpuMapParallelLoopsPass

    def get_distribute_stencil():
        from xdsl.transforms.experimental.dmp import stencil_global_to_local

        return stencil_global_to_local.DistributeStencilPass

    def get_lower_halo_to_mpi():
        from xdsl.transforms.experimental.dmp import stencil_global_to_local

        return stencil_global_to_local.LowerHaloToMPI

    def get_lower_affine():
        from xdsl.transforms import lower_affine

        return lower_affine.LowerAffinePass

    def get_lower_mpi():
        from xdsl.transforms import lower_mpi

        return lower_mpi.LowerMPIPass

    def get_lower_riscv_func():
        from xdsl.transforms import lower_riscv_func

        return lower_riscv_func.LowerRISCVFunc

    def get_lower_snitch():
        from xdsl.transforms import lower_snitch

        return lower_snitch.LowerSnitchPass

    def get_mlir_opt():
        from xdsl.transforms import mlir_opt

        return mlir_opt.MLIROptPass

    def get_printf_to_llvm():
        from xdsl.transforms import printf_to_llvm

        return printf_to_llvm.PrintfToLLVM

    def get_printf_to_putchar():
        from xdsl.transforms import printf_to_putchar

        return printf_to_putchar.PrintfToPutcharPass

    def get_reduce_register_pressure():
        from xdsl.backend.riscv.lowering import reduce_register_pressure

        return reduce_register_pressure.RiscvReduceRegisterPressurePass

    def get_riscv_register_allocation():
        from xdsl.transforms import riscv_register_allocation

        return riscv_register_allocation.RISCVRegisterAllocation

    def get_riscv_scf_loop_range_folding():
        from xdsl.transforms import riscv_scf_loop_range_folding

        return riscv_scf_loop_range_folding.RiscvScfLoopRangeFoldingPass

    def get_snitch_register_allocation():
        from xdsl.transforms import snitch_register_allocation

        return snitch_register_allocation.SnitchRegisterAllocation

    def get_convert_arith_to_riscv():
        from xdsl.backend.riscv.lowering import convert_arith_to_riscv

        return convert_arith_to_riscv.ConvertArithToRiscvPass

    def get_convert_func_to_riscv_func():
        from xdsl.backend.riscv.lowering import convert_func_to_riscv_func

        return convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass

    def get_convert_memref_to_riscv():
        from xdsl.backend.riscv.lowering import convert_memref_to_riscv

        return convert_memref_to_riscv.ConvertMemrefToRiscvPass

    def get_convert_print_format_to_riscv_debug():
        from xdsl.backend.riscv.lowering import convert_print_format_to_riscv_debug

        return convert_print_format_to_riscv_debug.ConvertPrintFormatToRiscvDebugPass

    def get_scf_parallel_loop_tiling():
        from xdsl.transforms import scf_parallel_loop_tiling

        return scf_parallel_loop_tiling.ScfParallelLoopTilingPass

    def get_convert_scf_to_riscv_scf():
        from xdsl.backend.riscv.lowering import convert_scf_to_riscv_scf

        return convert_scf_to_riscv_scf.ConvertScfToRiscvPass

    def get_lower_scf_for_to_labels():
        from xdsl.backend.riscv import riscv_scf_to_asm

        return riscv_scf_to_asm.LowerScfForToLabels

    def get_stencil_shape_inference():
        from xdsl.transforms.experimental import stencil_shape_inference

        return stencil_shape_inference.StencilShapeInferencePass

    def get_stencil_storage_materialization():
        from xdsl.transforms.experimental import stencil_storage_materialization

        return stencil_storage_materialization.StencilStorageMaterializationPass

    def get_reconcile_unrealized_casts():
        from xdsl.transforms import reconcile_unrealized_casts

        return reconcile_unrealized_casts.ReconcileUnrealizedCastsPass

    def get_hls_convert_stencil_to_ll_mlir():
        from xdsl.transforms.experimental import hls_convert_stencil_to_ll_mlir

        return hls_convert_stencil_to_ll_mlir.HLSConvertStencilToLLMLIRPass

    def get_lower_hls():
        from xdsl.transforms.experimental import lower_hls

        return lower_hls.LowerHLSPass

    def get_replace_incompatible_fpga():
        from xdsl.transforms.experimental import replace_incompatible_fpga

        return replace_incompatible_fpga.ReplaceIncompatibleFPGA

    def get_stencil_unroll():
        from xdsl.transforms import stencil_unroll

        return stencil_unroll.StencilUnrollPass

    return {
        "arith-add-fastmath": get_arith_add_fastmath,
        "canonicalize-dmp": get_canonicalize_dmp,
        "canonicalize": get_canonicalize,
        "constant-fold-interp": get_constant_fold_interp,
        "convert-arith-to-riscv": get_convert_arith_to_riscv,
        "convert-func-to-riscv-func": get_convert_func_to_riscv_func,
        "convert-memref-to-riscv": get_convert_memref_to_riscv,
        "convert-print-format-to-riscv-debug": get_convert_print_format_to_riscv_debug,
        "convert-riscv-scf-for-to-frep": get_convert_riscv_scf_for_to_frep,
        "convert-riscv-scf-to-riscv-cf": get_convert_riscv_scf_to_riscv_cf,
        "convert-scf-to-openmp": get_convert_scf_to_openmp,
        "convert-scf-to-riscv-scf": get_convert_scf_to_riscv_scf,
        "convert-snitch-stream-to-snitch": get_convert_snitch_stream_to_snitch,
        "convert-stencil-to-ll-mlir": get_convert_stencil_to_ll_mlir,
        "dce": get_dce,
        "distribute-stencil": get_distribute_stencil,
        "dmp-to-mpi": get_lower_halo_to_mpi,
        "frontend-desymrefy": get_desymrefy,
        "gpu-map-parallel-loops": get_gpu_map_parallel_loops,
        "hls-convert-stencil-to-ll-mlir": get_hls_convert_stencil_to_ll_mlir,
        "lower-affine": get_lower_affine,
        "lower-hls": get_lower_hls,
        "lower-mpi": get_lower_mpi,
        "lower-riscv-func": get_lower_riscv_func,
        "lower-riscv-scf-to-labels": get_lower_scf_for_to_labels,
        "lower-snitch": get_lower_snitch,
        "mlir-opt": get_mlir_opt,
        "printf-to-llvm": get_printf_to_llvm,
        "printf-to-putchar": get_printf_to_putchar,
        "reconcile-unrealized-casts": get_reconcile_unrealized_casts,
        "replace-incompatible-fpga": get_replace_incompatible_fpga,
        "riscv-allocate-registers": get_riscv_register_allocation,
        "riscv-reduce-register-pressure": get_reduce_register_pressure,
        "riscv-scf-loop-range-folding": get_riscv_scf_loop_range_folding,
        "scf-parallel-loop-tiling": get_scf_parallel_loop_tiling,
        "snitch-allocate-registers": get_snitch_register_allocation,
        "stencil-shape-inference": get_stencil_shape_inference,
        "stencil-storage-materialization": get_stencil_storage_materialization,
        "stencil-unroll": get_stencil_unroll,
    }


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
        for dialect_name, dialect_factory in get_all_dialects().items():
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
