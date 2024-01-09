import argparse
import os
import sys
from collections.abc import Callable
from typing import IO

from xdsl.dialects.affine import Affine
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.cf import Cf
from xdsl.dialects.cmath import CMath
from xdsl.dialects.comb import Comb
from xdsl.dialects.experimental.dmp import DMP
from xdsl.dialects.experimental.fir import FIR
from xdsl.dialects.experimental.hls import HLS
from xdsl.dialects.experimental.math import Math
from xdsl.dialects.fsm import FSM
from xdsl.dialects.func import Func
from xdsl.dialects.gpu import GPU
from xdsl.dialects.hw import HW
from xdsl.dialects.irdl.irdl import IRDL
from xdsl.dialects.linalg import Linalg
from xdsl.dialects.llvm import LLVM
from xdsl.dialects.ltl import LTL
from xdsl.dialects.memref import MemRef
from xdsl.dialects.mpi import MPI
from xdsl.dialects.omp import OMP
from xdsl.dialects.onnx import ONNX
from xdsl.dialects.pdl import PDL
from xdsl.dialects.printf import Printf
from xdsl.dialects.riscv import RISCV
from xdsl.dialects.riscv_cf import RISCV_Cf
from xdsl.dialects.riscv_func import RISCV_Func
from xdsl.dialects.riscv_scf import RISCV_Scf
from xdsl.dialects.riscv_snitch import RISCV_Snitch
from xdsl.dialects.scf import Scf
from xdsl.dialects.seq import Seq
from xdsl.dialects.snitch import Snitch
from xdsl.dialects.snitch_runtime import SnitchRuntime
from xdsl.dialects.snitch_stream import SnitchStream
from xdsl.dialects.stencil import Stencil
from xdsl.dialects.stream import Stream
from xdsl.dialects.test import Test
from xdsl.dialects.vector import Vector
from xdsl.frontend.symref import Symref
from xdsl.ir import Dialect, MLContext
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import ParseError


def get_all_dialects() -> list[Dialect]:
    """Return the list of all available dialects."""
    return [
        Affine,
        Arith,
        Builtin,
        Cf,
        CMath,
        Comb,
        DMP,
        FIR,
        FSM,
        Func,
        GPU,
        HLS,
        HW,
        Linalg,
        IRDL,
        LLVM,
        LTL,
        Math,
        MemRef,
        MPI,
        OMP,
        ONNX,
        PDL,
        Printf,
        RISCV,
        RISCV_Cf,
        RISCV_Func,
        RISCV_Scf,
        RISCV_Snitch,
        Scf,
        Seq,
        Snitch,
        SnitchRuntime,
        SnitchStream,
        Stencil,
        Stream,
        Symref,
        Test,
        Vector,
    ]


def get_all_passes() -> list[tuple[str, Callable[[], type[ModulePass]]]]:
    """Return the list of all available passes."""

    def get_canonicalize():
        from xdsl.transforms import canonicalize

        return canonicalize.CanonicalizePass

    def get_canonicalize_dmp():
        from xdsl.transforms import canonicalize_dmp

        return canonicalize_dmp.CanonicalizeDmpPass

    def get_convert_scf_to_openmp():
        from xdsl.transforms import convert_scf_to_openmp

        return convert_scf_to_openmp.ConvertScfToOpenMPPass

    def get_convert_snitch_stream_to_snitch():
        from xdsl.backend.riscv.lowering import (
            convert_snitch_stream_to_snitch,
        )

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

    def get_scf_parallel_loop_tiling():
        from xdsl.transforms import scf_parallel_loop_tiling

        return scf_parallel_loop_tiling.ScfParallelLoopTilingPass

    def get_convert_scf_to_riscv_scf():
        from xdsl.backend.riscv.lowering import convert_scf_to_riscv_scf

        return convert_scf_to_riscv_scf.ConvertScfToRiscvPass

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

    return [
        ("canonicalize", get_canonicalize),
        ("canonicalize-dmp", get_canonicalize_dmp),
        ("convert-scf-to-openmp", get_convert_scf_to_openmp),
        ("convert-snitch-stream-to-snitch", get_convert_snitch_stream_to_snitch),
        ("convert-riscv-scf-to-riscv-cf", get_convert_riscv_scf_to_riscv_cf),
        ("constant-fold-interp", get_constant_fold_interp),
        ("convert-stencil-to-ll-mlir", get_convert_stencil_to_ll_mlir),
        ("dce", get_dce),
        ("frontend-desymrefy", get_desymrefy),
        ("gpu-map-parallel-loops", get_gpu_map_parallel_loops),
        ("distribute-stencil", get_distribute_stencil),
        ("dmp-to-mpi", get_lower_halo_to_mpi),
        ("lower-affine", get_lower_affine),
        ("lower-mpi", get_lower_mpi),
        ("lower-riscv-func", get_lower_riscv_func),
        ("lower-snitch", get_lower_snitch),
        ("mlir-opt", get_mlir_opt),
        ("printf-to-llvm", get_printf_to_llvm),
        ("printf-to-putchar", get_printf_to_putchar),
        ("riscv-reduce-register-pressure", get_reduce_register_pressure),
        ("riscv-allocate-registers", get_riscv_register_allocation),
        ("riscv-scf-loop-range-folding", get_riscv_scf_loop_range_folding),
        ("snitch-allocate-registers", get_snitch_register_allocation),
        ("convert-arith-to-riscv", get_convert_arith_to_riscv),
        ("convert-func-to-riscv-func", get_convert_func_to_riscv_func),
        ("convert-memref-to-riscv", get_convert_memref_to_riscv),
        ("scf-parallel-loop-tiling", get_scf_parallel_loop_tiling),
        ("convert-scf-to-riscv-scf", get_convert_scf_to_riscv_scf),
        ("stencil-shape-inference", get_stencil_shape_inference),
        ("stencil-storage-materialization", get_stencil_storage_materialization),
        ("reconcile-unrealized-casts", get_reconcile_unrealized_casts),
        ("hls-convert-stencil-to-ll-mlir", get_hls_convert_stencil_to_ll_mlir),
        ("lower-hls", get_lower_hls),
        ("replace-incompatible-fpga", get_replace_incompatible_fpga),
        ("stencil-unroll", get_stencil_unroll),
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
            self.ctx.load_dialect(dialect)

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
