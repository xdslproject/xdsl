import sys
from collections.abc import Callable

from xdsl.ir import Dialect
from xdsl.utils.dialect_loader import IRDLDialectFinder


def get_all_dialects() -> dict[str, Callable[[], Dialect]]:
    """Returns all available dialects."""

    def get_accfg():
        from xdsl.dialects.accfg import ACCFG

        return ACCFG

    def get_affine():
        from xdsl.dialects.affine import Affine

        return Affine

    def get_aie():
        from xdsl.dialects.experimental.aie import AIE

        return AIE

    def get_air():
        from xdsl.dialects.experimental.air import AIR

        return AIR

    def get_arith():
        from xdsl.dialects.arith import Arith

        return Arith

    def get_arm():
        from xdsl.dialects.arm import ARM

        return ARM

    def get_arm_func():
        from xdsl.dialects.arm_func import ARM_FUNC

        return ARM_FUNC

    def get_bufferization():
        from xdsl.dialects.bufferization import Bufferization

        return Bufferization

    def get_builtin():
        from xdsl.dialects.builtin import Builtin

        return Builtin

    def get_cf():
        from xdsl.dialects.cf import Cf

        return Cf

    def get_cmath():
        from xdsl.dialects.cmath import Cmath

        return Cmath

    def get_comb():
        from xdsl.dialects.comb import Comb

        return Comb

    def get_csl():
        from xdsl.dialects.csl import CSL

        return CSL

    def get_csl_stencil():
        from xdsl.dialects.csl.csl_stencil import CSL_STENCIL

        return CSL_STENCIL

    def get_csl_wrapper():
        from xdsl.dialects.csl.csl_wrapper import CSL_WRAPPER

        return CSL_WRAPPER

    def get_dmp():
        from xdsl.dialects.experimental.dmp import DMP

        return DMP

    def get_eqsat():
        from xdsl.dialects.eqsat import EqSat

        return EqSat

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

    def get_hlfir():
        from xdsl.dialects.experimental.hlfir import HLFIR

        return HLFIR

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

    def get_memref_stream():
        from xdsl.dialects.memref_stream import MemrefStream

        return MemrefStream

    def get_ml_program():
        from xdsl.dialects.ml_program import MLProgram

        return MLProgram

    def get_mod_arith():
        from xdsl.dialects.mod_arith import ModArith

        return ModArith

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

    def get_ptr_xdsl():
        from xdsl.dialects.ptr import Ptr

        return Ptr

    def get_quantum():
        from xdsl.dialects.quantum import QUANTUM

        return QUANTUM

    def get_qref():
        from xdsl.dialects.qref import QREF

        return QREF

    def get_qssa():
        from xdsl.dialects.qssa import QSSA

        return QSSA

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

    def get_stablehlo():
        from xdsl.dialects.stablehlo import StableHLO

        return StableHLO

    def get_stencil():
        from xdsl.dialects.stencil import Stencil

        return Stencil

    def get_stim():
        from xdsl.dialects.stim import Stim

        return Stim

    def get_symref():
        from xdsl.frontend.symref import Symref

        return Symref

    def get_tensor():
        from xdsl.dialects.tensor import Tensor

        return Tensor

    def get_test():
        from xdsl.dialects.test import Test

        return Test

    def get_tosa():
        from xdsl.dialects.tosa import TOSA

        return TOSA

    def get_varith():
        from xdsl.dialects.varith import Varith

        return Varith

    def get_vector():
        from xdsl.dialects.vector import Vector

        return Vector

    def get_wasm():
        from xdsl.dialects.wasm import Wasm

        return Wasm

    def get_x86():
        from xdsl.dialects.x86 import X86

        return X86

    def get_transform():
        from xdsl.dialects.transform import Transform

        return Transform

    return {
        "accfg": get_accfg,
        "affine": get_affine,
        "aie": get_aie,
        "air": get_air,
        "arith": get_arith,
        "arm": get_arm,
        "arm_func": get_arm_func,
        "bufferization": get_bufferization,
        "builtin": get_builtin,
        "cf": get_cf,
        "cmath": get_cmath,
        "comb": get_comb,
        "csl": get_csl,
        "csl_stencil": get_csl_stencil,
        "csl_wrapper": get_csl_wrapper,
        "dmp": get_dmp,
        "eqsat": get_eqsat,
        "fir": get_fir,
        "fsm": get_fsm,
        "func": get_func,
        "gpu": get_gpu,
        "hlfir": get_hlfir,
        "hls": get_hls,
        "hw": get_hw,
        "linalg": get_linalg,
        "irdl": get_irdl,
        "llvm": get_llvm,
        "ltl": get_ltl,
        "math": get_math,
        "memref": get_memref,
        "memref_stream": get_memref_stream,
        "ml_program": get_ml_program,
        "mod_arith": get_mod_arith,
        "mpi": get_mpi,
        "omp": get_omp,
        "onnx": get_onnx,
        "pdl": get_pdl,
        "printf": get_printf,
        "ptr_xdsl": get_ptr_xdsl,
        "quantum": get_quantum,
        "qref": get_qref,
        "qssa": get_qssa,
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
        "stablehlo": get_stablehlo,
        "stencil": get_stencil,
        "stim": get_stim,
        "symref": get_symref,
        "tensor": get_tensor,
        "test": get_test,
        "tosa": get_tosa,
        "varith": get_varith,
        "vector": get_vector,
        "wasm": get_wasm,
        "x86": get_x86,
        "transform": get_transform,
    }


# Add the IRDLDialectFinder to the meta path as last resort, i.e, it will look for a
# .irdl implementation if no .py implementation is found.
sys.meta_path.append(IRDLDialectFinder(get_all_dialects))
