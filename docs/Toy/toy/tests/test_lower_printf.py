from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import func, memref, printf, riscv
from xdsl.dialects.builtin import ModuleOp, UnrealizedConversionCastOp, f64
from xdsl.ir import MLContext

from ..rewrites.lower_printf_riscv import LowerPrintfRiscvPass


def input_module():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            m1 = memref.Alloc.get(f64, shape=[2]).memref
            printf.PrintFormatOp("{}", m1)
            m2 = memref.Alloc.get(f64, shape=[3, 4]).memref
            printf.PrintFormatOp("{}", m2)

    return ir_module


INT_REGISTER = riscv.IntRegisterType.unallocated()


def expected_module():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            m1 = memref.Alloc.get(f64, shape=[2]).memref
            m1d0 = riscv.LiOp(2).rd
            m1ptr = UnrealizedConversionCastOp.get((m1,), (INT_REGISTER,)).results[0]
            riscv.CustomAssemblyInstructionOp("tensor.print1d", (m1ptr, m1d0), ())
            m2 = memref.Alloc.get(f64, shape=[3, 4]).memref
            m2d0 = riscv.LiOp(3).rd
            m2d1 = riscv.LiOp(4).rd
            m2ptr = UnrealizedConversionCastOp.get((m2,), (INT_REGISTER,)).results[0]
            riscv.CustomAssemblyInstructionOp("tensor.print2d", (m2ptr, m2d0, m2d1), ())

    return ir_module


def test_lower_printf():
    module = input_module()
    LowerPrintfRiscvPass().apply(MLContext(), module)
    assert f"{expected_module()}" == f"{module}"
