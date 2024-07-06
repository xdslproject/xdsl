from xdsl.builder import Builder, ImplicitBuilder
from xdsl.context import MLContext
from xdsl.dialects import func, memref, riscv
from xdsl.dialects.builtin import ModuleOp, UnrealizedConversionCastOp, f32
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.test_value import TestSSAValue

from ..rewrites.lower_memref_riscv import LowerMemrefToRiscv

MEMREF_TYPE_2XF32 = memref.MemRefType(f32, ([2]))
MEMREF_TYPE_2X2XF32 = memref.MemRefType(f32, ([2, 2]))
MEMREF_TYPE_2X2X2XF32 = memref.MemRefType(f32, ([2, 2, 2]))

INT_REGISTER_TYPE = riscv.IntRegisterType.unallocated()
FLOAT_REGISTER_TYPE = riscv.FloatRegisterType.unallocated()


def test_lower_memref_alloc():
    @ModuleOp
    @Builder.implicit_region
    def simple_alloc():
        v1 = memref.Alloc.get(f32, shape=[2]).memref
        riscv.CustomAssemblyInstructionOp("do_stuff_with_alloc", (v1,), ())

    @ModuleOp
    @Builder.implicit_region
    def expected():
        v1 = riscv.LiOp(2, comment="memref alloc size")
        v2 = riscv.CustomAssemblyInstructionOp(
            "buffer.alloc", (v1.results[0],), (INT_REGISTER_TYPE,)
        )
        v3 = UnrealizedConversionCastOp.get((v2.results[0],), (MEMREF_TYPE_2XF32,))
        riscv.CustomAssemblyInstructionOp("do_stuff_with_alloc", (v3.results[0],), ())

    LowerMemrefToRiscv().apply(MLContext(), simple_alloc)
    assert f"{expected}" == f"{simple_alloc}"


def test_lower_memref_dealloc():
    @ModuleOp
    @Builder.implicit_region
    def simple_dealloc():
        with ImplicitBuilder(func.FuncOp("impl", ((MEMREF_TYPE_2XF32,), ())).body) as (
            b,
        ):
            memref.Dealloc.get(b)

    @ModuleOp
    @Builder.implicit_region
    def expected():
        with ImplicitBuilder(func.FuncOp("impl", ((MEMREF_TYPE_2XF32,), ())).body) as (
            _,
        ):
            pass

    LowerMemrefToRiscv().apply(MLContext(), simple_dealloc)
    assert f"{expected}" == f"{simple_dealloc}"
