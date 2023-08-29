import pytest

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import func, memref, riscv
from xdsl.dialects.builtin import IndexType, ModuleOp, UnrealizedConversionCastOp, f32
from xdsl.ir import MLContext
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.test_value import TestSSAValue

from ..rewrites.lower_memref_riscv import LowerMemrefToRiscv, insert_shape_ops

MEMREF_TYPE_2XF32 = memref.MemRefType.from_element_type_and_shape(f32, ([2]))
MEMREF_TYPE_2X2XF32 = memref.MemRefType.from_element_type_and_shape(f32, ([2, 2]))
MEMREF_TYPE_2X2X2XF32 = memref.MemRefType.from_element_type_and_shape(f32, ([2, 2, 2]))

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


def test_insert_shape_ops_1d():
    mem = TestSSAValue(MEMREF_TYPE_2XF32)
    indices = [TestSSAValue(INT_REGISTER_TYPE)]

    @ModuleOp
    @Builder.implicit_region
    def input_1d():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            riscv.CustomAssemblyInstructionOp("some_memref_op", (), ())

    @ModuleOp
    @Builder.implicit_region
    def expected_1d():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            _ = riscv.AddOp(mem, indices[0])
            riscv.CustomAssemblyInstructionOp("some_memref_op", (), ())

    shape = [2]
    dummy_op = list(input_1d.walk())[-1]
    rewriter = PatternRewriter(dummy_op)
    _ = insert_shape_ops(mem, indices, shape, rewriter)

    assert f"{expected_1d}" == f"{input_1d}"


def test_insert_shape_ops_2d():
    mem = TestSSAValue(MEMREF_TYPE_2X2XF32)
    indices = [TestSSAValue(INT_REGISTER_TYPE), TestSSAValue(INT_REGISTER_TYPE)]

    @ModuleOp
    @Builder.implicit_region
    def input_2d():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            riscv.CustomAssemblyInstructionOp("some_memref_op", (), ())

    @ModuleOp
    @Builder.implicit_region
    def expected_2d():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            v1 = riscv.LiOp(2)
            v2 = riscv.MulOp(v1, indices[0])
            v3 = riscv.AddOp(v2, indices[1])
            v4 = riscv.SlliOp(v3, 2, comment="mutiply by elm size")
            _ = riscv.AddOp(mem, v4)
            riscv.CustomAssemblyInstructionOp("some_memref_op", (), ())

    shape = [2, 2]
    dummy_op = list(input_2d.walk())[-1]
    rewriter = PatternRewriter(dummy_op)
    _ = insert_shape_ops(mem, indices, shape, rewriter)

    assert f"{input_2d}" == f"{expected_2d}"


def test_insert_shape_ops_invalid_dim():
    mem = TestSSAValue(MEMREF_TYPE_2X2X2XF32)
    indices = [
        TestSSAValue(INT_REGISTER_TYPE),
        TestSSAValue(INT_REGISTER_TYPE),
        TestSSAValue(INT_REGISTER_TYPE),
    ]

    @ModuleOp
    @Builder.implicit_region
    def input_2d():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            riscv.CustomAssemblyInstructionOp("some_memref_op", (), ())

    shape = [2, 2, 2]
    dummy_op = list(input_2d.walk())[-1]
    rewriter = PatternRewriter(dummy_op)

    with pytest.raises(NotImplementedError):
        _ = insert_shape_ops(mem, indices, shape, rewriter)


# insert_shape_ops has already been tested in the above tests
# so we can reduce the code here a bit to only test a single dimension


def test_memref_load():
    @ModuleOp
    @Builder.implicit_region
    def simple_load():
        with ImplicitBuilder(
            func.FuncOp("impl", ((MEMREF_TYPE_2XF32, (IndexType())), ())).body
        ) as (v, i):
            memref.Load.get(v, [i])

    @ModuleOp
    @Builder.implicit_region
    def expected():
        with ImplicitBuilder(
            func.FuncOp("impl", ((MEMREF_TYPE_2XF32, IndexType()), ())).body
        ) as (v, i):
            v1 = UnrealizedConversionCastOp.get([v], (INT_REGISTER_TYPE,))
            v2 = UnrealizedConversionCastOp.get([i], (INT_REGISTER_TYPE,))
            v4 = riscv.AddOp(v1.results[0], v2.results[0])
            v5 = riscv.FLwOp(v4, 0, comment="load value from memref of shape (2,)")
            _ = UnrealizedConversionCastOp.get([v5], (f32,))

    LowerMemrefToRiscv().apply(MLContext(), simple_load)
    assert f"{expected}" == f"{simple_load}"


def test_memref_store():
    @ModuleOp
    @Builder.implicit_region
    def simple_store():
        with ImplicitBuilder(
            func.FuncOp("impl", ((f32, MEMREF_TYPE_2XF32, IndexType()), ())).body
        ) as (v, m, i):
            memref.Store.get(v, m, [i])

    @ModuleOp
    @Builder.implicit_region
    def expected():
        with ImplicitBuilder(
            func.FuncOp("impl", ((f32, MEMREF_TYPE_2XF32, IndexType()), ())).body
        ) as (v, m, i):
            v1 = UnrealizedConversionCastOp.get([v], (FLOAT_REGISTER_TYPE,))
            v2 = UnrealizedConversionCastOp.get([m], (INT_REGISTER_TYPE,))
            v3 = UnrealizedConversionCastOp.get([i], (INT_REGISTER_TYPE,))
            v4 = riscv.AddOp(v2.results[0], v3.results[0])
            riscv.FSwOp(
                v4,
                v1.results[0],
                0,
                comment="store float value to memref of shape (2,)",
            )

    LowerMemrefToRiscv().apply(MLContext(), simple_store)
    assert f"{expected}" == f"{simple_store}"
