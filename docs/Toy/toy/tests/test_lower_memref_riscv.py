from xdsl.backend.riscv.lowering.convert_memref_to_riscv import memref_shape_ops
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
            bytes_per_element = riscv.LiOp(4).rd
            offset_in_bytes = riscv.MulOp(
                indices[0],
                bytes_per_element,
                rd=riscv.IntRegisterType.unallocated(),
                comment="multiply by element size",
            ).rd
            _ = riscv.AddOp(
                mem, offset_in_bytes, rd=riscv.IntRegisterType.unallocated()
            )
            riscv.CustomAssemblyInstructionOp("some_memref_op", (), ())

    shape = [2]
    dummy_op = list(input_1d.walk())[-1]
    rewriter = PatternRewriter(dummy_op)
    ops, _ = memref_shape_ops(mem, indices, shape, MEMREF_TYPE_2XF32.element_type)
    rewriter.insert_op_before_matched_op(ops)

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
            v2 = riscv.MulOp(v1, indices[0], rd=riscv.IntRegisterType.unallocated())
            v3 = riscv.AddOp(v2, indices[1], rd=riscv.IntRegisterType.unallocated())
            v4 = riscv.LiOp(4).rd
            v5 = riscv.MulOp(
                v3,
                v4,
                rd=riscv.IntRegisterType.unallocated(),
                comment="multiply by element size",
            ).rd
            _ = riscv.AddOp(mem, v5, rd=riscv.IntRegisterType.unallocated())
            riscv.CustomAssemblyInstructionOp("some_memref_op", (), ())

    shape = [2, 2]
    dummy_op = list(input_2d.walk())[-1]
    rewriter = PatternRewriter(dummy_op)
    ops, _ = memref_shape_ops(mem, indices, shape, MEMREF_TYPE_2XF32.element_type)
    rewriter.insert_op_before_matched_op(ops)

    assert f"{input_2d}" == f"{expected_2d}"


def test_insert_shape_ops_3d():
    mem = TestSSAValue(MEMREF_TYPE_2X2X2XF32)
    indices = [
        TestSSAValue(INT_REGISTER_TYPE),
        TestSSAValue(INT_REGISTER_TYPE),
        TestSSAValue(INT_REGISTER_TYPE),
    ]

    @ModuleOp
    @Builder.implicit_region
    def input_3d():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            riscv.CustomAssemblyInstructionOp("some_memref_op", (), ())

    @ModuleOp
    @Builder.implicit_region
    def expected_3d():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            v1 = riscv.LiOp(2)
            v2 = riscv.MulOp(v1, indices[0], rd=riscv.IntRegisterType.unallocated())
            v3 = riscv.AddOp(v2, indices[1], rd=riscv.IntRegisterType.unallocated())
            v4 = riscv.LiOp(2)
            v4 = riscv.MulOp(v4, v3, rd=riscv.IntRegisterType.unallocated())
            v5 = riscv.AddOp(v4, indices[2], rd=riscv.IntRegisterType.unallocated())
            v6 = riscv.LiOp(4).rd
            v7 = riscv.MulOp(
                v5,
                v6,
                rd=riscv.IntRegisterType.unallocated(),
                comment="multiply by element size",
            ).rd
            _ = riscv.AddOp(mem, v7, rd=riscv.IntRegisterType.unallocated())
            riscv.CustomAssemblyInstructionOp("some_memref_op", (), ())

    shape = [2, 2, 2]
    dummy_op = list(input_3d.walk())[-1]
    rewriter = PatternRewriter(dummy_op)

    ops, _ = memref_shape_ops(mem, indices, shape, MEMREF_TYPE_2XF32.element_type)
    rewriter.insert_op_before_matched_op(ops)

    assert f"{input_3d}" == f"{expected_3d}"
