from xdsl.builder import ImplicitBuilder
from xdsl.dialects import riscv, snitch_stream
from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.riscv import RawPtr, RiscvFunctions
from xdsl.interpreters.snitch_stream import (
    SnitchStreamFunctions,
    StridedPointerInputStream,
    StridedPointerOutputStream,
    StridePattern,
    indexing_map_from_bounds,
    offset_map_from_strides,
)
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.utils.test_value import TestSSAValue


def test_indexing_map_constructor():
    assert indexing_map_from_bounds([]) == AffineMap(1, 0, ())
    assert indexing_map_from_bounds([2]) == AffineMap(
        1, 0, (AffineExpr.dimension(0) % 2,)
    )
    assert indexing_map_from_bounds([2, 3]) == AffineMap(
        1, 0, (AffineExpr.dimension(0).floor_div(3) % 2, AffineExpr.dimension(0) % 3)
    )
    assert indexing_map_from_bounds([2, 3, 4]) == AffineMap(
        1,
        0,
        (
            AffineExpr.dimension(0).floor_div(12) % 2,
            AffineExpr.dimension(0).floor_div(4) % 3,
            AffineExpr.dimension(0) % 4,
        ),
    )


def test_offset_map_constructor():
    assert offset_map_from_strides([]) == AffineMap(1, 0, ())
    assert offset_map_from_strides([2]) == AffineMap.from_callable(lambda i: (i * 2,))
    assert offset_map_from_strides([2, 1]) == AffineMap.from_callable(
        lambda i, j: (i * 2 + j * 1,)
    )
    assert offset_map_from_strides([3, 2, 1]) == AffineMap.from_callable(
        lambda i, j, k: (i * 3 + j * 2 + k * 1,)
    )


def test_snitch_stream_interpreter():
    register = riscv.IntRegisterType.unallocated()
    pattern_type = snitch_stream.StridePatternType()

    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(RiscvFunctions())
    interpreter.register_implementations(SnitchStreamFunctions())

    stride_pattern_op = snitch_stream.StridePatternOp(
        ArrayAttr((IntAttr(2), IntAttr(3))),
        ArrayAttr((IntAttr(24), IntAttr(8))),
        IntAttr(0),
    )

    stride_pattern = StridePattern([2, 3], [24, 8])
    assert interpreter.run_op(stride_pattern_op, ()) == (stride_pattern,)

    a = RawPtr.new_float64([2.0] * 6)
    b = RawPtr.new_float64([3.0] * 6)
    c = RawPtr.new_float64([4.0] * 6)

    a_stream_op = snitch_stream.StridedReadOp(
        TestSSAValue(register),
        TestSSAValue(pattern_type),
        riscv.Registers.FT0,
        IntAttr(0),
        IntAttr(2),
    )

    assert interpreter.run_op(a_stream_op, (a, stride_pattern)) == (
        StridedPointerInputStream(stride_pattern.offset_expr, a),
    )

    b_stream_op = snitch_stream.StridedReadOp(
        TestSSAValue(register),
        TestSSAValue(pattern_type),
        riscv.Registers.FT1,
        IntAttr(1),
        IntAttr(2),
    )

    assert interpreter.run_op(b_stream_op, (b, stride_pattern)) == (
        StridedPointerInputStream(stride_pattern.offset_expr, b),
    )

    c_stream_op = snitch_stream.StridedWriteOp(
        TestSSAValue(register),
        TestSSAValue(pattern_type),
        riscv.Registers.FT2,
        IntAttr(2),
        IntAttr(2),
    )

    assert interpreter.run_op(c_stream_op, (c, stride_pattern)) == (
        StridedPointerOutputStream(stride_pattern.offset_expr, c),
    )

    body = Region(Block(arg_types=(riscv.Registers.FT0, riscv.Registers.FT1)))

    with ImplicitBuilder(body) as (a_reg, b_reg):
        c_reg = riscv.FAddDOp(a_reg, b_reg, rd=riscv.Registers.FT2).rd
        snitch_stream.YieldOp(c_reg)

    assert (
        interpreter.run_op(
            snitch_stream.GenericOp(
                TestSSAValue(register),
                (a_stream_op.stream, b_stream_op.stream),
                (c_stream_op.stream,),
                body,
            ),
            (
                6,
                StridedPointerInputStream(stride_pattern.offset_expr, a),
                StridedPointerInputStream(stride_pattern.offset_expr, b),
                StridedPointerOutputStream(stride_pattern.offset_expr, c),
            ),
        )
        == ()
    )

    assert c.float64.get_list(6) == [5.0] * 6
