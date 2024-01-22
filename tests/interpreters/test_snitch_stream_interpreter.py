from xdsl.builder import ImplicitBuilder
from xdsl.dialects import riscv, riscv_snitch, snitch_stream, stream
from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.riscv import RawPtr, RiscvFunctions
from xdsl.interpreters.riscv_snitch import RiscvSnitchFunctions
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
    pattern_type = snitch_stream.StridePatternType(2)

    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(RiscvFunctions())
    interpreter.register_implementations(SnitchStreamFunctions())
    interpreter.register_implementations(RiscvSnitchFunctions())

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
    )

    assert interpreter.run_op(a_stream_op, (a, stride_pattern)) == (
        StridedPointerInputStream(stride_pattern.offset_expr, a),
    )

    b_stream_op = snitch_stream.StridedReadOp(
        TestSSAValue(register),
        TestSSAValue(pattern_type),
        riscv.Registers.FT1,
        IntAttr(1),
    )

    assert interpreter.run_op(b_stream_op, (b, stride_pattern)) == (
        StridedPointerInputStream(stride_pattern.offset_expr, b),
    )

    c_stream_op = snitch_stream.StridedWriteOp(
        TestSSAValue(register),
        TestSSAValue(pattern_type),
        riscv.Registers.FT2,
        IntAttr(2),
    )

    assert interpreter.run_op(c_stream_op, (c, stride_pattern)) == (
        StridedPointerOutputStream(stride_pattern.offset_expr, c),
    )

    streaming_region_body = Region(
        Block(
            arg_types=(
                stream.ReadableStreamType(riscv.Registers.FT0),
                stream.ReadableStreamType(riscv.Registers.FT1),
                stream.WritableStreamType(riscv.Registers.FT2),
            )
        )
    )

    with ImplicitBuilder(streaming_region_body) as (a_stream, b_stream, c_stream):
        count_reg = riscv.LiOp(6).rd

        frep_body = Region(Block())

        with ImplicitBuilder(frep_body):
            a_reg = riscv_snitch.ReadOp(a_stream).res
            b_reg = riscv_snitch.ReadOp(b_stream).res
            c_reg = riscv.FAddDOp(a_reg, b_reg, rd=riscv.Registers.FT2).rd
            riscv_snitch.WriteOp(c_reg, c_stream)

        riscv_snitch.FrepOuter(count_reg, frep_body)

    assert (
        interpreter.run_op(
            snitch_stream.StreamingRegionOp(
                (TestSSAValue(register), TestSSAValue(register)),
                (TestSSAValue(register),),
                (stride_pattern_op.pattern,),
                streaming_region_body,
            ),
            (a, b, c, stride_pattern),
        )
        == ()
    )

    assert c.float64.get_list(6) == [5.0] * 6
