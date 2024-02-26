from xdsl.builder import ImplicitBuilder
from xdsl.dialects import riscv, riscv_snitch, snitch_stream, stream
from xdsl.dialects.builtin import ArrayAttr, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.riscv import RawPtr, RiscvFunctions
from xdsl.interpreters.riscv_snitch import RiscvSnitchFunctions
from xdsl.interpreters.snitch_stream import (
    SnitchStreamFunctions,
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
    assert indexing_map_from_bounds([3, 2]) == AffineMap(
        1, 0, (AffineExpr.dimension(0).floor_div(3) % 2, AffineExpr.dimension(0) % 3)
    )
    assert indexing_map_from_bounds([4, 3, 2]) == AffineMap(
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
    assert offset_map_from_strides([1, 2]) == AffineMap.from_callable(
        lambda i, j: (i * 2 + j * 1,)
    )
    assert offset_map_from_strides([1, 2, 3]) == AffineMap.from_callable(
        lambda i, j, k: (i * 3 + j * 2 + k * 1,)
    )


def test_snitch_stream_interpreter():
    register = riscv.IntRegisterType.unallocated()

    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(RiscvFunctions())
    interpreter.register_implementations(SnitchStreamFunctions())
    interpreter.register_implementations(RiscvSnitchFunctions())

    a = RawPtr.new_float64([2.0] * 6)
    b = RawPtr.new_float64([3.0] * 6)
    c = RawPtr.new_float64([4.0] * 6)

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
                ArrayAttr(
                    (
                        snitch_stream.StridePattern.from_bounds_and_strides(
                            [3, 2], [8, 24]
                        ),
                    )
                ),
                streaming_region_body,
            ),
            (a, b, c),
        )
        == ()
    )

    assert c.float64.get_list(6) == [5.0] * 6
