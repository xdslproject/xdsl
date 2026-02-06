import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import riscv, riscv_snitch, rv32, snitch, snitch_stream
from xdsl.dialects.builtin import ArrayAttr, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.riscv import RiscvFunctions
from xdsl.interpreters.riscv_snitch import RiscvSnitchFunctions
from xdsl.interpreters.rv32 import Rv32Functions
from xdsl.interpreters.snitch_stream import SnitchStreamFunctions
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.ir import Block, Region
from xdsl.utils.test_value import create_ssa_value


@pytest.mark.parametrize(
    "ub, strides, offsets",
    [
        ((6,), (1,), tuple(range(6))),
        ((6,), (2,), tuple(range(0, 12, 2))),
        ((3, 2), (2, 1), tuple(range(6))),
        ((4, 3, 2), (6, 2, 1), tuple(range(24))),
    ],
)
def test_stride_pattern_offsets(
    ub: tuple[int, ...], strides: tuple[int, ...], offsets: tuple[int, ...]
):
    assert (
        snitch_stream.StridePattern.from_bounds_and_strides(ub, strides).offsets()
        == offsets
    )


@pytest.mark.parametrize(
    "inputs, outputs",
    [
        (((1, 1), (1, 1)), ((1,), (1,))),
        (
            ((1, 1), (2, 3)),
            ((1,), (3,)),
        ),  # Technically any number could be in the second position, since the bound is 1
        (((24,), (1,)), ((24,), (1,))),
        (((2, 3, 4), (12, 4, 1)), ((24,), (1,))),
        (((3, 2), (16, 8)), ((6,), (8,))),
        (((3, 2), (8, 0)), ((3,), (8,), 2)),
        (((3, 2), (8, 0), 5), ((3,), (8,), 10)),
        (((3, 2), (0, 8)), ((3, 2), (0, 8))),
        (((1, 1, 6, 1, 3, 3), (6, 5, 4, 3, 2, 1)), ((6, 3, 3), (4, 2, 1))),
    ],
)
def test_simplify_stride_pattern(
    inputs: tuple[tuple[int, ...], tuple[int, ...]],
    outputs: tuple[tuple[int, ...], tuple[int, ...]],
):
    assert snitch_stream.StridePattern.from_bounds_and_strides(
        *inputs
    ).simplified() == snitch_stream.StridePattern.from_bounds_and_strides(*outputs)


def test_snitch_stream_interpreter():
    register = riscv.Registers.UNALLOCATED_INT

    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(RiscvFunctions())
    interpreter.register_implementations(Rv32Functions())
    interpreter.register_implementations(SnitchStreamFunctions())
    interpreter.register_implementations(RiscvSnitchFunctions())

    a = TypedPtr.new_float64([2.0] * 6)
    b = TypedPtr.new_float64([3.0] * 6)
    c = TypedPtr.new_float64([4.0] * 6)

    streaming_region_body = Region(
        Block(
            arg_types=(
                snitch.ReadableStreamType(riscv.Registers.FT0),
                snitch.ReadableStreamType(riscv.Registers.FT1),
                snitch.WritableStreamType(riscv.Registers.FT2),
            )
        )
    )

    with ImplicitBuilder(streaming_region_body) as (a_stream, b_stream, c_stream):
        count_reg = rv32.LiOp(5).rd

        frep_body = Region(Block())

        with ImplicitBuilder(frep_body):
            a_reg = riscv_snitch.ReadOp(a_stream).res
            b_reg = riscv_snitch.ReadOp(b_stream).res
            c_reg = riscv.FAddDOp(a_reg, b_reg, rd=riscv.Registers.FT2).rd
            riscv_snitch.WriteOp(c_reg, c_stream)

        riscv_snitch.FrepOuterOp(count_reg, frep_body)

    assert (
        interpreter.run_op(
            snitch_stream.StreamingRegionOp(
                (create_ssa_value(register), create_ssa_value(register)),
                (create_ssa_value(register),),
                ArrayAttr(
                    (
                        snitch_stream.StridePattern.from_bounds_and_strides(
                            [2, 3], [24, 8]
                        ),
                    )
                ),
                streaming_region_body,
            ),
            (a.raw, b.raw, c.raw),
        )
        == ()
    )

    assert c.get_list(6) == [5.0] * 6
