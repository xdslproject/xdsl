import math
from collections.abc import Callable

import pytest

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import func, riscv, riscv_snitch, rv32, snitch
from xdsl.dialects.builtin import FloatAttr, IntegerAttr, ModuleOp, f64, i32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.riscv import RiscvFunctions
from xdsl.interpreters.riscv_snitch import (
    PACKED_F16_TYPE,
    PACKED_F32_TYPE,
    RiscvSnitchFunctions,
    create_packed_value,
)
from xdsl.interpreters.rv32 import Rv32Functions
from xdsl.interpreters.utils.stream import Acc, Nats
from xdsl.ir import BlockArgument
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.test_value import create_ssa_value


def test_read_write():
    interpreter = Interpreter(ModuleOp([]), index_bitwidth=32)
    interpreter.register_implementations(RiscvSnitchFunctions())

    a0 = riscv.Registers.A0
    a1 = riscv.Registers.A1

    input_stream = Nats()
    output_stream = Acc()

    assert interpreter.run_op(
        riscv_snitch.ReadOp(create_ssa_value(snitch.ReadableStreamType(a0)), a0),
        (input_stream,),
    ) == (IntegerAttr(1, i32),)
    assert interpreter.run_op(
        riscv_snitch.ReadOp(create_ssa_value(snitch.ReadableStreamType(a1)), a1),
        (input_stream,),
    ) == (IntegerAttr(2, i32),)

    assert (
        interpreter.run_op(
            riscv_snitch.WriteOp(
                create_ssa_value(a0), create_ssa_value(snitch.ReadableStreamType(a0))
            ),
            (
                IntegerAttr(1, i32),
                output_stream,
            ),
        )
        == ()
    )
    assert output_stream.values == [IntegerAttr(1, i32)]
    assert (
        interpreter.run_op(
            riscv_snitch.WriteOp(
                create_ssa_value(a1), create_ssa_value(snitch.ReadableStreamType(a1))
            ),
            (
                IntegerAttr(2, i32),
                output_stream,
            ),
        )
        == ()
    )
    assert output_stream.values == [IntegerAttr(1, i32), IntegerAttr(2, i32)]


def test_frep_carried_vars():
    float_register = riscv.Registers.UNALLOCATED_FLOAT
    acc_reg_type = riscv.Registers.FT[3]

    @ModuleOp
    @Builder.implicit_region
    def sum_to_for_op():
        with ImplicitBuilder(
            func.FuncOp(
                "sum_to", ((riscv.Registers.UNALLOCATED_INT,), (float_register,))
            ).body
        ) as (count,):
            one = rv32.LiOp(1).rd
            initial = riscv.FCvtDWOp(one, rd=acc_reg_type).rd

            @Builder.implicit_region((float_register,))
            def for_loop_region(args: tuple[BlockArgument, ...]):
                (acc,) = args
                res = riscv.FAddDOp(acc, acc, rd=acc_reg_type)
                riscv_snitch.FrepYieldOp(res)

            result = riscv_snitch.FrepOuterOp(count, for_loop_region, (initial,)).res
            func.ReturnOp(*result)

    interpreter = Interpreter(sum_to_for_op, index_bitwidth=32)
    interpreter.register_implementations(RiscvSnitchFunctions())
    interpreter.register_implementations(RiscvFunctions())
    interpreter.register_implementations(Rv32Functions())
    interpreter.register_implementations(FuncFunctions())

    assert interpreter.call_op("sum_to", (IntegerAttr(5, i32),)) == (
        FloatAttr(64.0, f64),
    )


def _snitch_interpreter() -> Interpreter:
    interpreter = Interpreter(ModuleOp([]), index_bitwidth=32)
    interpreter.register_implementations(RiscvSnitchFunctions())
    return interpreter


def _float_reg():
    return create_ssa_value(riscv.Registers.UNALLOCATED_FLOAT)


def test_vfcpka_s_s():
    interpreter = _snitch_interpreter()
    op = riscv_snitch.VFCpkASSOp(
        _float_reg(), _float_reg(), rd=riscv.Registers.UNALLOCATED_FLOAT
    )

    (packed,) = interpreter.run_op(op, (FloatAttr(0.1, f64), FloatAttr(-2.0, f64)))

    # 0.1 is rounded to the nearest f32
    assert packed == create_packed_value(PACKED_F32_TYPE, (0.1, -2.0))
    assert packed.get_values() == (0.10000000149011612, -2.0)


S_OPS: list[
    tuple[type[riscv.RdRsRsFloatOperationWithFastMath], Callable[[float, float], float]]
] = [
    (riscv_snitch.VFAddSOp, lambda a, b: a + b),
    (riscv_snitch.VFSubSOp, lambda a, b: a - b),
    (riscv_snitch.VFMulSOp, lambda a, b: a * b),
    (riscv_snitch.VFMaxSOp, max),
]

H_OPS: list[
    tuple[type[riscv.RdRsRsFloatOperationWithFastMath], Callable[[float, float], float]]
] = [
    (riscv_snitch.VFAddHOp, lambda a, b: a + b),
    (riscv_snitch.VFSubHOp, lambda a, b: a - b),
    (riscv_snitch.VFMulHOp, lambda a, b: a * b),
]


@pytest.mark.parametrize("op_type, fn", S_OPS)
def test_packed_f32_ops(
    op_type: type[riscv.RdRsRsFloatOperationWithFastMath],
    fn: Callable[[float, float], float],
):
    interpreter = _snitch_interpreter()
    lhs = (1.5, -3.0)
    rhs = (2.25, 4.0)

    (result,) = interpreter.run_op(
        op_type(_float_reg(), _float_reg()),
        (
            create_packed_value(PACKED_F32_TYPE, lhs),
            create_packed_value(PACKED_F32_TYPE, rhs),
        ),
    )

    assert result == create_packed_value(
        PACKED_F32_TYPE, (fn(lhs[0], rhs[0]), fn(lhs[1], rhs[1]))
    )


@pytest.mark.parametrize("op_type, fn", H_OPS)
def test_packed_f16_ops(
    op_type: type[riscv.RdRsRsFloatOperationWithFastMath],
    fn: Callable[[float, float], float],
):
    interpreter = _snitch_interpreter()
    lhs = (1.5, -3.0, 0.5, 8.0)
    rhs = (2.25, 4.0, -0.25, 0.125)

    (result,) = interpreter.run_op(
        op_type(_float_reg(), _float_reg()),
        (
            create_packed_value(PACKED_F16_TYPE, lhs),
            create_packed_value(PACKED_F16_TYPE, rhs),
        ),
    )

    assert result == create_packed_value(
        PACKED_F16_TYPE, tuple(fn(a, b) for a, b in zip(lhs, rhs))
    )


def test_packed_f32_rounding_and_overflow():
    interpreter = _snitch_interpreter()

    (result,) = interpreter.run_op(
        riscv_snitch.VFAddSOp(_float_reg(), _float_reg()),
        (
            create_packed_value(PACKED_F32_TYPE, (1.0, 3e38)),
            create_packed_value(PACKED_F32_TYPE, (2.0**-30, 3e38)),
        ),
    )

    # 1 + 2^-30 is not representable in f32, and 6e38 overflows f32
    assert result.get_values() == (1.0, math.inf)


def test_packed_f16_rounding_and_overflow():
    interpreter = _snitch_interpreter()

    (result,) = interpreter.run_op(
        riscv_snitch.VFMulHOp(_float_reg(), _float_reg()),
        (
            create_packed_value(PACKED_F16_TYPE, (0.1, 300.0, -300.0, 1.0)),
            create_packed_value(PACKED_F16_TYPE, (1.0, 300.0, 300.0, 1.0)),
        ),
    )

    assert result.get_values() == (0.0999755859375, math.inf, -math.inf, 1.0)


def test_packed_op_rejects_wrong_value():
    interpreter = _snitch_interpreter()
    op = riscv_snitch.VFAddSOp(_float_reg(), _float_reg())

    with pytest.raises(InterpretationError, match="Expected packed value"):
        interpreter.run_op(op, (FloatAttr(1.0, f64), FloatAttr(2.0, f64)))

    with pytest.raises(InterpretationError, match="Expected packed value"):
        interpreter.run_op(
            op,
            (
                create_packed_value(PACKED_F16_TYPE, (1.0, 2.0, 3.0, 4.0)),
                create_packed_value(PACKED_F16_TYPE, (1.0, 2.0, 3.0, 4.0)),
            ),
        )
