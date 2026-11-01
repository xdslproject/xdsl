import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from xdsl.dialects import riscv, riscv_snitch
from xdsl.dialects.builtin import (
    AnyFloat,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    VectorType,
    f16,
    f32,
)
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    ReturnedValues,
    impl,
    impl_terminator,
    register_impls,
)
from xdsl.interpreters.riscv import RiscvFunctions
from xdsl.interpreters.snitch_stream import StridedPointerInputStream
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.hints import isa

PACKED_F32_TYPE = VectorType(f32, [2])
"""Type of the value held by a float register that stores two packed f32 values."""

PACKED_F16_TYPE = VectorType(f16, [4])
"""Type of the value held by a float register that stores four packed f16 values."""


def _round_to_element_type(value: float, element_type: AnyFloat) -> float:
    """
    Round a Python float to the given float type, returning infinity on overflow like
    the hardware does.
    """
    try:
        (rounded,) = element_type.unpack(element_type.pack((value,)), 1)
    except OverflowError:
        return math.copysign(math.inf, value)
    return rounded


def create_packed_value(
    vector_type: VectorType[AnyFloat], values: Sequence[float]
) -> DenseIntOrFPElementsAttr[AnyFloat]:
    """
    Create the value held by a float register with packed lanes, rounding each lane to
    the element type of `vector_type`.
    """
    element_type = vector_type.get_element_type()
    return DenseIntOrFPElementsAttr.from_list(
        vector_type,
        tuple(_round_to_element_type(value, element_type) for value in values),
    )


def get_packed_values(
    vector_type: VectorType[AnyFloat], value: Any
) -> tuple[float, ...]:
    """
    Return the lanes of a value held by a float register with packed lanes.
    """
    if not isa(value, DenseIntOrFPElementsAttr[AnyFloat]) or value.type != vector_type:
        raise InterpretationError(
            f"Expected packed value of type {vector_type}, got {value}"
        )
    return value.get_values()


def _run_packed_binary_op(
    interpreter: Interpreter,
    op: riscv.RdRsRsFloatOperationWithFastMath,
    args: tuple[Any, ...],
    vector_type: VectorType[AnyFloat],
    fn: Callable[[float, float], float],
) -> tuple[Any, ...]:
    args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
    lhs = get_packed_values(vector_type, args[0])
    rhs = get_packed_values(vector_type, args[1])
    results = (create_packed_value(vector_type, tuple(map(fn, lhs, rhs))),)
    return RiscvFunctions.set_reg_values(interpreter, op.results, results)


@register_impls
@dataclass
class RiscvSnitchFunctions(InterpreterFunctions):
    @impl(riscv_snitch.FrepOuterOp)
    def run_frep_outer(
        self,
        interpreter: Interpreter,
        op: riscv_snitch.FrepOuterOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        count_minus_one, *loop_args = args
        loop_args = tuple(loop_args)

        for _ in range(count_minus_one.value.data + 1):
            loop_args = interpreter.run_ssacfg_region(op.body, loop_args, "frep.o")

        return loop_args

    @impl_terminator(riscv_snitch.FrepYieldOp)
    def run_yield(
        self,
        interpreter: Interpreter,
        op: riscv_snitch.FrepYieldOp,
        args: tuple[Any, ...],
    ):
        return ReturnedValues(args), ()

    @impl(riscv_snitch.ReadOp)
    def run_read(
        self,
        interpreter: Interpreter,
        op: riscv_snitch.ReadOp,
        args: tuple[Any, ...],
    ):
        (stream,) = args
        stream: StridedPointerInputStream = stream

        value = stream.read()

        RiscvFunctions.set_reg_value(interpreter, op.res.type, value)

        return (value,)

    @impl(riscv_snitch.WriteOp)
    def run_write(
        self,
        interpreter: Interpreter,
        op: riscv_snitch.WriteOp,
        args: tuple[Any, ...],
    ):
        value, stream = args

        value = RiscvFunctions.get_reg_value(interpreter, op.value.type, value)

        stream.write(value)

        return ()

    # region Snitch Packed SIMD Extension

    @impl(riscv_snitch.VFCpkASSOp)
    def run_vfcpka_s_s(
        self,
        interpreter: Interpreter,
        op: riscv_snitch.VFCpkASSOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        lo, hi = args
        assert isa(lo, FloatAttr)
        assert isa(hi, FloatAttr)
        results = (
            create_packed_value(PACKED_F32_TYPE, (lo.value.data, hi.value.data)),
        )
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv_snitch.VFAddSOp)
    def run_vfadd_s(
        self,
        interpreter: Interpreter,
        op: riscv_snitch.VFAddSOp,
        args: tuple[Any, ...],
    ):
        return _run_packed_binary_op(
            interpreter, op, args, PACKED_F32_TYPE, lambda a, b: a + b
        )

    @impl(riscv_snitch.VFSubSOp)
    def run_vfsub_s(
        self,
        interpreter: Interpreter,
        op: riscv_snitch.VFSubSOp,
        args: tuple[Any, ...],
    ):
        return _run_packed_binary_op(
            interpreter, op, args, PACKED_F32_TYPE, lambda a, b: a - b
        )

    @impl(riscv_snitch.VFMulSOp)
    def run_vfmul_s(
        self,
        interpreter: Interpreter,
        op: riscv_snitch.VFMulSOp,
        args: tuple[Any, ...],
    ):
        return _run_packed_binary_op(
            interpreter, op, args, PACKED_F32_TYPE, lambda a, b: a * b
        )

    @impl(riscv_snitch.VFMaxSOp)
    def run_vfmax_s(
        self,
        interpreter: Interpreter,
        op: riscv_snitch.VFMaxSOp,
        args: tuple[Any, ...],
    ):
        return _run_packed_binary_op(interpreter, op, args, PACKED_F32_TYPE, max)

    @impl(riscv_snitch.VFAddHOp)
    def run_vfadd_h(
        self,
        interpreter: Interpreter,
        op: riscv_snitch.VFAddHOp,
        args: tuple[Any, ...],
    ):
        return _run_packed_binary_op(
            interpreter, op, args, PACKED_F16_TYPE, lambda a, b: a + b
        )

    @impl(riscv_snitch.VFSubHOp)
    def run_vfsub_h(
        self,
        interpreter: Interpreter,
        op: riscv_snitch.VFSubHOp,
        args: tuple[Any, ...],
    ):
        return _run_packed_binary_op(
            interpreter, op, args, PACKED_F16_TYPE, lambda a, b: a - b
        )

    @impl(riscv_snitch.VFMulHOp)
    def run_vfmul_h(
        self,
        interpreter: Interpreter,
        op: riscv_snitch.VFMulHOp,
        args: tuple[Any, ...],
    ):
        return _run_packed_binary_op(
            interpreter, op, args, PACKED_F16_TYPE, lambda a, b: a * b
        )

    # endregion
