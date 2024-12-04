from dataclasses import dataclass
from typing import Any

from xdsl.dialects import riscv_snitch
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

        for _ in range(count_minus_one + 1):
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
