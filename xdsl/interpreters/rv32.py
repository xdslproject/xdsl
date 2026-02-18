from __future__ import annotations

from typing import Any

from xdsl.dialects import rv32
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)
from xdsl.interpreters.riscv import RiscvFunctions
from xdsl.utils.exceptions import InterpretationError


@register_impls
class Rv32Functions(InterpreterFunctions):
    @impl(rv32.LiOp)
    def run_li(
        self,
        interpreter: Interpreter,
        op: rv32.LiOp,
        args: tuple[Any, ...],
    ):
        results = (RiscvFunctions.get_immediate_value(interpreter, op.immediate),)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(rv32.GetRegisterOp)
    def run_get_register(
        self, interpreter: Interpreter, op: rv32.GetRegisterOp, args: PythonValues
    ) -> PythonValues:
        attr = op.res.type

        if not attr.is_allocated:
            raise InterpretationError(
                f"Cannot get value for unallocated register {attr}"
            )

        name = attr.register_name

        registers = RiscvFunctions.registers(interpreter)

        if name not in registers:
            raise InterpretationError(f"Value not found for register name {name.data}")

        stored_value = registers[name]

        return (stored_value,)
