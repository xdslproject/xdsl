from __future__ import annotations

from typing import Any

from xdsl.dialects import rv64
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
class Rv64Functions(InterpreterFunctions):
    @impl(rv64.LiOp)
    def run_li(
        self,
        interpreter: Interpreter,
        op: rv64.LiOp,
        args: tuple[Any, ...],
    ):
        results = (RiscvFunctions.get_immediate_value(interpreter, op.immediate),)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(rv64.GetRegisterOp)
    def run_get_register(
        self, interpreter: Interpreter, op: rv64.GetRegisterOp, args: PythonValues
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
