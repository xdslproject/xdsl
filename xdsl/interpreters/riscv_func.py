from typing import Any

from xdsl.dialects import riscv_func
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    TerminatorValue,
    impl,
    impl_terminator,
    register_impls,
)
from xdsl.interpreters.riscv import RiscvFunctions


@register_impls
class RiscvFuncFunctions(InterpreterFunctions):
    @impl_terminator(riscv_func.ReturnOp)
    def run_return(
        self, interpreter: Interpreter, op: riscv_func.ReturnOp, args: tuple[Any, ...]
    ) -> tuple[TerminatorValue, PythonValues]:
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        return ReturnedValues(args), ()

    @impl(riscv_func.CallOp)
    def run_call(
        self, interpreter: Interpreter, op: riscv_func.CallOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = interpreter.call_op(op.callee.string_value(), args)
        results = RiscvFunctions.set_reg_values(interpreter, op.results, results)
        return results
