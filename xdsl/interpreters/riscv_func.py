from typing import Any

from xdsl.dialects import riscv_func
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    TerminatorValue,
    impl,
    impl_callable,
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

    @impl_callable(riscv_func.FuncOp)
    def run_func(
        self, interpreter: Interpreter, op: riscv_func.FuncOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        if (first_block := op.body.blocks.first) is None or not first_block.ops:
            return interpreter.call_external(op.sym_name.data, op, args)
        else:
            # Either this is the entry function, and the register values are not set,
            # or this is a result of a call impl, and the registers have already been
            # validated, so it is safe to set them again.
            args = RiscvFunctions.set_reg_values_for_types(
                interpreter, op.function_type.inputs.data, args
            )
            return interpreter.run_ssacfg_region(op.body, args, op.sym_name.data)
