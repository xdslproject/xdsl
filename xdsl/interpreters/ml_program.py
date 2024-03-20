from typing import Any

from xdsl.dialects import ml_program
from xdsl.dialects.builtin import TensorType
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl,
    register_impls,
)
from xdsl.interpreters.shaped_array import ShapedArray


@register_impls
class MLProgramFunctions(InterpreterFunctions):

    @impl(ml_program.Global)
    def run_global(
        self, interpreter: Interpreter, op: ml_program.Global, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        if op.is_mutable is not None:
            raise NotImplementedError(
                "mutable global not yet supported in ml_program.global interpreter"
            )
        # FIX ME
        result_type = op.type
        assert isinstance(result_type, TensorType)
        result_shape = result_type.get_shape()[0]
        return (ShapedArray([], [result_shape]),)

    @impl(ml_program.GlobalLoadConstant)
    def run_global_load_constant(
        self,
        interpreter: Interpreter,
        op: ml_program.GlobalLoadConstant,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)
        result_shape = result_type.get_shape()[0]
        return (ShapedArray(list([None]), [result_shape]),)
