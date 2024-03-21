from typing import Any, cast

from xdsl.dialects import tensor
from xdsl.dialects.builtin import TensorType
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl,
    register_impls,
)
from xdsl.interpreters.shaped_array import ShapedArray


@register_impls
class TensorFunctions(InterpreterFunctions):

    @impl(tensor.EmptyOp)
    def run_empty(
        self, interpreter: Interpreter, op: tensor.EmptyOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        result_type = op.tensor.type
        assert isinstance(result_type, TensorType)
        result_shape = list(result_type.get_shape())
        return (ShapedArray([None], result_shape),)

    @impl(tensor.ReshapeOp)
    def run_reshape(
        self, interpreter: Interpreter, op: tensor.ReshapeOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        operand = args[0]
        assert isinstance(operand, ShapedArray)
        operand = cast(ShapedArray[float], operand)
        result_type = op.result.type
        assert isinstance(result_type, TensorType)
        new_shape = list(result_type.get_shape())
        return (ShapedArray(list(operand.data), new_shape),)
