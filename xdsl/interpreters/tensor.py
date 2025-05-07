import math
from typing import Any, cast

from xdsl.dialects import tensor
from xdsl.dialects.builtin import TensorType
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl,
    register_impls,
)
from xdsl.interpreters.builtin import xtype_for_el_type
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.utils.exceptions import InterpretationError


@register_impls
class TensorFunctions(InterpreterFunctions):
    @impl(tensor.EmptyOp)
    def run_empty(
        self, interpreter: Interpreter, op: tensor.EmptyOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        result_type = op.tensor.type
        assert isinstance(result_type, TensorType)
        result_shape = list(result_type.get_shape())
        xtype = xtype_for_el_type(result_type.element_type, interpreter.index_bitwidth)
        return (
            ShapedArray(
                TypedPtr[Any].new((0,) * math.prod(result_shape), xtype=xtype),
                result_shape,
            ),
        )

    @impl(tensor.ReshapeOp)
    def run_reshape(
        self, interpreter: Interpreter, op: tensor.ReshapeOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        input, new_shape = args
        assert isinstance(input, ShapedArray)
        input = cast(ShapedArray[float], input)
        result_type = op.result.type
        assert isinstance(result_type, TensorType)
        static_shape = list(result_type.get_shape())
        assert static_shape is not None
        if static_shape != new_shape.data:
            raise InterpretationError("Mismatch between static shape and new shape")
        result = ShapedArray(input.data_ptr, static_shape)
        return (result,)
