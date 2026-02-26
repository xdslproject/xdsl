import math
from collections.abc import Sequence
from typing import Any, cast

from typing_extensions import TypeVar

from xdsl.dialects import tensor
from xdsl.dialects.builtin import DYNAMIC_INDEX, TensorType
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

T = TypeVar("T")


@register_impls
class TensorFunctions(InterpreterFunctions):
    @impl(tensor.EmptyOp)
    def run_empty(
        self, interpreter: Interpreter, op: tensor.EmptyOp, args: tuple[int, ...]
    ) -> tuple[ShapedArray[Any]]:
        result_type = op.tensor.type
        assert isinstance(result_type, TensorType)
        result_shape = list(result_type.get_shape())

        dynamic_dims = iter(args)
        for i in range(len(result_shape)):
            if result_shape[i] == DYNAMIC_INDEX:
                result_shape[i] = next(dynamic_dims)
        assert next(dynamic_dims, None) is None

        xtype = xtype_for_el_type(result_type.element_type, interpreter.index_bitwidth)
        return (
            ShapedArray(
                TypedPtr[Any].new((0,) * math.prod(result_shape), xtype=xtype),
                result_shape,
            ),
        )

    @impl(tensor.ReshapeOp)
    def run_reshape(
        self,
        interpreter: Interpreter,
        op: tensor.ReshapeOp,
        args: tuple[ShapedArray[T], ShapedArray[int]],
    ) -> tuple[ShapedArray[T]]:
        input, new_shape = args
        assert isinstance(input, ShapedArray)
        result_type = op.result.type
        assert isinstance(result_type, TensorType)
        static_shape = list(result_type.get_shape())
        assert static_shape is not None
        if static_shape != new_shape.data:
            raise InterpretationError("Mismatch between static shape and new shape")
        result = ShapedArray(input.data_ptr, static_shape)
        return (result,)

    @impl(tensor.InsertOp)
    def run_insert(
        self,
        interpreter: Interpreter,
        op: tensor.InsertOp,
        args: tuple[T | ShapedArray[T] | int, ...],
    ) -> tuple[ShapedArray[T]]:
        value = cast(T, args[0])
        dest = cast(ShapedArray[T], args[1])
        indices = cast(Sequence[int], args[2:])

        assert isinstance(dest, ShapedArray)
        assert len(indices) == len(dest.shape)

        result = dest.copy()
        result.store(indices, value)

        return (result,)

    @impl(tensor.ExtractOp)
    def run_extract(
        self,
        interpreter: Interpreter,
        op: tensor.ExtractOp,
        args: tuple[ShapedArray[T] | int, ...],
    ) -> tuple[T]:
        tensor = cast(ShapedArray[T], args[0])
        indices = cast(Sequence[int], args[1:])

        assert isinstance(tensor, ShapedArray)
        assert len(indices) == len(tensor.shape)

        return (tensor.load(indices),)

    @impl(tensor.DimOp)
    def run_dim(
        self,
        interpreter: Interpreter,
        op: tensor.DimOp,
        args: tuple[ShapedArray[T] | int, ...],
    ) -> tuple[int]:
        tensor = cast(ShapedArray[T], args[0])
        dim = cast(int, args[1])

        assert isinstance(tensor, ShapedArray)

        return (tensor.shape[dim],)
