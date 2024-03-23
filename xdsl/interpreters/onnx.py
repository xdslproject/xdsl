from typing import Any, cast

import numpy as np

from xdsl.dialects import onnx
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl,
    register_impls,
)
from xdsl.interpreters import ptr
from xdsl.interpreters.shaped_array import ShapedArray


def to_dtype(
    xtype: ptr.XType[int] | ptr.XType[float],
) -> type[np.int32] | type[np.int64] | type[np.float32] | type[np.float64]:
    match xtype.format:
        case "<i":
            return np.int32
        case "<I":
            return np.int64
        case "<f":
            return np.float32
        case "<d":
            return np.float64
        case _:
            raise NotImplementedError()


def from_dtype(
    dtype: type[np.float32] | type[np.float64] | type[np.int32] | type[np.int64],
) -> ptr.XType[float] | ptr.XType[int]:
    if dtype == np.float32:
        return ptr.float32
    elif dtype == np.float64:
        return ptr.float64
    elif dtype == np.float32:
        return ptr.int32
    elif dtype == np.float64:
        return ptr.int64
    else:
        raise NotImplementedError()


def to_ndarray(
    shaped_array: ShapedArray[int] | ShapedArray[float],
) -> np.ndarray[Any, np.dtype[np.float64 | np.float32 | np.int64 | np.int32]]:
    dtype = to_dtype(shaped_array.data_ptr.xtype)
    flat = np.frombuffer(shaped_array.data_ptr.raw.memory, dtype)
    shaped = flat.reshape(shaped_array.shape)
    return shaped


def from_ndarray(
    ndarray: np.ndarray[
        Any,
        np.dtype[np.float32]
        | np.dtype[np.float64]
        | np.dtype[np.int32]
        | np.dtype[np.int64],
    ]
) -> ShapedArray[float] | ShapedArray[int]:
    return ShapedArray(
        ptr.TypedPtr(
            ptr.RawPtr(bytearray(ndarray.data)),
            xtype=from_dtype(ndarray.dtype.type),
        ),
        list(ndarray.shape),
    )


@register_impls
class OnnxFunctions(InterpreterFunctions):
    @impl(onnx.Add)
    def run_add(self, interpreter: Interpreter, op: onnx.Add, args: tuple[Any, ...]):
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        result = to_ndarray(lhs) + to_ndarray(rhs)
        return (from_ndarray(result),)

    @impl(onnx.Sub)
    def run_sub(self, interpreter: Interpreter, op: onnx.Sub, args: tuple[Any, ...]):
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        result = to_ndarray(lhs) - to_ndarray(rhs)
        return (from_ndarray(result),)

    @impl(onnx.Mul)
    def run_mul(self, interpreter: Interpreter, op: onnx.Mul, args: tuple[Any, ...]):
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        result = to_ndarray(lhs) * to_ndarray(rhs)
        return (from_ndarray(result),)

    @impl(onnx.Div)
    def run_div(self, interpreter: Interpreter, op: onnx.Div, args: tuple[Any, ...]):
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        result = to_ndarray(lhs) / to_ndarray(rhs)
        return (from_ndarray(result),)

    @impl(onnx.Relu)
    def run_relu(self, interpreter: Interpreter, op: onnx.Relu, args: tuple[Any, ...]):
        operand = args[0]
        assert isinstance(operand, ShapedArray)
        operand = cast(ShapedArray[float], operand)
        operand_data = to_ndarray(operand)
        result = operand_data * (operand_data > 0)
        return (from_ndarray(result),)
