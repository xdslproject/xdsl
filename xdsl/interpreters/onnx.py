from typing import Any, cast

import numpy as np

from xdsl.dialects import onnx
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl,
    register_impls,
)
from xdsl.interpreters.shaped_array import ShapedArray


@register_impls
class OnnxFunctions(InterpreterFunctions):
    @impl(onnx.Add)
    def run_add(
        self, interpreter: Interpreter, op: onnx.Add, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) + np.array(rhs.data)
        return (ShapedArray(list(result), lhs.shape),)

    @impl(onnx.Sub)
    def run_sub(
        self, interpreter: Interpreter, op: onnx.Sub, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) - np.array(rhs.data)
        return (ShapedArray(list(result), lhs.shape),)

    @impl(onnx.Mul)
    def run_mul(
        self, interpreter: Interpreter, op: onnx.Mul, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) * np.array(rhs.data)
        return (ShapedArray(list(result), lhs.shape),)

    @impl(onnx.Div)
    def run_div(
        self, interpreter: Interpreter, op: onnx.Div, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) / np.array(rhs.data)
        return (ShapedArray(list(result), lhs.shape),)

    @impl(onnx.Relu)
    def run_relu(
        self, interpreter: Interpreter, op: onnx.Relu, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        operand = args[0]
        assert isinstance(operand, ShapedArray)
        operand = cast(ShapedArray[float], operand)
        operand_data = np.array(operand.data)
        result = operand_data * (operand_data > 0)
        return (ShapedArray(list(result), operand.shape),)
