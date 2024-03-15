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
        lhs = cast(ShapedArray[int], lhs)
        rhs = cast(ShapedArray[int], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) + np.array(rhs.data)
        return (ShapedArray(result, lhs.shape),)

    @impl(onnx.Sub)
    def run_sub(
        self, interpreter: Interpreter, op: onnx.Sub, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[int], lhs)
        rhs = cast(ShapedArray[int], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) - np.array(rhs.data)
        return (ShapedArray(result, lhs.shape),)

    @impl(onnx.Mul)
    def run_mul(
        self, interpreter: Interpreter, op: onnx.Mul, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[int], lhs)
        rhs = cast(ShapedArray[int], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) * np.array(rhs.data)
        return (ShapedArray(result, lhs.shape),)

    @impl(onnx.Div)
    def run_div(
        self, interpreter: Interpreter, op: onnx.Div, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[int], lhs)
        rhs = cast(ShapedArray[int], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) / np.array(rhs.data)
        return (ShapedArray(result, lhs.shape),)

    @impl(onnx.Relu)
    def run_relu(
        self, interpreter: Interpreter, op: onnx.Relu, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        operand = args[0]
        assert isinstance(operand, ShapedArray)
        operand = cast(ShapedArray[int], operand)
        operand_data = np.array(operand.data)
        result = np.maximum(np.zeros_like(operand.data), operand_data)
        return (ShapedArray(result, operand.shape),)
