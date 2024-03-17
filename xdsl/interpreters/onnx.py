from typing import Any, cast

import numpy as np

from xdsl.dialects import onnx
from xdsl.dialects.builtin import TensorType
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    ReturnedValues,
    impl,
    register_impls,
)
from xdsl.interpreters.shaped_array import ShapedArray


@register_impls
class OnnxFunctions(InterpreterFunctions):
    @impl(onnx.Add)
    def run_add(self, interpreter: Interpreter, op: onnx.Add, args: tuple[Any, ...]):
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[int], lhs)
        rhs = cast(ShapedArray[int], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) + np.array(rhs.data)
        return ShapedArray(list(result), lhs.shape)

    @impl(onnx.Sub)
    def run_sub(self, interpreter: Interpreter, op: onnx.Sub, args: tuple[Any, ...]):
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[int], lhs)
        rhs = cast(ShapedArray[int], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) - np.array(rhs.data)
        return ShapedArray(list(result), lhs.shape)

    @impl(onnx.Mul)
    def run_mul(self, interpreter: Interpreter, op: onnx.Mul, args: tuple[Any, ...]):
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[int], lhs)
        rhs = cast(ShapedArray[int], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) * np.array(rhs.data)
        return ShapedArray(list(result), lhs.shape)

    @impl(onnx.Div)
    def run_div(self, interpreter: Interpreter, op: onnx.Div, args: tuple[Any, ...]):
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[int], lhs)
        rhs = cast(ShapedArray[int], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) / np.array(rhs.data)
        return ShapedArray(list(result), lhs.shape)

    @impl(onnx.Relu)
    def run_relu(self, interpreter: Interpreter, op: onnx.Relu, args: tuple[Any, ...]):
        operand = args[0]
        assert isinstance(operand, ShapedArray)
        operand = cast(ShapedArray[int], operand)
        operand_data = np.array(operand.data)
        result = operand_data * (operand_data > 0)
        return ShapedArray(list(result), operand.shape)

    @impl(onnx.Constant)
    def run_constant(
        self, interpreter: Interpreter, op: onnx.Constant, args: tuple[Any, ...]
    ):
        attr_value = list(op.attributes.values())[0]
        constant_data = list(x.value.data for x in attr_value.data.data)
        result_type = op.output.type
        assert isinstance(result_type, TensorType)
        output_shape = list(result_type.get_shape())

        return ShapedArray(constant_data, output_shape)

    @impl(onnx.Reshape)
    def run_reshape(
        self, interpreter: Interpreter, op: onnx.Reshape, args: tuple[Any, ...]
    ):
        operand = args[0]
        assert isinstance(operand, ShapedArray)
        operand = cast(ShapedArray[int], operand)
        result_type = op.reshaped.type
        assert isinstance(result_type, TensorType)
        new_shape = list(result_type.get_shape())
        operand_data = np.array(operand.data)
        return ShapedArray(list(operand_data), new_shape)

    @impl(onnx.Gemm)
    def run_gemm(self, interpreter: Interpreter, op: onnx.Gemm, args: tuple[Any, ...]):
        a, b, c = args[0], args[1], args[2]

        alpha = op.alpha.value.data if op.alpha is not None else 1.0
        beta = op.beta.value.data if op.beta is not None else 1.0

        assert isinstance(a, ShapedArray)
        assert isinstance(b, ShapedArray)
        assert isinstance(c, ShapedArray)

        a = np.array(a.data)
        b = np.array(b.data)
        c = np.array(c.data)

        if op.trans_a is not None and op.trans_a.value.data == 1:
            a = np.transpose(a)

        if op.trans_b is not None and op.trans_b.value.data == 1:
            b = np.transpose(b)

        result = alpha * np.dot(a, b) + beta * c

        result_type = op.res_tensor.type
        result_shape = list(result_type.get_shape())
        assert isinstance(result_type, TensorType)

        return ShapedArray(list(result), result_shape)

    @impl(onnx.Conv)
    def run_conv(self, interpreter: Interpreter, op: onnx.Conv, args: tuple[Any, ...]):
        return args

    @impl(onnx.MaxPoolSingleOut)
    def run_max_pool_single_out(
        self, interpreter: Interpreter, op: onnx.MaxPoolSingleOut, args: tuple[Any, ...]
    ):
        return args

    @impl(onnx.EntryPoint)
    def run_entry_point(
        self, interpreter: Interpreter, op: onnx.EntryPoint, args: tuple[Any, ...]
    ):
        return ReturnedValues(args), ()
