import pytest

from xdsl.dialects import onnx
from xdsl.dialects.builtin import ModuleOp, TensorType, f32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.ptr import TypedPtr
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.utils.test_value import TestSSAValue

pytest.importorskip("numpy", reason="numpy is an optional dependency in xDSL")

from xdsl.interpreters.onnx import OnnxFunctions  # noqa: E402


def test_onnx_add():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Add(
        TestSSAValue(TensorType(f32, [2, 3])),
        TestSSAValue(TensorType(f32, [2, 3])),
        res_type=TensorType(f32, [2, 3]),
    )

    a = ShapedArray(TypedPtr.new_float32([1, 2, 3, 4, 5, 6]), [2, 3])
    b = ShapedArray(TypedPtr.new_float32([1, 4, 2, 5, 3, 6]), [2, 3])

    (c,) = interpreter.run_op(op, (a, b))
    assert c == ShapedArray(TypedPtr.new_float32([2, 6, 5, 9, 8, 12]), [2, 3])


def test_onnx_sub():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Sub(
        TestSSAValue(TensorType(f32, [2, 3])),
        TestSSAValue(TensorType(f32, [2, 3])),
        res_type=TensorType(f32, [2, 3]),
    )

    a = ShapedArray(TypedPtr.new_float32([1, 2, 3, 4, 5, 6]), [2, 3])
    b = ShapedArray(TypedPtr.new_float32([1, 4, 2, 5, 3, 6]), [2, 3])

    (c,) = interpreter.run_op(op, (a, b))
    assert c == ShapedArray(TypedPtr.new_float32([0, -2, 1, -1, 2, 0]), [2, 3])


def test_onnx_mul():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Mul(
        TestSSAValue(TensorType(f32, [2, 2])),
        TestSSAValue(TensorType(f32, [2, 2])),
        res_type=TensorType(f32, [2, 2]),
    )

    a = ShapedArray(TypedPtr.new_float32([1, 4, 7, 1]), [2, 2])
    b = ShapedArray(TypedPtr.new_float32([2, 3, 1, 8]), [2, 2])

    (c,) = interpreter.run_op(op, (a, b))
    assert c == ShapedArray(TypedPtr.new_float32([2, 12, 7, 8]), [2, 2])


def test_onnx_div():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Div(
        TestSSAValue(TensorType(f32, [2, 2])),
        TestSSAValue(TensorType(f32, [2, 2])),
        res_type=TensorType(f32, [2, 2]),
    )

    a = ShapedArray(TypedPtr.new_float32([1, 1, 1, 1]), [2, 2])
    b = ShapedArray(TypedPtr.new_float32([5, 2, 1, 2]), [2, 2])

    (c,) = interpreter.run_op(op, (a, b))
    assert c == ShapedArray(TypedPtr.new_float32([0.2, 0.5, 1.0, 0.5]), [2, 2])


def test_onnx_relu():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Relu(
        TestSSAValue(TensorType(f32, [2, 2])),
    )

    a = ShapedArray(TypedPtr.new_float32([1, 1, 1, 1]), [2, 2])
    (b,) = interpreter.run_op(op, (a,))
    assert b == ShapedArray(TypedPtr.new_float32([1, 1, 1, 1]), [2, 2])
