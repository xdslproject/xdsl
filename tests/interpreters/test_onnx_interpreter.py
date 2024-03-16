import pytest

from xdsl.dialects import onnx
from xdsl.dialects.builtin import ModuleOp, TensorType, f32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.utils.test_value import TestSSAValue

# it looks for numpy if it's not there it skips the tests
pytest.importorskip("numpy", reason="numpy is an optional dependency in xDSL")
from xdsl.interpreters.onnx import OnnxFunctions  # noqa: I001, E402

interpreter = Interpreter(ModuleOp([]))
interpreter.register_implementations(OnnxFunctions())


def test_onnx_add():
    op = onnx.Add(
        TestSSAValue(TensorType(f32, [2, 3])),
        TestSSAValue(TensorType(f32, [2, 3])),
        res_type=TensorType(f32, [2, 3]),
    )

    a = ShapedArray([1, 2, 3, 4, 5, 6], [2, 3])
    b = ShapedArray([1, 4, 2, 5, 3, 6], [2, 3])

    c = interpreter.run_op(op, (a, b))
    assert c == ShapedArray([2, 6, 5, 9, 8, 12], [2, 3])


def test_onnx_sub():
    op = onnx.Sub(
        TestSSAValue(TensorType(f32, [2, 3])),
        TestSSAValue(TensorType(f32, [2, 3])),
        res_type=TensorType(f32, [2, 3]),
    )

    a = ShapedArray([1, 2, 3, 4, 5, 6], [2, 3])
    b = ShapedArray([1, 4, 2, 5, 3, 6], [2, 3])

    c = interpreter.run_op(op, (a, b))
    assert c == ShapedArray([0, -2, 1, -1, 2, 0], [2, 3])


def test_onnx_mul():
    op = onnx.Mul(
        TestSSAValue(TensorType(f32, [2, 2])),
        TestSSAValue(TensorType(f32, [2, 2])),
        res_type=TensorType(f32, [2, 2]),
    )

    a = ShapedArray([1, 4, 7, 1], [2, 2])
    b = ShapedArray([2, 3, 1, 8], [2, 2])

    c = interpreter.run_op(op, (a, b))
    assert c == ShapedArray([2, 12, 7, 8], [2, 2])


def test_onnx_div():
    op = onnx.Div(
        TestSSAValue(TensorType(f32, [2, 2])),
        TestSSAValue(TensorType(f32, [2, 2])),
        res_type=TensorType(f32, [2, 2]),
    )

    a = ShapedArray([1, 1, 1, 1], [2, 2])
    b = ShapedArray([5, 2, 1, 2], [2, 2])

    c = interpreter.run_op(op, (a, b))
    assert c == ShapedArray([0.2, 0.5, 1.0, 0.5], [2, 2])


def test_onnx_relu():
    op = onnx.Relu(
        TestSSAValue(TensorType(f32, [2, 2])),
    )

    a = ShapedArray([1, 1, 1, 1], [2, 2])
    b = interpreter.run_op(op, (a,))
    assert b == ShapedArray([1, 1, 1, 1], [2, 2])
