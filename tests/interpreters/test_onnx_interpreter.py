from xdsl.dialects import onnx
from xdsl.dialects.builtin import ModuleOp, TensorType, f32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.onnx import OnnxFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.utils.test_value import TestSSAValue

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
    c = ShapedArray([2, 6, 5, 9, 8, 12], [2, 3])

    interpreter.run_op(op, (a, b, c))
    assert c.data == [2, 6, 5, 9, 8, 12]


def test_onnx_sub():
    op = onnx.Sub(
        TestSSAValue(TensorType(f32, [2, 3])),
        TestSSAValue(TensorType(f32, [2, 3])),
        res_type=TensorType(f32, [2, 3]),
    )

    a = ShapedArray([1, 2, 3, 4, 5, 6], [2, 3])
    b = ShapedArray([1, 4, 2, 5, 3, 6], [2, 3])
    c = ShapedArray([0, -2, 1, -1, 2, 0], [2, 3])

    interpreter.run_op(op, (a, b, c))
    assert c.data == [0, -2, 1, -1, 2, 0]


def test_onnx_mul():
    op = onnx.Mul(
        TestSSAValue(TensorType(f32, [2, 2])),
        TestSSAValue(TensorType(f32, [2, 2])),
        res_type=TensorType(f32, [2, 2]),
    )

    a = ShapedArray([1, 0, 1, 0], [2, 2])
    b = ShapedArray([1, 1, 1, 1], [2, 2])
    c = ShapedArray([1, 0, 1, 0], [2, 2])

    interpreter.run_op(op, (a, b, c))
    assert c.data == [1, 0, 1, 0]


def test_onnx_div():
    op = onnx.Div(
        TestSSAValue(TensorType(f32, [2, 2])),
        TestSSAValue(TensorType(f32, [2, 2])),
        res_type=TensorType(f32, [2, 2]),
    )

    a = ShapedArray([1, 1, 1, 1], [2, 2])
    b = ShapedArray([5, 2, 1, 2], [2, 2])
    c = ShapedArray([0.2, 0.5, 0.5, 0.5], [2, 2])

    interpreter.run_op(op, (a, b, c))
    assert c.data == [0.2, 0.5, 0.5, 0.5]


def test_onnx_relu():
    op = onnx.Relu(
        TestSSAValue(TensorType(f32, [2, 2])),
    )

    a = ShapedArray([0, 0, 0, 0], [2, 2])
    b = ShapedArray([1, 1, 1, 1], [2, 2])
    interpreter.run_op(op, (a, b))
    assert b.data == [1, 1, 1, 1]
