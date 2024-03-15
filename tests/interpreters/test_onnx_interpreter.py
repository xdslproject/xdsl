import numpy as np
import pytest

from xdsl.dialects import onnx
from xdsl.dialects.builtin import ModuleOp, TensorType, f32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.onnx import OnnxFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.utils.test_value import TestSSAValue

pytest.importorskip("numpy", reason="numpy is an optional dependency in xDSL")
pytest.importorskip("wgpu", reason="wgpu is an optional dependency")

interpreter = Interpreter(ModuleOp([]))


interpreter.register_implementations(OnnxFunctions())


def test_onnx_add():
    op = onnx.Add(
        TestSSAValue(TensorType(f32, [2, 3])),
        TestSSAValue(TensorType(f32, [2, 3])),
        res_type=TensorType(f32, [2, 3]),
    )

    a = ShapedArray(np.array([1, 2, 3, 4, 5, 6]), [2, 3])
    b = ShapedArray(np.array([1, 4, 2, 5, 3, 6]), [2, 3])
    c = ShapedArray(np.array([0, 0, 0, 0, 0, 0]), [2, 3])

    c = interpreter.run_op(op, (a, b, c))
    assert len(c) == 1
    assert np.array_equal(c[0].data[0][0], np.array([2, 6, 5, 9, 8, 12]))
    assert c[0].shape == [2, 3]


def test_onnx_sub():
    op = onnx.Sub(
        TestSSAValue(TensorType(f32, [2, 3])),
        TestSSAValue(TensorType(f32, [2, 3])),
        res_type=TensorType(f32, [2, 3]),
    )

    a = ShapedArray(np.array([1, 2, 3, 4, 5, 6]), [2, 3])
    b = ShapedArray(np.array([1, 4, 2, 5, 3, 6]), [2, 3])
    c = ShapedArray(np.array([0, 0, 0, 0, 0, 0]), [2, 3])

    c = interpreter.run_op(op, (a, b, c))
    assert len(c) == 1
    assert np.array_equal(c[0].data[0][0], np.array([0, -2, 1, -1, 2, 0]))
    assert c[0].shape == [2, 3]


def test_onnx_mul():
    op = onnx.Mul(
        TestSSAValue(TensorType(f32, [2, 2])),
        TestSSAValue(TensorType(f32, [2, 2])),
        res_type=TensorType(f32, [2, 2]),
    )

    a = ShapedArray(np.array([1, 4, 7, 1]), [2, 2])
    b = ShapedArray(np.array([2, 3, 1, 8]), [2, 2])
    c = ShapedArray(np.array([0, 0, 0, 0]), [2, 2])

    c = interpreter.run_op(op, (a, b, c))
    assert len(c) == 1
    assert np.array_equal(c[0].data[0][0], np.array([2, 12, 7, 8]))
    assert c[0].shape == [2, 2]


def test_onnx_div():
    op = onnx.Div(
        TestSSAValue(TensorType(f32, [2, 2])),
        TestSSAValue(TensorType(f32, [2, 2])),
        res_type=TensorType(f32, [2, 2]),
    )

    a = ShapedArray(np.array([1, 1, 1, 1]), [2, 2])
    b = ShapedArray(np.array([5, 2, 1, 2]), [2, 2])
    c = ShapedArray(np.array([0, 0, 0, 0]), [2, 2])

    c = interpreter.run_op(op, (a, b, c))
    assert len(c) == 1
    assert np.array_equal(c[0].data[0][0], np.array([0.2, 0.5, 1.0, 0.5]))
    assert c[0].shape == [2, 2]


def test_onnx_relu():
    op = onnx.Relu(
        TestSSAValue(TensorType(f32, [2, 2])),
    )

    a = ShapedArray(np.array([1, 1, 1, 1]), [2, 2])
    b = interpreter.run_op(op, (a,))
    assert (b[0].data[0] == np.array([1, 1, 1, 1])).all()
    assert len(b) == 1
    assert b[0].shape == [2, 2]
