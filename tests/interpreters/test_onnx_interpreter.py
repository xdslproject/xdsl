import pytest

from xdsl.dialects import onnx
from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    ModuleOp,
    TensorType,
    f32,
    i64,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.builtin import BuiltinFunctions
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

    a = ShapedArray([1, 2, 3, 4, 5, 6], [2, 3])
    b = ShapedArray([1, 4, 2, 5, 3, 6], [2, 3])

    c = interpreter.run_op(op, (a, b))
    assert c == ShapedArray([2, 6, 5, 9, 8, 12], [2, 3])


def test_onnx_sub():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
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
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
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
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
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
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Relu(
        TestSSAValue(TensorType(f32, [2, 2])),
    )

    a = ShapedArray([1, 1, 1, 1], [2, 2])
    b = interpreter.run_op(op, (a,))
    assert b == ShapedArray([1, 1, 1, 1], [2, 2])


def test_onnx_constant():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    interpreter.register_implementations(BuiltinFunctions())
    op = onnx.Constant(
        (
            DenseIntOrFPElementsAttr.create_dense_int(
                TensorType(i64, [4]), [5, 5, 16, 2]
            )
        ),
        None,
        None,
        None,
        None,
        None,
        None,
        output_type=TensorType(i64, [4]),
    )

    a = interpreter.run_op(op, ())
    assert a == ShapedArray([5, 5, 16, 2], [4])


def test_onnx_reshape():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Reshape(
        (TestSSAValue(TensorType(f32, [1, 10]))),
        (TestSSAValue(TensorType(i64, [2]))),
        AnyIntegerAttr(0, i64),
    )
    a = ShapedArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 10])
    b = ShapedArray([], [2])
    c = interpreter.run_op(op, (a, b))
    assert c == ShapedArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 10])


def test_onnx_gemm():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Gemm(
        TestSSAValue(TensorType(f32, [2, 2])),
        TestSSAValue(TensorType(f32, [2, 2])),
        TestSSAValue(TensorType(f32, [2, 2])),
        FloatAttr(1, f32),
        AnyIntegerAttr(0, i64),
        AnyIntegerAttr(0, i64),
        FloatAttr(1, f32),
    )

    a = ShapedArray([1, 2, 3, 4], [2, 2])
    b = ShapedArray([2, 4, 6, 8], [2, 2])
    c = ShapedArray([1, 1, 1, 1], [2, 2])
    d = interpreter.run_op(op, (a, b, c))
    assert d == ShapedArray([61, 61, 61, 61], [2, 2])


def test_onnx_gemm_transpose_b():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Gemm(
        TestSSAValue(TensorType(f32, [2, 1])),
        TestSSAValue(TensorType(f32, [2, 1])),
        TestSSAValue(TensorType(f32, [2, 2])),
        FloatAttr(1, f32),
        AnyIntegerAttr(0, i64),
        AnyIntegerAttr(1, i64),
        FloatAttr(1, f32),
    )

    a = ShapedArray([1, 2], [2, 1])
    b = ShapedArray([4, 9], [2, 1])
    c = ShapedArray([1, 2, 3, 4], [2, 2])
    d = interpreter.run_op(op, (a, b, c))
    assert d == ShapedArray([23, 24, 25, 26], [2, 2])
