import pytest

from xdsl.dialects import onnx
from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    ArrayAttr,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    ModuleOp,
    NoneType,
    StringAttr,
    TensorType,
    f32,
    i64,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.builtin import BuiltinFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.utils.exceptions import InterpretationError
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
    b = ShapedArray([1, 10], [2])
    (c,) = interpreter.run_op(op, (a, b))
    assert c == ShapedArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 10])


def test_onnx_reshape_error():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Reshape(
        (TestSSAValue(TensorType(f32, [1, 10]))),
        (TestSSAValue(TensorType(i64, [2]))),
        AnyIntegerAttr(0, i64),
    )
    a = ShapedArray([1, 2, 3, 4], [1, 4])
    b = ShapedArray([2, 2], [2])
    with pytest.raises(
        InterpretationError, match="Mismatch between static shape and new shape"
    ):
        interpreter.run_op(op, (a, b))


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


def test_onnx_gemm_alpha():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Gemm(
        TestSSAValue(TensorType(f32, [2, 1])),
        TestSSAValue(TensorType(f32, [1, 2])),
        TestSSAValue(TensorType(f32, [2, 2])),
        FloatAttr(2, f32),
        AnyIntegerAttr(0, i64),
        AnyIntegerAttr(0, i64),
        FloatAttr(1, f32),
    )

    a = ShapedArray([1, 2], [2, 1])
    b = ShapedArray([4, 9], [1, 2])
    c = ShapedArray([1, 2, 3, 4], [2, 2])
    d = interpreter.run_op(op, (a, b, c))
    assert d == ShapedArray([45, 46, 47, 48], [2, 2])


def test_onnx_conv_no_padding():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Conv(
        TestSSAValue(TensorType(f32, [1, 1, 5, 5])),
        TestSSAValue(TensorType(f32, [1, 1, 3, 3])),
        TestSSAValue(NoneType()),
        StringAttr("NOTSET"),
        ArrayAttr([AnyIntegerAttr(1, i64), AnyIntegerAttr(1, i64)]),
        AnyIntegerAttr(1, i64),
        ArrayAttr([AnyIntegerAttr(3, i64), AnyIntegerAttr(3, i64)]),
        ArrayAttr(
            [
                AnyIntegerAttr(0, i64),
                AnyIntegerAttr(0, i64),
                AnyIntegerAttr(0, i64),
                AnyIntegerAttr(0, i64),
            ]
        ),
        ArrayAttr([AnyIntegerAttr(1, i64), AnyIntegerAttr(1, i64)]),
    )
    a = ShapedArray(list(range(25)), [1, 1, 5, 5])
    b = ShapedArray(
        [
            1,
        ]
        * 9,
        [1, 1, 3, 3],
    )
    c = ShapedArray([0], [1])
    d = interpreter.run_op(op, (a, b, c))

    assert d == ShapedArray([54, 63, 72, 99, 108, 117, 144, 153, 162], [1, 1, 3, 3])


def test_onnx_conv_with_padding():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Conv(
        TestSSAValue(TensorType(f32, [1, 1, 5, 5])),
        TestSSAValue(TensorType(f32, [1, 1, 3, 3])),
        TestSSAValue(NoneType()),
        StringAttr("NOTSET"),
        ArrayAttr([AnyIntegerAttr(1, i64), AnyIntegerAttr(1, i64)]),
        AnyIntegerAttr(1, i64),
        ArrayAttr([AnyIntegerAttr(3, i64), AnyIntegerAttr(3, i64)]),
        ArrayAttr(
            [
                AnyIntegerAttr(1, i64),
                AnyIntegerAttr(1, i64),
                AnyIntegerAttr(1, i64),
                AnyIntegerAttr(1, i64),
            ]
        ),
        ArrayAttr([AnyIntegerAttr(1, i64), AnyIntegerAttr(1, i64)]),
    )
    a = ShapedArray(list(range(25)), [1, 1, 5, 5])
    b = ShapedArray(
        [
            1,
        ]
        * 9,
        [1, 1, 3, 3],
    )
    c = ShapedArray([0], [1])
    d = interpreter.run_op(op, (a, b, c))

    assert d == ShapedArray(
        [
            12.0,
            21.0,
            27.0,
            33.0,
            24.0,
            33.0,
            54.0,
            63.0,
            72.0,
            51.0,
            63.0,
            99.0,
            108.0,
            117.0,
            81.0,
            93.0,
            144.0,
            153.0,
            162.0,
            111.0,
            72.0,
            111.0,
            117.0,
            123.0,
            84.0,
        ],
        [1, 1, 5, 5],
    )


def test_onnx_conv_with_same_lower_strides():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Conv(
        TestSSAValue(TensorType(f32, [1, 1, 5, 5])),
        TestSSAValue(TensorType(f32, [1, 1, 3, 3])),
        TestSSAValue(NoneType()),
        StringAttr("SAME_LOWER"),
        ArrayAttr([AnyIntegerAttr(1, i64), AnyIntegerAttr(1, i64)]),
        AnyIntegerAttr(1, i64),
        ArrayAttr([AnyIntegerAttr(3, i64), AnyIntegerAttr(3, i64)]),
        ArrayAttr(
            [
                AnyIntegerAttr(0, i64),
                AnyIntegerAttr(0, i64),
                AnyIntegerAttr(0, i64),
                AnyIntegerAttr(0, i64),
            ]
        ),
        ArrayAttr([AnyIntegerAttr(2, i64), AnyIntegerAttr(2, i64)]),
    )
    a = ShapedArray(list(range(25)), [1, 1, 5, 5])
    b = ShapedArray(
        [
            1,
        ]
        * 9,
        [1, 1, 3, 3],
    )
    c = ShapedArray([0], [1])
    d = interpreter.run_op(op, (a, b, c))

    assert d == ShapedArray(
        [12.0, 27.0, 24.0, 63.0, 108.0, 81.0, 72.0, 117.0, 84.0], [1, 1, 3, 3]
    )


def test_onnx_conv_with_strides_padding():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Conv(
        TestSSAValue(TensorType(f32, [1, 1, 5, 5])),
        TestSSAValue(TensorType(f32, [1, 1, 3, 3])),
        TestSSAValue(NoneType()),
        StringAttr("NOTSET"),
        ArrayAttr([AnyIntegerAttr(1, i64), AnyIntegerAttr(1, i64)]),
        AnyIntegerAttr(1, i64),
        ArrayAttr([AnyIntegerAttr(3, i64), AnyIntegerAttr(3, i64)]),
        ArrayAttr(
            [
                AnyIntegerAttr(1, i64),
                AnyIntegerAttr(1, i64),
                AnyIntegerAttr(1, i64),
                AnyIntegerAttr(1, i64),
            ]
        ),
        ArrayAttr([AnyIntegerAttr(2, i64), AnyIntegerAttr(2, i64)]),
    )
    a = ShapedArray(list(range(35)), [1, 1, 7, 5])
    b = ShapedArray(
        [
            1,
        ]
        * 9,
        [1, 1, 3, 3],
    )
    c = ShapedArray([0], [1])
    d = interpreter.run_op(op, (a, b, c))

    assert d == ShapedArray(
        [12.0, 27.0, 24.0, 63.0, 108.0, 81.0, 123.0, 198.0, 141.0, 112.0, 177.0, 124.0],
        [1, 1, 4, 3],
    )


def test_onnx_conv_with_strides_no_padding():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Conv(
        TestSSAValue(TensorType(f32, [1, 1, 5, 5])),
        TestSSAValue(TensorType(f32, [1, 1, 3, 3])),
        TestSSAValue(NoneType()),
        StringAttr("NOTSET"),
        ArrayAttr([AnyIntegerAttr(1, i64), AnyIntegerAttr(1, i64)]),
        AnyIntegerAttr(1, i64),
        ArrayAttr([AnyIntegerAttr(3, i64), AnyIntegerAttr(3, i64)]),
        ArrayAttr(
            [
                AnyIntegerAttr(0, i64),
                AnyIntegerAttr(0, i64),
                AnyIntegerAttr(0, i64),
                AnyIntegerAttr(0, i64),
            ]
        ),
        ArrayAttr([AnyIntegerAttr(2, i64), AnyIntegerAttr(2, i64)]),
    )
    a = ShapedArray(list(range(35)), [1, 1, 7, 5])
    b = ShapedArray(
        [
            1,
        ]
        * 9,
        [1, 1, 3, 3],
    )
    c = ShapedArray([0], [1])
    d = interpreter.run_op(op, (a, b, c))

    assert d == ShapedArray([54.0, 72.0, 144.0, 162.0, 234.0, 252.0], [1, 1, 3, 2])


def test_onnx_conv_with_strides_asy_padding():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.Conv(
        TestSSAValue(TensorType(f32, [1, 1, 5, 5])),
        TestSSAValue(TensorType(f32, [1, 1, 3, 3])),
        TestSSAValue(NoneType()),
        StringAttr("NOTSET"),
        ArrayAttr([AnyIntegerAttr(1, i64), AnyIntegerAttr(1, i64)]),
        AnyIntegerAttr(1, i64),
        ArrayAttr([AnyIntegerAttr(3, i64), AnyIntegerAttr(3, i64)]),
        ArrayAttr(
            [
                AnyIntegerAttr(1, i64),
                AnyIntegerAttr(0, i64),
                AnyIntegerAttr(1, i64),
                AnyIntegerAttr(0, i64),
            ]
        ),
        ArrayAttr([AnyIntegerAttr(2, i64), AnyIntegerAttr(2, i64)]),
    )
    a = ShapedArray(list(range(35)), [1, 1, 7, 5])
    b = ShapedArray(
        [
            1,
        ]
        * 9,
        [1, 1, 3, 3],
    )
    c = ShapedArray([0], [1])
    d = interpreter.run_op(op, (a, b, c))

    assert d == ShapedArray(
        [21.0, 33.0, 99.0, 117.0, 189.0, 207.0, 171.0, 183.0], [1, 1, 4, 2]
    )


def test_onnx_max_pool_single_out():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(OnnxFunctions())
    op = onnx.MaxPoolSingleOut(
        TestSSAValue(TensorType(f32, [1, 1, 4, 4])),
        StringAttr("NOTSET"),
        AnyIntegerAttr(0, i64),
        ArrayAttr([AnyIntegerAttr(1, i64), AnyIntegerAttr(1, i64)]),
        ArrayAttr([AnyIntegerAttr(2, i64), AnyIntegerAttr(2, i64)]),
        ArrayAttr(
            [
                AnyIntegerAttr(0, i64),
                AnyIntegerAttr(0, i64),
                AnyIntegerAttr(0, i64),
                AnyIntegerAttr(0, i64),
            ]
        ),
        AnyIntegerAttr(0, i64),
        ArrayAttr([AnyIntegerAttr(2, i64), AnyIntegerAttr(2, i64)]),
    )
    a = ShapedArray(list(range(1, 17)), [1, 1, 4, 4])
    b = interpreter.run_op(op, (a,))

    assert b == ShapedArray([6, 8, 14, 16], [1, 1, 2, 2])
