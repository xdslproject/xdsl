import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import (
    AffineMapAttr,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    MemRefType,
    ModuleOp,
    StringAttr,
    TensorType,
    f32,
    i32,
    i64,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.linalg import LinalgFunctions
from xdsl.interpreters.ptr import TypedPtr
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.utils.test_value import TestSSAValue


def test_unimplemented_inputs():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(LinalgFunctions())

    with pytest.raises(
        NotImplementedError,
        match="library_call not yet supported in linalg.generic interpreter",
    ):
        op = linalg.Generic(
            (),
            (),
            Region(Block([linalg.YieldOp()])),
            (),
            (),
            library_call=StringAttr("hello"),
        )
        op.verify()
        interpreter.run_op(op, ())


def test_linalg_generic():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(LinalgFunctions())
    interpreter.register_implementations(ArithFunctions())

    op = linalg.Generic(
        (
            TestSSAValue(MemRefType(i32, [2, 3])),
            TestSSAValue(MemRefType(i32, [3, 2])),
        ),
        (TestSSAValue(MemRefType(i32, [1, 6])),),
        Region(Block(arg_types=(i32, i32))),
        (
            AffineMapAttr(AffineMap.identity(2)),
            AffineMapAttr(AffineMap.transpose_map()),
            AffineMapAttr(
                AffineMap(
                    2,
                    0,
                    (
                        AffineExpr.constant(0),
                        AffineExpr.dimension(0) * 3 + AffineExpr.dimension(1),
                    ),
                )
            ),
        ),
        (
            linalg.IteratorTypeAttr.parallel(),
            linalg.IteratorTypeAttr.parallel(),
            linalg.IteratorTypeAttr.parallel(),
        ),
    )

    with ImplicitBuilder(op.body) as (a, b):
        c = arith.Muli(a, b).result
        linalg.YieldOp(c)

    a = ShapedArray(TypedPtr.new_int32([1, 2, 3, 4, 5, 6]), [2, 3])
    b = ShapedArray(TypedPtr.new_int32([1, 4, 2, 5, 3, 6]), [3, 2])
    c = ShapedArray(TypedPtr.new_int32([-1, -1, -1, -1, -1, -1]), [1, 6])

    interpreter.run_op(op, (a, b, c))

    assert c.data == [1, 4, 9, 16, 25, 36]


def test_linalg_generic_scalar():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(LinalgFunctions())
    interpreter.register_implementations(ArithFunctions())

    op = linalg.Generic(
        (
            TestSSAValue(MemRefType(i32, [2, 3])),
            TestSSAValue(i32),
        ),
        (TestSSAValue(MemRefType(i32, [1, 6])),),
        Region(Block(arg_types=(i32, i32))),
        (
            AffineMapAttr(AffineMap.identity(2)),
            AffineMapAttr(AffineMap.from_callable(lambda i, j: ())),
            AffineMapAttr(
                AffineMap(
                    2,
                    0,
                    (
                        AffineExpr.constant(0),
                        AffineExpr.dimension(0) * 3 + AffineExpr.dimension(1),
                    ),
                )
            ),
        ),
        (
            linalg.IteratorTypeAttr.parallel(),
            linalg.IteratorTypeAttr.parallel(),
            linalg.IteratorTypeAttr.parallel(),
        ),
    )

    with ImplicitBuilder(op.body) as (a, b):
        c = arith.Muli(a, b).result
        linalg.YieldOp(c)

    a = ShapedArray(TypedPtr.new_int32([1, 2, 3, 4, 5, 6]), [2, 3])
    b = 2
    c = ShapedArray(TypedPtr.new_int32([-1, -1, -1, -1, -1, -1]), [1, 6])

    interpreter.run_op(op, (a, b, c))

    assert c.data == [2, 4, 6, 8, 10, 12]


def test_linalg_generic_reduction():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(LinalgFunctions())
    interpreter.register_implementations(ArithFunctions())

    op = linalg.Generic(
        (
            TestSSAValue(MemRefType(i32, [3])),
            TestSSAValue(MemRefType(i32, [3])),
        ),
        (TestSSAValue(MemRefType(i32, [])),),
        Region(Block(arg_types=(i32, i32, i32))),
        (
            AffineMapAttr(AffineMap.identity(1)),
            AffineMapAttr(AffineMap.identity(1)),
            AffineMapAttr(AffineMap.from_callable(lambda d0: ())),
        ),
        (linalg.IteratorTypeAttr.reduction(),),
    )

    with ImplicitBuilder(op.body) as (lhs, rhs, acc):
        sum = arith.Muli(lhs, rhs).result
        new_acc = arith.Addi(sum, acc).result
        linalg.YieldOp(new_acc)

    a = ShapedArray(TypedPtr.new_int32([1, 2, 3]), [3])
    b = ShapedArray(TypedPtr.new_int32([4, 5, 6]), [3])
    c = ShapedArray(TypedPtr.new_int32([0]), [])

    interpreter.run_op(op, (a, b, c))

    assert c.data == [32]


def test_linalg_add():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(LinalgFunctions())
    op = linalg.AddOp(
        (
            TestSSAValue(TensorType(f32, [2, 2])),
            TestSSAValue(TensorType(f32, [2, 2])),
        ),
        (TestSSAValue(TensorType(f32, [2, 2])),),
        (TensorType(f32, [2, 2]),),
    )

    a = ShapedArray(TypedPtr.new_float32([1.0, 2.0, 3.0, 4.0]), [2, 2])
    b = ShapedArray(TypedPtr.new_float32([6.0, 4.0, 9.0, 5.0]), [2, 2])
    c = ShapedArray(TypedPtr.new_float32([0.0, 0.0, 0.0, 0.0]), [2, 2])

    (c,) = interpreter.run_op(op, (a, b, c))

    assert c == ShapedArray(TypedPtr.new_float32([7, 6, 12, 9]), [2, 2])


def test_fill_op():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(ArithFunctions())
    interpreter.register_implementations(LinalgFunctions())
    constant = arith.Constant(FloatAttr(1.0, f32))
    op = linalg.FillOp(
        (TestSSAValue(constant.result.type),),
        (TestSSAValue(TensorType(f32, [2, 3])),),
        (TensorType(f32, [2, 3]),),
    )
    a = ShapedArray(TypedPtr.new_float32([1.0]), [1])
    b = ShapedArray(TypedPtr.new_float32([0.0] * 6), [2, 3])
    (b,) = interpreter.run_op(op, (a, b))
    assert b == ShapedArray(
        TypedPtr.new_float32([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), [2, 3]
    )


def test_linalg_mul():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(LinalgFunctions())
    op = linalg.MulOp(
        (
            TestSSAValue(TensorType(f32, [2, 2])),
            TestSSAValue(TensorType(f32, [2, 2])),
        ),
        (TestSSAValue(TensorType(f32, [2, 2])),),
        (TensorType(f32, [2, 2]),),
    )

    a = ShapedArray(TypedPtr.new_float32([1.0, 0.0, 8.0, 4.0]), [2, 2])
    b = ShapedArray(TypedPtr.new_float32([3.0, 9.0, 1.0, 6.0]), [2, 2])
    c = ShapedArray(TypedPtr.new_float32([0.0, 0.0, 0.0, 0.0]), [2, 2])

    (c,) = interpreter.run_op(op, (a, b, c))
    assert c == ShapedArray(TypedPtr.new_float32([3.0, 0.0, 8.0, 24.0]), [2, 2])


def test_linalg_transpose():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(LinalgFunctions())
    op = linalg.TransposeOp(
        TestSSAValue(TensorType(f32, [3, 2])),
        TestSSAValue(TensorType(f32, [2, 3])),
        DenseArrayBase.from_list(i64, [1, 0]),
        TensorType(f32, [2, 3]),
    )

    a = ShapedArray(TypedPtr.new_float32([3.0, 5.0, 6.0, 7.0, 8.0, 9.0]), [3, 2])
    b = ShapedArray(TypedPtr.new_float32([0.0] * 6), [2, 3])
    (b,) = interpreter.run_op(op, (a, b))
    assert b == ShapedArray(
        TypedPtr.new_float32([3.0, 6.0, 8.0, 5.0, 7.0, 9.0]), [2, 3]
    )


def test_linalg_matmul():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(LinalgFunctions())
    op = linalg.MatmulOp(
        (
            TestSSAValue(TensorType(f32, [3, 2])),
            TestSSAValue(TensorType(f32, [2, 3])),
        ),
        (TestSSAValue(TensorType(f32, [3, 3])),),
        (TensorType(f32, [3, 3]),),
    )

    a = ShapedArray(TypedPtr.new_float32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), [3, 2])
    b = ShapedArray(TypedPtr.new_float32([4.0, 3.0, 5.0, 1.0, 2.0, 8.0]), [2, 3])
    c = ShapedArray(TypedPtr.new_float32([0.0] * 9), [3, 3])

    (c,) = interpreter.run_op(op, (a, b, c))
    assert c == ShapedArray(
        TypedPtr.new_float32([6.0, 7.0, 21.0, 16.0, 17.0, 47.0, 26.0, 27.0, 73.0]),
        [3, 3],
    )


def test_linalg_pooling_nchw_max():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(LinalgFunctions())
    op = linalg.PoolingNchwMaxOp(
        DenseIntOrFPElementsAttr.tensor_from_list([1], i64, [2]),
        DenseIntOrFPElementsAttr.tensor_from_list([1], i64, [2]),
        (
            TestSSAValue(TensorType(f32, [1, 1, 4, 4])),
            TestSSAValue(TensorType(f32, [2, 2])),
        ),
        (TestSSAValue(TensorType(f32, [1, 1, 3, 3])),),
        (TensorType(f32, [1, 1, 3, 3]),),
    )
    a = ShapedArray(TypedPtr.new_float32(list(range(1, 17))), [1, 1, 4, 4])
    b = ShapedArray(
        TypedPtr.new_float32(
            [
                1.0,
            ]
            * 4
        ),
        [2, 2],
    )
    c = ShapedArray(TypedPtr.new_float32([0.0] * 9), [1, 1, 3, 3])
    (b,) = interpreter.run_op(op, (a, b, c))
    assert b == ShapedArray(
        TypedPtr.new_float32([6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]),
        [1, 1, 3, 3],
    )


def test_linalg_pooling_nchw_max_strides_two():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(LinalgFunctions())
    op = linalg.PoolingNchwMaxOp(
        DenseIntOrFPElementsAttr.tensor_from_list([1], i64, [2]),
        DenseIntOrFPElementsAttr.tensor_from_list([2], i64, [2]),
        (
            TestSSAValue(TensorType(f32, [1, 1, 4, 4])),
            TestSSAValue(TensorType(f32, [2, 2])),
        ),
        (TestSSAValue(TensorType(f32, [1, 1, 2, 2])),),
        (TensorType(f32, [1, 1, 2, 2]),),
    )
    a = ShapedArray(
        TypedPtr.new_float32([1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4]),
        [1, 1, 4, 4],
    )
    b = ShapedArray(
        TypedPtr.new_float32(
            [
                1.0,
            ]
            * 4
        ),
        [2, 2],
    )
    c = ShapedArray(TypedPtr.new_float32([0.0] * 4), [1, 1, 2, 2])
    (b,) = interpreter.run_op(op, (a, b, c))
    assert b == ShapedArray(TypedPtr.new_float32([6.0, 8.0, 3.0, 4.0]), [1, 1, 2, 2])


def test_linalg_conv_2d_nchw_fchw():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(LinalgFunctions())
    op = linalg.Conv2DNchwFchwOp(
        DenseIntOrFPElementsAttr.tensor_from_list([1], i64, [2]),
        DenseIntOrFPElementsAttr.tensor_from_list([1], i64, [2]),
        (
            TestSSAValue(TensorType(f32, [1, 1, 5, 5])),
            TestSSAValue(TensorType(f32, [1, 1, 3, 3])),
        ),
        (TestSSAValue(TensorType(f32, [1, 1, 3, 3])),),
        (TensorType(f32, [1, 1, 3, 3]),),
    )
    a = ShapedArray(TypedPtr.new_float32(list(range(25))), [1, 1, 5, 5])
    b = ShapedArray(
        TypedPtr.new_float32(
            [
                1,
            ]
            * 9
        ),
        [1, 1, 3, 3],
    )
    c = ShapedArray(
        TypedPtr.new_float32(
            [
                0.0,
            ]
            * 9
        ),
        [1, 1, 3, 3],
    )
    (c,) = interpreter.run_op(op, (a, b, c))
    assert c == ShapedArray(
        TypedPtr.new_float32([54, 63, 72, 99, 108, 117, 144, 153, 162]),
        [1, 1, 3, 3],
    )
