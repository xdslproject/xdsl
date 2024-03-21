from typing import cast

import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import (
    AffineMapAttr,
    DenseArrayBase,
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
from xdsl.ir import Attribute, Block, Region
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

    a = ShapedArray([1, 2, 3, 4], [2, 2])
    b = ShapedArray([6, 4, 9, 5], [2, 2])

    c = interpreter.run_op(op, (a, b))
    assert c[0] == ShapedArray([7, 6, 12, 9], [2, 2])


def test_fill_op():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(ArithFunctions())
    interpreter.register_implementations(LinalgFunctions())
    constant = arith.Constant(FloatAttr(0.0, f32))
    constant = cast(Attribute, constant)
    op = linalg.FillOp(
        (TestSSAValue(constant),),
        (TestSSAValue(TensorType(f32, [2, 3])),),
        (TensorType(f32, [2, 3]),),
    )
    a = ShapedArray([0], [1])
    c = interpreter.run_op(op, (a,))
    assert c[0] == ShapedArray([0, 0, 0, 0, 0, 0], [2, 3])


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

    a = ShapedArray([1, 0, 8, 4], [2, 2])
    b = ShapedArray([3, 9, 1, 6], [2, 2])

    c = interpreter.run_op(op, (a, b))
    assert c[0] == ShapedArray([3, 0, 8, 24], [2, 2])


def test_linalg_transpose():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(LinalgFunctions())
    op = linalg.TransposeOp(
        TestSSAValue(TensorType(f32, [3, 2])),
        TestSSAValue(TensorType(f32, [2, 3])),
        DenseArrayBase.from_list(i64, [1, 0]),
        TensorType(f32, [2, 2]),
    )

    a = ShapedArray([3, 5, 6, 7, 8, 9], [3, 2])
    c = interpreter.run_op(op, (a,))
    assert c[0] == ShapedArray([3, 6, 8, 5, 7, 9], [2, 3])


def test_linalg_matmul():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(LinalgFunctions())
    op = linalg.MatmulOp(
        (TestSSAValue(TensorType(f32, [3, 2])),),
        (TestSSAValue(TensorType(f32, [2, 3])),),
        (TensorType(f32, [3, 3]),),
    )

    a = ShapedArray([1, 2, 3, 4, 5, 6], [3, 2])
    b = ShapedArray([4, 3, 5, 1, 2, 8], [2, 3])

    c = interpreter.run_op(op, (a, b))
    assert c[0] == ShapedArray([6, 7, 21, 16, 17, 47, 26, 27, 73], [3, 3])
