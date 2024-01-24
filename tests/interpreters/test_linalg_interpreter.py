import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import AffineMapAttr, ModuleOp, StringAttr, i32
from xdsl.dialects.memref import MemRefType
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.linalg import LinalgFunctions
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

    a = ShapedArray([1, 2, 3, 4, 5, 6], [2, 3])
    b = ShapedArray([1, 4, 2, 5, 3, 6], [3, 2])
    c = ShapedArray([-1, -1, -1, -1, -1, -1], [1, 6])

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

    a = ShapedArray([1, 2, 3, 4, 5, 6], [2, 3])
    b = 2
    c = ShapedArray([-1, -1, -1, -1, -1, -1], [1, 6])

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

    a = ShapedArray([1, 2, 3], [3])
    b = ShapedArray([4, 5, 6], [3])
    c = ShapedArray([0], [])

    interpreter.run_op(op, (a, b, c))

    assert c.data == [32]
