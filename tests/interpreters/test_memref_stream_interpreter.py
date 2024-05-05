from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, linalg, memref_stream
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    IntAttr,
    MemRefType,
    ModuleOp,
    i32,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.memref_stream import MemrefStreamFunctions
from xdsl.interpreters.ptr import TypedPtr
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.utils.test_value import TestSSAValue


def test_memref_stream_generic():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(MemrefStreamFunctions())
    interpreter.register_implementations(ArithFunctions())

    op = memref_stream.GenericOp(
        (
            TestSSAValue(MemRefType(i32, [2, 3])),
            TestSSAValue(MemRefType(i32, [3, 2])),
        ),
        (TestSSAValue(MemRefType(i32, [1, 6])),),
        Region(Block(arg_types=(i32, i32))),
        ArrayAttr(
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
            )
        ),
        ArrayAttr(
            (
                linalg.IteratorTypeAttr.parallel(),
                linalg.IteratorTypeAttr.parallel(),
                linalg.IteratorTypeAttr.parallel(),
            )
        ),
        ArrayAttr((IntAttr(2), IntAttr(3))),
    )

    with ImplicitBuilder(op.body) as (a, b):
        c = arith.Muli(a, b).result
        memref_stream.YieldOp(c)

    a = ShapedArray(TypedPtr.new_float64([1, 2, 3, 4, 5, 6]), [2, 3])
    b = ShapedArray(TypedPtr.new_float64([1, 4, 2, 5, 3, 6]), [3, 2])
    c = ShapedArray(TypedPtr.new_float64([-1, -1, -1, -1, -1, -1]), [1, 6])

    interpreter.run_op(op, (a, b, c))

    assert c.data == [1, 4, 9, 16, 25, 36]


def test_memref_stream_generic_scalar():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(MemrefStreamFunctions())
    interpreter.register_implementations(ArithFunctions())

    op = memref_stream.GenericOp(
        (
            TestSSAValue(MemRefType(i32, [2, 3])),
            TestSSAValue(i32),
        ),
        (TestSSAValue(MemRefType(i32, [1, 6])),),
        Region(Block(arg_types=(i32, i32))),
        ArrayAttr(
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
            )
        ),
        ArrayAttr(
            (
                linalg.IteratorTypeAttr.parallel(),
                linalg.IteratorTypeAttr.parallel(),
                linalg.IteratorTypeAttr.parallel(),
            )
        ),
        ArrayAttr((IntAttr(2), IntAttr(3))),
    )

    with ImplicitBuilder(op.body) as (a, b):
        c = arith.Muli(a, b).result
        memref_stream.YieldOp(c)

    a = ShapedArray(TypedPtr.new_float64([1, 2, 3, 4, 5, 6]), [2, 3])
    b = 2
    c = ShapedArray(TypedPtr.new_float64([-1, -1, -1, -1, -1, -1]), [1, 6])

    interpreter.run_op(op, (a, b, c))

    assert c.data == [2, 4, 6, 8, 10, 12]


def test_memref_stream_generic_reduction():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(MemrefStreamFunctions())
    interpreter.register_implementations(ArithFunctions())

    op = memref_stream.GenericOp(
        (
            TestSSAValue(MemRefType(i32, [3])),
            TestSSAValue(MemRefType(i32, [3])),
        ),
        (TestSSAValue(MemRefType(i32, [])),),
        Region(Block(arg_types=(i32, i32, i32))),
        ArrayAttr(
            (
                AffineMapAttr(AffineMap.identity(1)),
                AffineMapAttr(AffineMap.identity(1)),
                AffineMapAttr(AffineMap.from_callable(lambda d0: ())),
            )
        ),
        ArrayAttr((linalg.IteratorTypeAttr.reduction(),)),
        ArrayAttr((IntAttr(3),)),
    )

    with ImplicitBuilder(op.body) as (lhs, rhs, acc):
        sum = arith.Muli(lhs, rhs).result
        new_acc = arith.Addi(sum, acc).result
        memref_stream.YieldOp(new_acc)

    a = ShapedArray(TypedPtr.new_float64([1, 2, 3]), [3])
    b = ShapedArray(TypedPtr.new_float64([4, 5, 6]), [3])
    c = ShapedArray(TypedPtr.new_float64([0]), [])

    interpreter.run_op(op, (a, b, c))

    assert c.data == [32]
