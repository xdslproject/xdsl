from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, memref_stream
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    Float32Type,
    IndexType,
    IntAttr,
    IntegerAttr,
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

indextype = IndexType()


def index(value: int) -> IntegerAttr[IndexType]:
    return IntegerAttr(value, indextype)


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
        (),
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
                memref_stream.IteratorTypeAttr.parallel(),
                memref_stream.IteratorTypeAttr.parallel(),
            )
        ),
        ArrayAttr((index(2), index(3))),
        ArrayAttr(()),
    )

    with ImplicitBuilder(op.body) as (a, b):
        c = arith.Muli(a, b).result
        memref_stream.YieldOp(c)

    op.verify()

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
        (),
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
                memref_stream.IteratorTypeAttr.parallel(),
                memref_stream.IteratorTypeAttr.parallel(),
            )
        ),
        ArrayAttr((index(2), index(3))),
        ArrayAttr(()),
    )

    with ImplicitBuilder(op.body) as (a, b):
        c = arith.Muli(a, b).result
        memref_stream.YieldOp(c)

    op.verify()

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
        (),
        Region(Block(arg_types=(i32, i32, i32))),
        ArrayAttr(
            (
                AffineMapAttr(AffineMap.identity(1)),
                AffineMapAttr(AffineMap.identity(1)),
                AffineMapAttr(AffineMap.from_callable(lambda d0: ())),
            )
        ),
        ArrayAttr((memref_stream.IteratorTypeAttr.reduction(),)),
        ArrayAttr((index(3),)),
        ArrayAttr(()),
    )

    with ImplicitBuilder(op.body) as (lhs, rhs, acc):
        sum = arith.Muli(lhs, rhs).result
        new_acc = arith.Addi(sum, acc).result
        memref_stream.YieldOp(new_acc)

    op.verify()

    a = ShapedArray(TypedPtr.new_float64([1, 2, 3]), [3])
    b = ShapedArray(TypedPtr.new_float64([4, 5, 6]), [3])
    c = ShapedArray(TypedPtr.new_float64([0]), [])

    interpreter.run_op(op, (a, b, c))

    assert c.data == [32]


def test_memref_stream_generic_imperfect_nesting():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(MemrefStreamFunctions())
    interpreter.register_implementations(ArithFunctions())

    f32 = Float32Type()

    op = memref_stream.GenericOp(
        (
            TestSSAValue(MemRefType(f32, [3, 2])),
            TestSSAValue(MemRefType(f32, [2, 3])),
        ),
        (TestSSAValue(MemRefType(f32, [3, 3])),),
        (),
        Region(Block(arg_types=(f32, f32, f32))),
        ArrayAttr(
            (
                AffineMapAttr(AffineMap.from_callable(lambda n, m, k: (n, k))),
                AffineMapAttr(AffineMap.from_callable(lambda n, m, k: (k, m))),
                AffineMapAttr(AffineMap.from_callable(lambda n, m: (n, m))),
            )
        ),
        ArrayAttr(
            (
                memref_stream.IteratorTypeAttr.parallel(),
                memref_stream.IteratorTypeAttr.parallel(),
                memref_stream.IteratorTypeAttr.reduction(),
            )
        ),
        ArrayAttr((index(3), index(3), index(2))),
        ArrayAttr(()),
    )

    with ImplicitBuilder(op.body) as (lhs, rhs, acc):
        sum = arith.Mulf(lhs, rhs).result
        new_acc = arith.Addf(sum, acc).result
        memref_stream.YieldOp(new_acc)

    op.verify()

    a = ShapedArray(TypedPtr.new_float32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), [3, 2])
    b = ShapedArray(TypedPtr.new_float32([4.0, 3.0, 5.0, 1.0, 2.0, 8.0]), [2, 3])
    c = ShapedArray(TypedPtr.new_float32([0.0] * 9), [3, 3])

    interpreter.run_op(op, (a, b, c))
    assert c == ShapedArray(
        TypedPtr.new_float32([6.0, 7.0, 21.0, 16.0, 17.0, 47.0, 26.0, 27.0, 73.0]),
        [3, 3],
    )


def test_memref_stream_generic_reduction_with_initial_value():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(MemrefStreamFunctions())
    interpreter.register_implementations(ArithFunctions())

    f32 = Float32Type()

    op = memref_stream.GenericOp(
        (
            TestSSAValue(MemRefType(f32, [3, 2])),
            TestSSAValue(MemRefType(f32, [2, 3])),
        ),
        (TestSSAValue(MemRefType(f32, [3, 3])),),
        (TestSSAValue(f32),),
        Region(Block(arg_types=(f32, f32, f32))),
        ArrayAttr(
            (
                AffineMapAttr(AffineMap.from_callable(lambda n, m, k: (n, k))),
                AffineMapAttr(AffineMap.from_callable(lambda n, m, k: (k, m))),
                AffineMapAttr(AffineMap.from_callable(lambda n, m: (n, m))),
            )
        ),
        ArrayAttr(
            (
                memref_stream.IteratorTypeAttr.parallel(),
                memref_stream.IteratorTypeAttr.parallel(),
                memref_stream.IteratorTypeAttr.reduction(),
            )
        ),
        ArrayAttr((index(3), index(3), index(2))),
        ArrayAttr((IntAttr(0),)),
    )

    with ImplicitBuilder(op.body) as (lhs, rhs, acc):
        sum = arith.Mulf(lhs, rhs).result
        new_acc = arith.Addf(sum, acc).result
        memref_stream.YieldOp(new_acc)

    op.verify()

    a = ShapedArray(TypedPtr.new_float32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), [3, 2])
    b = ShapedArray(TypedPtr.new_float32([4.0, 3.0, 5.0, 1.0, 2.0, 8.0]), [2, 3])
    c = ShapedArray(TypedPtr.new_float32([0.0] * 9), [3, 3])

    interpreter.run_op(op, (a, b, c, 0.5))
    assert c == ShapedArray(
        TypedPtr.new_float32([6.5, 7.5, 21.5, 16.5, 17.5, 47.5, 26.5, 27.5, 73.5]),
        [3, 3],
    )
