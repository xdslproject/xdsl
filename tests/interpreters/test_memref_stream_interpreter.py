import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, memref_stream
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    f32,
    i32,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.memref_stream import MemRefStreamFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.utils.test_value import create_ssa_value

indextype = IndexType()


def index(value: int) -> IntegerAttr[IndexType]:
    return IntegerAttr(value, indextype)


def test_memref_stream_generic():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(MemRefStreamFunctions())
    interpreter.register_implementations(ArithFunctions())

    op = memref_stream.GenericOp(
        (
            create_ssa_value(MemRefType(i32, [2, 3])),
            create_ssa_value(MemRefType(i32, [3, 2])),
        ),
        (create_ssa_value(MemRefType(i32, [1, 6])),),
        (),
        Region(Block(arg_types=(i32, i32, i32))),
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

    with ImplicitBuilder(op.body) as (a, b, _c_init):
        c = arith.MuliOp(a, b).result
        memref_stream.YieldOp(c)

    op.verify()

    a = ShapedArray(TypedPtr.new_float64([1, 2, 3, 4, 5, 6]), [2, 3])
    b = ShapedArray(TypedPtr.new_float64([1, 4, 2, 5, 3, 6]), [3, 2])
    c = ShapedArray(TypedPtr.new_float64([-1, -1, -1, -1, -1, -1]), [1, 6])

    interpreter.run_op(op, (a, b, c))

    assert c.data == [1, 4, 9, 16, 25, 36]


def test_memref_stream_generic_scalar():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(MemRefStreamFunctions())
    interpreter.register_implementations(ArithFunctions())

    op = memref_stream.GenericOp(
        (
            create_ssa_value(MemRefType(i32, [2, 3])),
            create_ssa_value(i32),
        ),
        (create_ssa_value(MemRefType(i32, [1, 6])),),
        (),
        Region(Block(arg_types=(i32, i32, i32))),
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

    with ImplicitBuilder(op.body) as (a, b, _c_init):
        c = arith.MuliOp(a, b).result
        memref_stream.YieldOp(c)

    op.verify()

    a = ShapedArray(TypedPtr.new_float64([1, 2, 3, 4, 5, 6]), [2, 3])
    b = 2
    c = ShapedArray(TypedPtr.new_float64([-1, -1, -1, -1, -1, -1]), [1, 6])

    interpreter.run_op(op, (a, b, c))

    assert c.data == [2, 4, 6, 8, 10, 12]


def test_memref_stream_generic_reduction():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(MemRefStreamFunctions())
    interpreter.register_implementations(ArithFunctions())

    op = memref_stream.GenericOp(
        (
            create_ssa_value(MemRefType(i32, [3])),
            create_ssa_value(MemRefType(i32, [3])),
        ),
        (create_ssa_value(MemRefType(i32, [])),),
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
        sum = arith.MuliOp(lhs, rhs).result
        new_acc = arith.AddiOp(sum, acc).result
        memref_stream.YieldOp(new_acc)

    op.verify()

    a = ShapedArray(TypedPtr.new_float64([1, 2, 3]), [3])
    b = ShapedArray(TypedPtr.new_float64([4, 5, 6]), [3])
    c = ShapedArray(TypedPtr.new_float64([0]), [])

    interpreter.run_op(op, (a, b, c))

    assert c.data == [32]


def test_memref_stream_generic_imperfect_nesting():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(MemRefStreamFunctions())
    interpreter.register_implementations(ArithFunctions())

    op = memref_stream.GenericOp(
        (
            create_ssa_value(MemRefType(f32, [3, 2])),
            create_ssa_value(MemRefType(f32, [2, 3])),
        ),
        (create_ssa_value(MemRefType(f32, [3, 3])),),
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
        sum = arith.MulfOp(lhs, rhs).result
        new_acc = arith.AddfOp(sum, acc).result
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
    interpreter.register_implementations(MemRefStreamFunctions())
    interpreter.register_implementations(ArithFunctions())

    op = memref_stream.GenericOp(
        (
            create_ssa_value(MemRefType(f32, [3, 2])),
            create_ssa_value(MemRefType(f32, [2, 3])),
        ),
        (create_ssa_value(MemRefType(f32, [3, 3])),),
        (create_ssa_value(f32),),
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
        sum = arith.MulfOp(lhs, rhs).result
        new_acc = arith.AddfOp(sum, acc).result
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


def test_memref_stream_interleaved_reduction_with_initial_value():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(MemRefStreamFunctions())
    interpreter.register_implementations(ArithFunctions())

    op = memref_stream.GenericOp(
        (
            create_ssa_value(MemRefType(f32, [3, 5])),
            create_ssa_value(MemRefType(f32, [5, 8])),
        ),
        (create_ssa_value(MemRefType(f32, [3, 8])),),
        (create_ssa_value(f32),),
        Region(
            Block(
                arg_types=(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
            )
        ),
        ArrayAttr(
            (
                AffineMapAttr(AffineMap.from_callable(lambda n, m, k, j: (n, k))),
                AffineMapAttr(
                    AffineMap.from_callable(lambda n, m, k, j: (k, m * 4 + j))
                ),
                AffineMapAttr(AffineMap.from_callable(lambda n, m, j: (n, m * 4 + j))),
            )
        ),
        ArrayAttr(
            (
                memref_stream.IteratorTypeAttr.parallel(),
                memref_stream.IteratorTypeAttr.parallel(),
                memref_stream.IteratorTypeAttr.reduction(),
                memref_stream.IteratorTypeAttr.interleaved(),
            )
        ),
        ArrayAttr((index(3), index(2), index(5), index(4))),
        ArrayAttr((IntAttr(0),)),
    )

    with ImplicitBuilder(op.body) as (
        lhs0,
        lhs1,
        lhs2,
        lhs3,
        rhs0,
        rhs1,
        rhs2,
        rhs3,
        acc0,
        acc1,
        acc2,
        acc3,
    ):
        sum0 = arith.MulfOp(lhs0, rhs0).result
        sum1 = arith.MulfOp(lhs1, rhs1).result
        sum2 = arith.MulfOp(lhs2, rhs2).result
        sum3 = arith.MulfOp(lhs3, rhs3).result
        new_acc0 = arith.AddfOp(sum0, acc0).result
        new_acc1 = arith.AddfOp(sum1, acc1).result
        new_acc2 = arith.AddfOp(sum2, acc2).result
        new_acc3 = arith.AddfOp(sum3, acc3).result
        memref_stream.YieldOp(new_acc0, new_acc1, new_acc2, new_acc3)

    op.verify()

    a = ShapedArray(TypedPtr.new_float32([float(i) for i in range(3 * 5)]), [3, 5])
    b = ShapedArray(
        TypedPtr.new_float32([float(i) / 100 for i in range(5 * 8)]), [5, 8]
    )
    c = ShapedArray(TypedPtr.new_float32([-1000.0] * (3 * 5)), [3, 5])

    with pytest.raises(
        NotImplementedError,
        match="Interpreter for interleaved operations not yet implemented",
    ):
        interpreter.run_op(op, (a, b, c, 0.5))
    # assert c == ShapedArray(
    #     TypedPtr.new_float32([ 2.9 ,  3.  ,  3.1 ,  3.2 ,  3.3 ,  3.4 ,  3.5 ,  3.6 ,  6.9 ,
    #     7.25,  7.6 ,  7.95,  8.3 ,  8.65,  9.  ,  9.35, 10.9 , 11.5 ,
    #    12.1 , 12.7 , 13.3 , 13.9 , 14.5 , 15.1 ]),
    #     [3, 3],
    # )
