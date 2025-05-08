from typing import Any

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import affine, arith, func, memref
from xdsl.dialects.builtin import AffineMapAttr, IndexType, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.affine import AffineFunctions
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.memref import MemRefFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.ir.affine import AffineMap
from xdsl.utils.test_value import create_ssa_value

index = IndexType()


@ModuleOp
@Builder.implicit_region
def module_op():
    with ImplicitBuilder(
        func.FuncOp(
            "my_func",
            (
                (),
                (
                    memref.MemRefType(index, (2, 3)),
                    memref.MemRefType(index, (3, 2)),
                ),
            ),
        ).body
    ):
        alloc_op_0 = memref.AllocOp.get(index, None, (2, 3))
        alloc_op_1 = memref.AllocOp.get(index, None, (3, 2))

        @Builder.implicit_region((index,))
        def init_rows_region(args: tuple[Any, ...]):
            (row,) = args

            @Builder.implicit_region((index,))
            def init_cols_region(args: tuple[Any, ...]):
                (col,) = args
                sum_op = arith.AddiOp(row, col)
                affine.StoreOp(sum_op.result, alloc_op_0.memref, (row, col))
                affine.YieldOp.get()

            affine.ForOp.from_region((), (), (), (), 0, 3, init_cols_region)

            affine.YieldOp.get()

        affine.ForOp.from_region((), (), (), (), 0, 2, init_rows_region)

        @Builder.implicit_region((index,))
        def transpose_rows_region(args: tuple[Any, ...]):
            (row,) = args

            @Builder.implicit_region((index,))
            def transpose_cols_region(args: tuple[Any, ...]):
                (col,) = args
                res = affine.LoadOp(alloc_op_0.memref, (row, col)).result
                affine.StoreOp(res, alloc_op_1.memref, (col, row))
                affine.YieldOp.get()

            affine.ForOp.from_region((), (), (), (), 0, 3, transpose_cols_region)

            affine.YieldOp.get()

        affine.ForOp.from_region((), (), (), (), 0, 2, transpose_rows_region)
        func.ReturnOp(alloc_op_0.memref, alloc_op_1.memref)


def test_functions():
    module_op.verify()

    interpreter = Interpreter(module_op)
    interpreter.register_implementations(ArithFunctions())
    interpreter.register_implementations(MemRefFunctions())
    interpreter.register_implementations(AffineFunctions())
    interpreter.register_implementations(FuncFunctions())

    res = interpreter.call_op("my_func", ())

    assert res == (
        ShapedArray(
            TypedPtr.new_index([0, 1, 2, 1, 2, 3], interpreter.index_bitwidth), [2, 3]
        ),
        ShapedArray(
            TypedPtr.new_index([0, 1, 1, 2, 2, 3], interpreter.index_bitwidth), [3, 2]
        ),
    )


def test_apply():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(AffineFunctions())

    assert interpreter.run_op(
        affine.ApplyOp(
            (create_ssa_value(index), create_ssa_value(index)),
            AffineMapAttr(AffineMap.from_callable(lambda d0, d1: (d0 + d1,))),
        ),
        (1, 2),
    ) == (3,)
