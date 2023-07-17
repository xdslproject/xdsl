from typing import Any

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import affine, arith, func, memref
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.affine import AffineFunctions
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.memref import MemrefFunctions
from xdsl.interpreters.shaped_array import ShapedArray

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
                    memref.MemRefType.from_element_type_and_shape(index, (2, 3)),
                    memref.MemRefType.from_element_type_and_shape(index, (3, 2)),
                ),
            ),
        ).body
    ):
        alloc_op_0 = memref.Alloc.get(index, None, (2, 3))
        alloc_op_1 = memref.Alloc.get(index, None, (3, 2))

        @Builder.implicit_region((index,))
        def init_rows_region(args: tuple[Any, ...]):
            (row,) = args

            @Builder.implicit_region((index,))
            def init_cols_region(args: tuple[Any, ...]):
                (col,) = args
                sum_op = arith.Addi(row, col)
                affine.Store(sum_op.result, alloc_op_0.memref, (row, col))
                affine.Yield.get()

            affine.For.from_region((), (), 0, 3, init_cols_region)

            affine.Yield.get()

        affine.For.from_region((), (), 0, 2, init_rows_region)

        @Builder.implicit_region((index,))
        def transpose_rows_region(args: tuple[Any, ...]):
            (row,) = args

            @Builder.implicit_region((index,))
            def transpose_cols_region(args: tuple[Any, ...]):
                (col,) = args
                res = affine.Load(alloc_op_0.memref, (row, col)).result
                affine.Store(res, alloc_op_1.memref, (col, row))
                affine.Yield.get()

            affine.For.from_region((), (), 0, 3, transpose_cols_region)

            affine.Yield.get()

        affine.For.from_region((), (), 0, 2, transpose_rows_region)
        func.Return(alloc_op_0.memref, alloc_op_1.memref)


def test_functions():
    module_op.verify()

    interpreter = Interpreter(module_op)
    interpreter.register_implementations(ArithFunctions())
    interpreter.register_implementations(MemrefFunctions())
    interpreter.register_implementations(AffineFunctions())
    interpreter.register_implementations(FuncFunctions())

    res = interpreter.call_op("my_func", ())

    assert res == (
        ShapedArray(data=[0, 1, 2, 1, 2, 3], shape=[2, 3]),
        ShapedArray(data=[0, 1, 1, 2, 2, 3], shape=[3, 2]),
    )
