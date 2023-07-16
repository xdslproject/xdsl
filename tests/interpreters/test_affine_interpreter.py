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
            ((), (memref.MemRefType.from_element_type_and_shape(index, (2, 3)),)),
        ).body
    ):
        alloc_op = memref.Alloc.get(index, None, (2, 3))

        @Builder.implicit_region((index,))
        def rows_region(args: tuple[Any, ...]):
            (row,) = args

            @Builder.implicit_region((index,))
            def cols_region(args: tuple[Any, ...]):
                (col,) = args
                sum_op = arith.Addi(row, col)
                affine.Store(sum_op.result, alloc_op.memref, (row, col))
                affine.Yield.get()

            affine.For.from_region((), (), 0, 3, cols_region)

            affine.Yield.get()

        affine.For.from_region((), (), 0, 2, rows_region)
        func.Return(alloc_op.memref)


def test_functions():
    module_op.verify()

    interpreter = Interpreter(module_op)
    interpreter.register_implementations(ArithFunctions())
    interpreter.register_implementations(MemrefFunctions())
    interpreter.register_implementations(AffineFunctions())
    interpreter.register_implementations(FuncFunctions())

    res = interpreter.call_op("my_func", ())

    assert res == (ShapedArray(data=[0, 1, 2, 1, 2, 3], shape=[2, 3]),)
