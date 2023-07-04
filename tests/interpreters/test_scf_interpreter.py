import pytest

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import arith, scf, func
from xdsl.dialects.builtin import ModuleOp, IndexType
from xdsl.interpreter import Interpreter

from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.scf import ScfFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.ir.core import BlockArgument

index = IndexType()


def sum_to_for_fn(n: int) -> int:
    """
    Python implementation of sum_to_for_op
    """
    result = 0
    for i in range(0, n, 1):
        result += i
    return result


@ModuleOp
@Builder.implicit_region
def sum_to_for_op():
    with ImplicitBuilder(func.FuncOp("sum_to", ((index,), (index,))).body) as (ub,):
        lb = arith.Constant.from_int_and_width(0, index)
        step = arith.Constant.from_int_and_width(1, index)
        initial = arith.Constant.from_int_and_width(0, index)

        @Builder.implicit_region((index, index))
        def for_loop_region(args: tuple[BlockArgument, ...]):
            (i, acc) = args
            res = arith.Addi(i, acc)
            scf.Yield.get(res)

        result = scf.For.get(lb, ub, step, (initial,), for_loop_region)
        func.Return(result)


def sum_to_interp(module_op: ModuleOp, n: int) -> int:
    module_op.verify()
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(ScfFunctions())
    interpreter.register_implementations(FuncFunctions())
    interpreter.register_implementations(ArithFunctions())
    (result,) = interpreter.call_op("sum_to", (n,))
    return result


@pytest.mark.parametrize("n,res", ((0, 0), (1, 0), (2, 1), (3, 3), (4, 6), (5, 10)))
def test_sum_to(n: int, res: int):
    assert res == sum_to_interp(sum_to_for_op, n)
