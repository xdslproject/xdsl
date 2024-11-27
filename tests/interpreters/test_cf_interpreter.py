import pytest

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import arith, cf, func
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.cf import CfFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.ir import Block, Region


def sum_to_fn(n: int) -> int:
    result = 0
    i = 0
    while i <= n:
        result += i
        i += 1
    return result


@ModuleOp
@Builder.implicit_region
def sum_to_op():
    prologue = Block(arg_types=(i32,))
    loop_iter = Block(arg_types=(i32, i32))
    loop_body = Block(arg_types=(i32, i32))
    epilogue = Block(arg_types=(i32,))

    with ImplicitBuilder(prologue):
        result = arith.ConstantOp.from_int_and_width(0, 32).result
        i = arith.ConstantOp.from_int_and_width(0, 32).result
        cf.BranchOp(loop_iter, result, i)

    with ImplicitBuilder(loop_iter):
        (n,) = prologue.args
        result, i = loop_iter.args
        cond = arith.CmpiOp(i, n, "sle")
        cf.ConditionalBranchOp(cond, loop_body, (result, i), epilogue, (result,))

    with ImplicitBuilder(loop_body):
        result, i = loop_iter.args
        new_result = arith.AddiOp(result, i)
        one = arith.ConstantOp.from_int_and_width(1, 32)
        new_i = arith.AddiOp(i, one)
        cf.BranchOp(loop_iter, new_result, new_i)

    with ImplicitBuilder(epilogue):
        (result,) = epilogue.args
        func.ReturnOp(result)

    func.FuncOp(
        "sum_to", ((i32,), (i32,)), Region([prologue, loop_iter, loop_body, epilogue])
    )


def sum_to_interp(n: int) -> int:
    interpreter = Interpreter(sum_to_op)
    interpreter.register_implementations(CfFunctions())
    interpreter.register_implementations(FuncFunctions())
    interpreter.register_implementations(ArithFunctions())
    (result,) = interpreter.call_op("sum_to", (n,))
    return result


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_sum_to(n: int):
    assert sum_to_fn(n) == sum_to_interp(n)
