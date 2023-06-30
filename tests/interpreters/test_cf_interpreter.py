from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import arith, cf, func
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.cf import CfFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.ir.core import Block, Region


def triangle_fn(n: int) -> int:
    result = 0
    i = 0
    while i <= n:
        result += i
        i += 1
    return result


@ModuleOp
@Builder.implicit_region
def triangle_op():
    prologue = Block(arg_types=(i32,))
    loop_iter = Block(arg_types=(i32, i32))
    loop_body = Block(arg_types=(i32, i32))
    epilogue = Block(arg_types=(i32,))

    with ImplicitBuilder(prologue):
        result = arith.Constant.from_int_and_width(0, 32).result
        i = arith.Constant.from_int_and_width(0, 32).result
        cf.Branch.get(loop_iter, result, i)

    with ImplicitBuilder(loop_iter):
        (n,) = prologue.args
        result, i = loop_iter.args
        cond = arith.Cmpi.get(i, n, "sle")
        cf.ConditionalBranch.get(cond, loop_body, (result, i), epilogue, (result,))

    with ImplicitBuilder(loop_body):
        result, i = loop_iter.args
        new_result = arith.Addi(result, i)
        one = arith.Constant.from_int_and_width(1, 32)
        new_i = arith.Addi(i, one)
        cf.Branch.get(loop_iter, new_result, new_i)

    with ImplicitBuilder(epilogue):
        (result,) = epilogue.args
        func.Return(result)

    func.FuncOp(
        "triangle", ((i32,), (i32,)), Region([prologue, loop_iter, loop_body, epilogue])
    )


def triangle_interp(n: int) -> int:
    interpreter = Interpreter(triangle_op)
    interpreter.register_implementations(CfFunctions())
    interpreter.register_implementations(FuncFunctions())
    interpreter.register_implementations(ArithFunctions())
    (result,) = interpreter.call_op("triangle", (n,))
    return result


def test_triangle():
    assert triangle_fn(0) == triangle_interp(0)
    assert triangle_fn(1) == 1
    assert triangle_fn(2) == 3
    assert triangle_fn(3) == 6
    assert triangle_fn(4) == 10
