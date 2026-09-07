import pytest

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import arith, func, scf
from xdsl.dialects.builtin import IndexType, ModuleOp, i1, i32
from xdsl.interpreter import Interpreter, OpCounter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.scf import ScfFunctions
from xdsl.ir import BlockArgument

index = IndexType()


def scf_interp(module_op: ModuleOp, func_name: str, n: int) -> int:
    module_op.verify()
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(ScfFunctions())
    interpreter.register_implementations(FuncFunctions())
    interpreter.register_implementations(ArithFunctions())
    (result,) = interpreter.call_op(func_name, (n,))
    return result


def test_if():
    @ModuleOp
    @Builder.implicit_region
    def module_op():
        with ImplicitBuilder(func.FuncOp("indicator", ((i1,), (i32,))).body) as (cond,):

            @Builder.implicit_region
            def true_region():
                one = arith.ConstantOp.from_int_and_width(1, 32)
                scf.YieldOp(one)

            @Builder.implicit_region
            def false_region():
                zero = arith.ConstantOp.from_int_and_width(0, 32)
                scf.YieldOp(zero)

            result = scf.IfOp(cond, (i32,), true_region, false_region)

            func.ReturnOp(result)

    assert scf_interp(module_op, "indicator", True) == 1
    assert scf_interp(module_op, "indicator", False) == 0


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
        lb = arith.ConstantOp.from_int_and_width(0, index)
        step = arith.ConstantOp.from_int_and_width(1, index)
        initial = arith.ConstantOp.from_int_and_width(0, index)

        @Builder.implicit_region((index, index))
        def for_loop_region(args: tuple[BlockArgument, ...]):
            (i, acc) = args
            res = arith.AddiOp(i, acc)
            scf.YieldOp(res)

        result = scf.ForOp(lb, ub, step, (initial,), for_loop_region)
        func.ReturnOp(result)


@pytest.mark.parametrize("n,res", [(0, 0), (1, 0), (2, 1), (3, 3), (4, 6), (5, 10)])
def test_for_via_sum_to(n: int, res: int):
    assert res == scf_interp(sum_to_for_op, "sum_to", n)


def test_for_tracer():
    tracer = OpCounter()
    interpreter = Interpreter(sum_to_for_op.clone(), listeners=(tracer,))
    interpreter.register_implementations(ScfFunctions())
    interpreter.register_implementations(FuncFunctions())
    interpreter.register_implementations(ArithFunctions())
    (result,) = interpreter.call_op("sum_to", (5,))

    assert result == 10
    assert dict(tracer.ops) == {
        "arith.constant": 3,
        "scf.for": 1,
        "scf.yield": 5,
        "arith.addi": 5,
        "func.return": 1,
    }


def sum_to_while_fn(ub: int) -> int:
    """
    Python implementation of sum_to_while_op
    """
    acc = 0
    i = 0
    while i != ub:
        acc += i
        i += 1
    return acc


@ModuleOp
@Builder.implicit_region
def sum_to_while_op():
    with ImplicitBuilder(func.FuncOp("sum_to", ((index,), (index,))).body) as (ub,):
        zero = arith.ConstantOp.from_int_and_width(0, index)

        @Builder.implicit_region((index, index))
        def before_region(args: tuple[BlockArgument, ...]):
            (i, acc) = args
            cond = arith.CmpiOp(i, ub, "ne")
            scf.ConditionOp(cond, i, acc)

        @Builder.implicit_region((index, index))
        def after_region(args: tuple[BlockArgument, ...]):
            (i, acc) = args
            one = arith.ConstantOp.from_int_and_width(1, index)
            new_acc = arith.AddiOp(acc, i)
            new_i = arith.AddiOp(i, one)
            scf.YieldOp(new_i, new_acc)

        result = scf.WhileOp((zero, zero), (index, index), before_region, after_region)
        func.ReturnOp(result.res[1])


@pytest.mark.parametrize(
    ("n", "res"), [(0, 0), (1, 0), (2, 1), (3, 3), (4, 6), (5, 10)]
)
def test_while_via_sum_to(n: int, res: int):
    assert res == sum_to_while_fn(ub=n)
    assert res == scf_interp(sum_to_while_op, "sum_to", n)


def test_while_tracer():
    tracer = OpCounter()
    interpreter = Interpreter(sum_to_while_op.clone(), listeners=(tracer,))
    interpreter.register_implementations(ScfFunctions())
    interpreter.register_implementations(FuncFunctions())
    interpreter.register_implementations(ArithFunctions())
    (result,) = interpreter.call_op("sum_to", (5,))

    assert result == 10
    assert dict(tracer.ops) == {
        "arith.constant": 6,
        "scf.while": 1,
        "scf.condition": 6,
        "scf.yield": 5,
        "arith.cmpi": 6,
        "arith.addi": 10,
        "func.return": 1,
    }
