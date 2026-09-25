from unittest.mock import Mock, call

import pytest

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import arith, func, scf
from xdsl.dialects.builtin import IndexType, ModuleOp, i1, i32
from xdsl.interpreter import Interpreter, OpCounter, OpImplResult, ReturnedValues
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.scf import ScfFunctions
from xdsl.ir import Block, BlockArgument, Region
from xdsl.utils.test_value import create_ssa_value

index = IndexType()


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


@pytest.mark.parametrize(
    "ub,body_args,expected_result",
    [
        (1, (), 10),
        (4, ((1, 10),), 20),
        (8, ((1, 10), (4, 20), (7, 30)), 40),
    ],
)
def test_for(ub: int, body_args: tuple[tuple[int, int], ...], expected_result: int):
    lb, upper_bound, step, initial = (create_ssa_value(index) for _ in range(4))
    body = Region(Block(arg_types=(index, index)))
    for_op = scf.ForOp(lb, upper_bound, step, (initial,), body)

    interpreter = Mock(spec=Interpreter)
    interpreter.run_ssacfg_region = Mock(side_effect=[(20,), (30,), (40,)])

    assert ScfFunctions().run_for(interpreter, for_op, (1, ub, 3, 10)) == OpImplResult(
        (expected_result,), None
    )
    assert interpreter.run_ssacfg_region.call_args_list == [
        call(body, args, "for_loop") for args in body_args
    ]


@pytest.mark.parametrize("cond_value", [True, False])
def test_if(cond_value: bool):
    true_region = Region(Block())
    false_region = Region(Block())
    if_op = scf.IfOp(create_ssa_value(i1), (i32,), true_region, false_region)

    interpreter = Mock(spec=Interpreter)
    interpreter.run_ssacfg_region = Mock(return_value=(42,))

    assert ScfFunctions().run_if(interpreter, if_op, (cond_value,)) == OpImplResult(
        (42,), None
    )
    expected_region = true_region if cond_value else false_region
    interpreter.run_ssacfg_region.assert_called_once_with(expected_region, ())


def test_tracer():
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


@pytest.mark.parametrize("cond_value", [True, False])
def test_condition_op(cond_value: bool):
    interpreter = Interpreter(ModuleOp([]))
    scf_functions = ScfFunctions()
    interpreter.register_implementations(scf_functions)

    cond = create_ssa_value(i1)
    a = create_ssa_value(i32)
    b = create_ssa_value(i32)

    condition_op = scf.ConditionOp(cond, a, b)

    res = scf_functions.run_condition(interpreter, condition_op, (cond_value, 1, 2))

    assert res.values == ()
    assert res.terminator_value is not None
    assert isinstance(res.terminator_value, ReturnedValues)
    assert res.terminator_value.values == (cond_value, 1, 2)
