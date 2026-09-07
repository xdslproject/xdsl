"""Constant folding must retain IEEE zero signs and NaN behavior."""

import math

import pytest

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.ir import Block, Region
from xdsl.transforms.canonicalize import CanonicalizePass


@pytest.mark.parametrize("float_type", [builtin.f16, builtin.f32, builtin.f64])
@pytest.mark.parametrize(
    "left,right,expected",
    [
        (1.0, 0.0, math.inf),
        (1.0, -0.0, -math.inf),
        (-1.0, 0.0, -math.inf),
        (-1.0, -0.0, math.inf),
        (math.inf, 0.0, math.inf),
        (math.inf, -0.0, -math.inf),
        (-math.inf, 0.0, -math.inf),
        (-math.inf, -0.0, math.inf),
        (math.nan, 0.0, math.nan),
        (math.nan, -0.0, math.nan),
        (0.0, 0.0, math.nan),
        (0.0, -0.0, math.nan),
        (-0.0, 0.0, math.nan),
        (-0.0, -0.0, math.nan),
        (0.0, -1.0, -0.0),
        (-0.0, -1.0, 0.0),
        (6.0, 2.0, 3.0),
        (6.0, -2.0, -3.0),
    ],
)
def test_divf_constant_folding_special_values(
    float_type: builtin.Float16Type | builtin.Float32Type | builtin.Float64Type,
    left: float,
    right: float,
    expected: float,
):
    a = arith.ConstantOp(builtin.FloatAttr(left, float_type))
    b = arith.ConstantOp(builtin.FloatAttr(right, float_type))
    one = arith.ConstantOp(builtin.FloatAttr(1.0, float_type))
    quotient = arith.DivfOp(a, b)
    comparison = arith.CmpfOp(quotient, one, "ogt")
    block = Block(
        [a, b, one, quotient, comparison, func.ReturnOp(quotient, comparison)]
    )
    module = builtin.ModuleOp(
        [func.FuncOp("test", ([], [float_type, builtin.i1]), Region(block))]
    )
    module.verify()
    CanonicalizePass().apply(Context(), module)
    module.verify()
    assert not any(isinstance(op, arith.DivfOp) for op in module.walk())

    # Divf has no interpreter implementation; interpret only the folded program.
    interpreter = Interpreter(module)
    interpreter.register_implementations(ArithFunctions())
    interpreter.register_implementations(FuncFunctions())
    value, greater_than_one = interpreter.call_op("test", ())
    if math.isnan(expected):
        assert math.isnan(value)
    else:
        assert value == expected
        assert math.copysign(1.0, value) == math.copysign(1.0, expected)
    assert bool(greater_than_one) == (expected > 1.0)
