import operator
from collections.abc import Callable
from math import copysign, isnan

import pytest

from xdsl.dialects import arith, test
from xdsl.dialects.arith import (
    AddfOp,
    AddiOp,
    CmpiOp,
    ConstantOp,
    MulfOp,
    MuliOp,
    SubfOp,
    SubiOp,
)
from xdsl.dialects.builtin import IndexType, IntegerType, ModuleOp, Signedness
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions

interpreter = Interpreter(ModuleOp([]))
interpreter.register_implementations(ArithFunctions())

lhs_op = test.TestOp(result_types=[IndexType()])
rhs_op = test.TestOp(result_types=[IndexType()])


@pytest.mark.parametrize("value", [1, 0, -1, 127])
@pytest.mark.parametrize(
    "value_type",
    [
        IndexType(),
        IntegerType(8),
        IntegerType(16),
        IntegerType(32),
        IntegerType(32, Signedness.SIGNED),
    ],
)
def test_constant(value: int, value_type: int | IndexType | IntegerType):
    constant = ConstantOp.from_int_and_width(value, value_type)

    ret = interpreter.run_op(constant, ())

    assert len(ret) == 1
    assert ret[0] == value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_subi(lhs_value: int, rhs_value: int):
    subi = SubiOp(lhs_op, rhs_op)

    ret = interpreter.run_op(subi, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value - rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_addi(lhs_value: int, rhs_value: int):
    addi = AddiOp(lhs_op, rhs_op)

    ret = interpreter.run_op(addi, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value + rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_muli(lhs_value: int, rhs_value: int):
    muli = MuliOp(lhs_op, rhs_op)

    ret = interpreter.run_op(muli, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value * rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_subf(lhs_value: int, rhs_value: int):
    subf = SubfOp(lhs_op, rhs_op)

    ret = interpreter.run_op(subf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value - rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_addf(lhs_value: int, rhs_value: int):
    addf = AddfOp(lhs_op, rhs_op)

    ret = interpreter.run_op(addf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value + rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_mulf(lhs_value: int, rhs_value: int):
    mulf = MulfOp(lhs_op, rhs_op)

    ret = interpreter.run_op(mulf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value * rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_minf(lhs_value: float, rhs_value: float):
    minf = arith.MinimumfOp(lhs_op, rhs_op)

    ret = interpreter.run_op(minf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == min(lhs_value, rhs_value)


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_maximumf(lhs_value: int, rhs_value: int):
    maxf = arith.MaximumfOp(lhs_op, rhs_op)

    ret = interpreter.run_op(maxf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == max(lhs_value, rhs_value)


def test_minmax_corner():
    maxf = arith.MaximumfOp(lhs_op, rhs_op)

    assert copysign(1.0, interpreter.run_op(maxf, (0.0, 0.0))[0]) == 1.0
    assert copysign(1.0, interpreter.run_op(maxf, (-0.0, 0.0))[0]) == 1.0
    assert copysign(1.0, interpreter.run_op(maxf, (0.0, -0.0))[0]) == 1.0
    assert copysign(1.0, interpreter.run_op(maxf, (-0.0, -0.0))[0]) == -1.0
    assert isnan(interpreter.run_op(maxf, (float("NaN"), 0.0))[0])
    assert isnan(interpreter.run_op(maxf, (0.0, float("NaN")))[0])

    minf = arith.MinimumfOp(lhs_op, rhs_op)

    assert copysign(1.0, interpreter.run_op(minf, (0.0, 0.0))[0]) == 1.0
    assert copysign(1.0, interpreter.run_op(minf, (-0.0, 0.0))[0]) == -1.0
    assert copysign(1.0, interpreter.run_op(minf, (0.0, -0.0))[0]) == -1.0
    assert copysign(1.0, interpreter.run_op(minf, (-0.0, -0.0))[0]) == -1.0
    assert isnan(interpreter.run_op(minf, (float("NaN"), 0.0))[0])
    assert isnan(interpreter.run_op(minf, (0.0, float("NaN")))[0])


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize(
    "pred",
    [
        ("eq", operator.eq),
        ("ne", operator.ne),
        ("slt", operator.lt),
        ("sle", operator.le),
        ("sgt", operator.gt),
        ("sge", operator.ge),
        ("ult", operator.lt),
        ("ule", operator.le),
        ("ugt", operator.gt),
        ("uge", operator.ge),
    ],
)
def test_cmpi(
    lhs_value: int, rhs_value: int, pred: tuple[str, Callable[[int, int], int]]
):
    arg, fn = pred
    cmpi = CmpiOp(lhs_op, rhs_op, arg)

    ret = interpreter.run_op(cmpi, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == fn(lhs_value, rhs_value)
