import operator
from collections.abc import Callable

import pytest

from xdsl.dialects import arith, test
from xdsl.dialects.arith import Addf, Addi, Cmpi, Constant, Mulf, Muli, Subf, Subi
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
    constant = Constant.from_int_and_width(value, value_type)

    ret = interpreter.run_op(constant, ())

    assert len(ret) == 1
    assert ret[0] == value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_subi(lhs_value: int, rhs_value: int):
    subi = Subi(lhs_op, rhs_op)

    ret = interpreter.run_op(subi, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value - rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_addi(lhs_value: int, rhs_value: int):
    addi = Addi(lhs_op, rhs_op)

    ret = interpreter.run_op(addi, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value + rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_muli(lhs_value: int, rhs_value: int):
    muli = Muli(lhs_op, rhs_op)

    ret = interpreter.run_op(muli, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value * rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_subf(lhs_value: int, rhs_value: int):
    subf = Subf(lhs_op, rhs_op)

    ret = interpreter.run_op(subf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value - rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_addf(lhs_value: int, rhs_value: int):
    addf = Addf(lhs_op, rhs_op)

    ret = interpreter.run_op(addf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value + rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_mulf(lhs_value: int, rhs_value: int):
    mulf = Mulf(lhs_op, rhs_op)

    ret = interpreter.run_op(mulf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value * rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_minf(lhs_value: float, rhs_value: float):
    minf = arith.Minimumf(lhs_op, rhs_op)

    ret = interpreter.run_op(minf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == min(lhs_value, rhs_value)


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_maxf(lhs_value: int, rhs_value: int):
    maxf = arith.Maximumf(lhs_op, rhs_op)

    ret = interpreter.run_op(maxf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == max(lhs_value, rhs_value)


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
    cmpi = Cmpi(lhs_op, rhs_op, arg)

    ret = interpreter.run_op(cmpi, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == fn(lhs_value, rhs_value)
