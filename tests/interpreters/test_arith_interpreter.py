import operator
from collections.abc import Callable
from math import copysign, isnan

import pytest

from xdsl.dialects import arith, builtin, test
from xdsl.dialects.arith import (
    AddfOp,
    AddiOp,
    AndIOp,
    CmpiOp,
    CmpfOp,
    ConstantOp,
    DivSIOp,
    FloorDivSIOp,
    IndexCastOp,
    MulfOp,
    MuliOp,
    OrIOp,
    RemSIOp,
    ShLIOp,
    ShRSIOp,
    SubfOp,
    SubiOp,
    XOrIOp,
)
from xdsl.dialects.builtin import IndexType, IntegerType, ModuleOp, Signedness, i8, i32
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
def test_andi(lhs_value: int, rhs_value: int):
    andi = AndIOp(lhs_op, rhs_op)

    ret = interpreter.run_op(andi, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value & rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_ori(lhs_value: int, rhs_value: int):
    ori = OrIOp(lhs_op, rhs_op)

    ret = interpreter.run_op(ori, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value | rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [1, 0, -1, 127])
def test_xori(lhs_value: int, rhs_value: int):
    xori = XOrIOp(lhs_op, rhs_op)

    ret = interpreter.run_op(xori, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value ^ rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1])
@pytest.mark.parametrize("rhs_value", [1, 0, -1])
def test_xori_i1(lhs_value: int, rhs_value: int):
    lhs_op = test.TestOp(result_types=[builtin.i1])
    rhs_op = test.TestOp(result_types=[builtin.i1])
    xori = XOrIOp(lhs_op, rhs_op)

    ret = interpreter.run_op(xori, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == -(abs(lhs_value) ^ abs(rhs_value))


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

o = lambda x, y : (not isnan(x) and not isnan(y))
u = lambda x, y : (isnan(x) or isnan(y))

@pytest.mark.parametrize("lhs_value", [1.5, 0.5, -1.5, 127.5, float("nan")])
@pytest.mark.parametrize("rhs_value", [1.5, 0.5, -1.5, 127.5, float("nan")])
@pytest.mark.parametrize(
    "pred",
     list(
             {"false": lambda x, y: False,  # "false"
             "oeq": lambda x, y: (x == y) and o(x, y),  # "oeq"
             "ogt": lambda x, y: (x > y) and o(x, y),   # "ogt"
             "oge": lambda x, y: (x >= y) and o(x, y),  # "oge"
             "olt": lambda x, y: (x < y) and o(x, y),   # "olt"
             "ole": lambda x, y: (x <= y) and o(x, y),   # "ole"
             "one": lambda x, y: (x != y) and o(x, y),   # "one
             "ord": lambda x, y: o(x,y), # "ord"
             "ueq": lambda x, y: (x == y) or u(x,y), # "ueq"
             "ugt": lambda x, y: (x > y) or u(x,y), # "ugt"
             "uge": lambda x, y: (x >= y) or u(x,y), # "uge"
             "ult": lambda x, y: (x < y) or u(x,y), # "ult"
             "ule": lambda x, y: (x <= y) or u(x,y), # "ule"
             "une": lambda x, y: (x != y) or u(x,y), # "une"
             "uno": lambda x, y: u(x,y), # "uno"
             "true": lambda x, y: True # "true"
       }.items()))
def test_cmpf(
    lhs_value: int, rhs_value: int, pred: tuple[str, Callable[[int, int], int]]
):
    arg, fn = pred
    cmpf = CmpfOp(lhs_op, rhs_op, arg)

    ret = interpreter.run_op(cmpf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == fn(lhs_value, rhs_value)


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [0, 1, 2, 8])
def test_shlsi(lhs_value: int, rhs_value: int):
    shlsi = ShLIOp(lhs_op, rhs_op)

    ret = interpreter.run_op(shlsi, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value << rhs_value


@pytest.mark.parametrize("lhs_value", [1, 0, -1, 127])
@pytest.mark.parametrize("rhs_value", [0, 1, 2, 8])
def test_shrsi(lhs_value: int, rhs_value: int):
    shrsi = ShRSIOp(lhs_op, rhs_op)

    ret = interpreter.run_op(shrsi, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value >> rhs_value


@pytest.mark.parametrize(
    "lhs_value,rhs_value,result", [(-3, -2, 1), (-3, 2, -1), (3, -2, -1), (3, 2, 1)]
)
def test_divsi(lhs_value: int, rhs_value: int, result: int):
    divsi = DivSIOp(lhs_op, rhs_op)

    ret = interpreter.run_op(divsi, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == result


@pytest.mark.parametrize(
    "lhs_value,rhs_value,result", [(-3, -2, -1), (-3, 2, -1), (3, -2, 1), (3, 2, 1)]
)
def test_remsi(lhs_value: int, rhs_value: int, result: int):
    remsi = RemSIOp(lhs_op, rhs_op)

    ret = interpreter.run_op(remsi, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == result


@pytest.mark.parametrize(
    "lhs_value,rhs_value,result", [(-3, -2, 1), (-3, 2, -2), (3, -2, -2), (3, 2, 1)]
)
def test_floordivsi(lhs_value: int, rhs_value: int, result: int):
    floordivsi = FloorDivSIOp(lhs_op, rhs_op)

    ret = interpreter.run_op(floordivsi, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == result


@pytest.mark.parametrize("x", [1, 0, -1, 127, 1111])
def test_indexcast_to_i32(x: int):
    x_op = test.TestOp(result_types=[IndexType()])
    indexcast = IndexCastOp(x_op, i32)

    ret = interpreter.run_op(indexcast, (x,))

    assert len(ret) == 1
    assert ret[0] == x


@pytest.mark.parametrize("x", [1, 0, -1, 127, 1111])
def test_indexcast_from_i32(x: int):
    x_op = test.TestOp(result_types=[i32])
    indexcast = IndexCastOp(x_op, IndexType())

    ret = interpreter.run_op(indexcast, (x,))

    assert len(ret) == 1
    assert ret[0] == x


@pytest.mark.parametrize(
    "x,expected", [(1, 1), (0, 0), (-1, -1), (127, 127), (1111, 87)]
)
def test_indexcast_to_i8(x: int, expected: int):
    x_op = test.TestOp(result_types=[IndexType()])
    indexcast = IndexCastOp(x_op, i8)

    ret = interpreter.run_op(indexcast, (x,))

    assert len(ret) == 1
    assert ret[0] == expected


@pytest.mark.parametrize(
    "x,expected", [(1, 1), (0, 0), (-1, -1), (127, 127), (255, -1)]
)
def test_indexcast_from_i8(x: int, expected: int):
    x_op = test.TestOp(result_types=[i8])
    indexcast = IndexCastOp(x_op, IndexType())

    ret = interpreter.run_op(indexcast, (x,))

    assert len(ret) == 1
    assert ret[0] == expected
