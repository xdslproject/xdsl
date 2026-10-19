import operator
from collections.abc import Callable
from math import copysign, isnan
from typing import Literal

import pytest

from xdsl.dialects import arith, builtin, test
from xdsl.dialects.arith import (
    AddfOp,
    AddiOp,
    AndIOp,
    CmpfOp,
    CmpiOp,
    ConstantOp,
    DivSIOp,
    FloorDivSIOp,
    IndexCastOp,
    MulfOp,
    MuliOp,
    OrIOp,
    RemSIOp,
    SelectOp,
    ShLIOp,
    ShRSIOp,
    SubfOp,
    SubiOp,
    XOrIOp,
)
from xdsl.dialects.builtin import (
    IndexType,
    IntegerType,
    ModuleOp,
    VectorType,
    f64,
    i1,
    i8,
    i32,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.utils.exceptions import InterpretationError

interpreter = Interpreter(ModuleOp([]), index_bitwidth=64)
interpreter.register_implementations(ArithFunctions())

lhs_op = test.TestOp(result_types=[IndexType()])
rhs_op = test.TestOp(result_types=[IndexType()])
lhs_f64_op = test.TestOp(result_types=[f64])
rhs_f64_op = test.TestOp(result_types=[f64])


@pytest.mark.parametrize("value", [1, 0, -1, 127])
@pytest.mark.parametrize(
    "value_type",
    [
        IndexType(),
        IntegerType(8),
        IntegerType(16),
        IntegerType(32),
    ],
)
def test_constant(value: int, value_type: int | IndexType | IntegerType):
    constant = ConstantOp.from_int_and_width(value, value_type)
    constant.verify()

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


@pytest.mark.parametrize("lhs_value", [1.0, 0.0, -1.0, 127.0])
@pytest.mark.parametrize("rhs_value", [1.0, 0.0, -1.0, 127.0])
def test_subf(lhs_value: float, rhs_value: float):
    subf = SubfOp(lhs_f64_op, rhs_f64_op)
    subf.verify()

    ret = interpreter.run_op(subf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value - rhs_value


@pytest.mark.parametrize("lhs_value", [1.0, 0.0, -1.0, 127.0])
@pytest.mark.parametrize("rhs_value", [1.0, 0.0, -1.0, 127.0])
def test_addf(lhs_value: float, rhs_value: float):
    addf = AddfOp(lhs_f64_op, rhs_f64_op)
    addf.verify()

    ret = interpreter.run_op(addf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value + rhs_value


@pytest.mark.parametrize("lhs_value", [1.0, 0.0, -1.0, 127.0])
@pytest.mark.parametrize("rhs_value", [1.0, 0.0, -1.0, 127.0])
def test_mulf(lhs_value: float, rhs_value: float):
    mulf = MulfOp(lhs_f64_op, rhs_f64_op)
    mulf.verify()

    ret = interpreter.run_op(mulf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == lhs_value * rhs_value


@pytest.mark.parametrize("lhs_value", [1.0, 0.0, -1.0, 127.0])
@pytest.mark.parametrize("rhs_value", [1.0, 0.0, -1.0, 127.0])
def test_minf(lhs_value: float, rhs_value: float):
    minf = arith.MinimumfOp(lhs_f64_op, rhs_f64_op)
    minf.verify()

    ret = interpreter.run_op(minf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == min(lhs_value, rhs_value)


@pytest.mark.parametrize("lhs_value", [1.0, 0.0, -1.0, 127.0])
@pytest.mark.parametrize("rhs_value", [1.0, 0.0, -1.0, 127.0])
def test_maximumf(lhs_value: float, rhs_value: float):
    maxf = arith.MaximumfOp(lhs_f64_op, rhs_f64_op)
    maxf.verify()

    ret = interpreter.run_op(maxf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == max(lhs_value, rhs_value)


def test_minmax_corner():
    maxf = arith.MaximumfOp(lhs_f64_op, rhs_f64_op)
    maxf.verify()

    assert copysign(1.0, interpreter.run_op(maxf, (0.0, 0.0))[0]) == 1.0
    assert copysign(1.0, interpreter.run_op(maxf, (-0.0, 0.0))[0]) == 1.0
    assert copysign(1.0, interpreter.run_op(maxf, (0.0, -0.0))[0]) == 1.0
    assert copysign(1.0, interpreter.run_op(maxf, (-0.0, -0.0))[0]) == -1.0
    assert isnan(interpreter.run_op(maxf, (float("NaN"), 0.0))[0])
    assert isnan(interpreter.run_op(maxf, (0.0, float("NaN")))[0])

    minf = arith.MinimumfOp(lhs_f64_op, rhs_f64_op)
    minf.verify()

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

    if arg.startswith("u"):
        mask = (1 << interpreter.index_bitwidth) - 1
        lhs_value &= mask
        rhs_value &= mask
    assert len(ret) == 1
    assert ret[0] == fn(lhs_value, rhs_value)


@pytest.mark.parametrize(
    "value_type,index_bitwidth,value",
    [
        (i8, 32, 1 << 7),
        (i32, 64, 1 << 31),
        (IndexType(), 32, 1 << 31),
        (IndexType(), 64, 1 << 63),
    ],
)
@pytest.mark.parametrize(
    "predicate,expected_value_minus_one,expected_value",
    [
        ("eq", False, True),
        ("ne", True, False),
        ("slt", True, False),
        ("sle", True, True),
        ("sgt", False, False),
        ("sge", False, True),
        ("ult", False, False),
        ("ule", False, True),
        ("ugt", True, False),
        ("uge", True, True),
    ],
)
def test_cmpi_high_bit(
    value_type: IntegerType | IndexType,
    index_bitwidth: Literal[32, 64],
    value: int,
    predicate: str,
    expected_value_minus_one: bool,
    expected_value: bool,
):
    interpreter = Interpreter(ModuleOp([]), index_bitwidth=index_bitwidth)
    interpreter.register_implementations(ArithFunctions())
    lhs = test.TestOp(result_types=[value_type])
    rhs = test.TestOp(result_types=[value_type])
    cmpi = CmpiOp(lhs, rhs, predicate)
    cmpi.verify()

    # The sign bit alone is the minimum signed value, above all positive signed
    # values when interpreted as unsigned. Both encodings denote the same bits.
    assert interpreter.run_op(cmpi, (value, value - 1)) == (expected_value_minus_one,)
    assert interpreter.run_op(cmpi, (-value, value - 1)) == (expected_value_minus_one,)

    assert interpreter.run_op(cmpi, (-value, value)) == (expected_value,)


@pytest.mark.parametrize("lhs_value", [1.5, 0.5, -1.5, 127.5, float("nan")])
@pytest.mark.parametrize("rhs_value", [1.5, 0.5, -1.5, 127.5, float("nan")])
@pytest.mark.parametrize(
    "pred",
    [
        "false",
        "oeq",
        "ogt",
        "oge",
        "olt",
        "ole",
        "one",
        "ord",
        "ueq",
        "ugt",
        "uge",
        "ult",
        "ule",
        "une",
        "uno",
        "true",
    ],
)
def test_cmpf(lhs_value: float, rhs_value: float, pred: str):
    x = lhs_value
    y = rhs_value

    o = not isnan(x) and not isnan(y)
    u = isnan(x) or isnan(y)

    match pred:
        case "false":
            expectation = False
        case "oeq":
            expectation = (x == y) and o
        case "ogt":
            expectation = (x > y) and o
        case "oge":
            expectation = (x >= y) and o
        case "olt":
            expectation = (x < y) and o
        case "ole":
            expectation = (x <= y) and o
        case "one":
            expectation = (x != y) and o
        case "ord":
            expectation = o
        case "ueq":
            expectation = (x == y) or u
        case "ugt":
            expectation = (x > y) or u
        case "uge":
            expectation = (x >= y) or u
        case "ult":
            expectation = (x < y) or u
        case "ule":
            expectation = (x <= y) or u
        case "une":
            expectation = (x != y) or u
        case "uno":
            expectation = u
        case "true":
            expectation = True
        case _:
            expectation = None

    cmpf = CmpfOp(lhs_f64_op, rhs_f64_op, pred)
    cmpf.verify()

    ret = interpreter.run_op(cmpf, (lhs_value, rhs_value))

    assert len(ret) == 1
    assert ret[0] == expectation


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
    "value_type,index_bitwidth,sign_bit",
    [
        (i8, 32, 128),
        (i32, 64, 1 << 31),
        (IndexType(), 32, 1 << 31),
        (IndexType(), 64, 1 << 63),
    ],
)
def test_shifts_bitwidth(
    value_type: IntegerType | IndexType,
    index_bitwidth: Literal[32, 64],
    sign_bit: int,
):
    interpreter = Interpreter(ModuleOp([]), index_bitwidth=index_bitwidth)
    interpreter.register_implementations(ArithFunctions())
    lhs = test.TestOp(result_types=[value_type])
    rhs = test.TestOp(result_types=[value_type])
    shl = ShLIOp(lhs, rhs)
    shr = ShRSIOp(lhs, rhs)
    shl.verify()
    shr.verify()

    assert interpreter.run_op(shl, (sign_bit // 2, 1)) == (-sign_bit,)
    assert interpreter.run_op(shl, (-sign_bit, 1)) == (0,)
    assert interpreter.run_op(shr, (sign_bit, 1)) == (-sign_bit // 2,)
    assert interpreter.run_op(shr, (-sign_bit, 1)) == (-sign_bit // 2,)


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


@pytest.mark.parametrize(
    "value_type,index_bitwidth,modulus",
    [
        (i8, 32, 256),
        (i32, 64, 1 << 32),
        (IndexType(), 32, 1 << 32),
        (IndexType(), 64, 1 << 64),
    ],
)
@pytest.mark.parametrize(
    "op_type,negative_lhs_result,negative_rhs_result",
    [(DivSIOp, -1, -1), (RemSIOp, -2, 2), (FloorDivSIOp, -2, -2)],
)
def test_signed_division_bitwidth(
    value_type: IntegerType | IndexType,
    index_bitwidth: Literal[32, 64],
    modulus: int,
    op_type: type[DivSIOp | RemSIOp | FloorDivSIOp],
    negative_lhs_result: int,
    negative_rhs_result: int,
):
    interpreter = Interpreter(ModuleOp([]), index_bitwidth=index_bitwidth)
    interpreter.register_implementations(ArithFunctions())
    lhs = test.TestOp(result_types=[value_type])
    rhs = test.TestOp(result_types=[value_type])
    op = op_type(lhs, rhs)
    op.verify()

    assert interpreter.run_op(op, (modulus - 5, 3)) == (negative_lhs_result,)
    assert interpreter.run_op(op, (-5, 3)) == (negative_lhs_result,)
    assert interpreter.run_op(op, (5, modulus - 3)) == (negative_rhs_result,)
    assert interpreter.run_op(op, (5, -3)) == (negative_rhs_result,)


@pytest.mark.parametrize("cond_value", [1, 0])
def test_select(cond_value: int):
    cond_op = test.TestOp(result_types=[i1])
    select = SelectOp(cond_op, lhs_op, rhs_op)

    ret = interpreter.run_op(select, (cond_value, 1, 2))

    assert len(ret) == 1
    assert ret[0] == (1 if cond_value else 2)


@pytest.mark.parametrize("cond_value", [1, 0])
def test_select_scalar_cond_shaped_operands(cond_value: int):
    vector_type = VectorType(i32, (2, 2))
    cond_op = test.TestOp(result_types=[i1])
    vector_lhs_op = test.TestOp(result_types=[vector_type])
    vector_rhs_op = test.TestOp(result_types=[vector_type])
    select = SelectOp(cond_op, vector_lhs_op, vector_rhs_op)

    lhs = ShapedArray(TypedPtr.new_int32([1, 2, 3, 4]), [2, 2])
    rhs = ShapedArray(TypedPtr.new_int32([5, 6, 7, 8]), [2, 2])

    ret = interpreter.run_op(select, (cond_value, lhs, rhs))

    assert len(ret) == 1
    assert ret[0] == (lhs if cond_value else rhs)


def test_select_shaped_cond():
    vector_type = VectorType(i32, (2, 2))
    cond_op = test.TestOp(result_types=[VectorType(i1, (2, 2))])
    vector_lhs_op = test.TestOp(result_types=[vector_type])
    vector_rhs_op = test.TestOp(result_types=[vector_type])
    select = SelectOp(cond_op, vector_lhs_op, vector_rhs_op)

    cond = ShapedArray(TypedPtr[int].new([1, 0, 0, 1], xtype=i1), [2, 2])
    lhs = ShapedArray(TypedPtr.new_int32([1, 2, 3, 4]), [2, 2])
    rhs = ShapedArray(TypedPtr.new_int32([5, 6, 7, 8]), [2, 2])

    ret = interpreter.run_op(select, (cond, lhs, rhs))

    assert len(ret) == 1
    assert ret[0] == ShapedArray(TypedPtr.new_int32([1, 6, 7, 4]), [2, 2])


def test_select_shaped_cond_shape_mismatch():
    cond_op = test.TestOp(result_types=[VectorType(i1, (4,))])
    vector_lhs_op = test.TestOp(result_types=[VectorType(i32, (2, 2))])
    vector_rhs_op = test.TestOp(result_types=[VectorType(i32, (2, 2))])
    select = SelectOp(cond_op, vector_lhs_op, vector_rhs_op)

    cond = ShapedArray(TypedPtr[int].new([1, 0, 0, 1], xtype=i1), [4])
    lhs = ShapedArray(TypedPtr.new_int32([1, 2, 3, 4]), [2, 2])
    rhs = ShapedArray(TypedPtr.new_int32([5, 6, 7, 8]), [2, 2])

    with pytest.raises(InterpretationError, match=r"arith\.select shape mismatch"):
        interpreter.run_op(select, (cond, lhs, rhs))


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


@pytest.mark.parametrize("index_bitwidth", [32, 64])
def test_indexcast_same_width(index_bitwidth: Literal[32, 64]):
    interpreter = Interpreter(ModuleOp([]), index_bitwidth=index_bitwidth)
    interpreter.register_implementations(ArithFunctions())
    int_type = IntegerType(index_bitwidth)
    index_type = IndexType()
    for input_type, output_type in ((int_type, index_type), (index_type, int_type)):
        op = IndexCastOp(test.TestOp(result_types=[input_type]), output_type)
        op.verify()
        assert interpreter.run_op(op, ((1 << index_bitwidth) - 1,)) == (-1,)
        assert interpreter.run_op(op, (-1,)) == (-1,)
