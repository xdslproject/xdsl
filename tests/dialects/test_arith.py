from typing import TypeVar

import pytest

from xdsl.dialects.arith import (
    Addf,
    Addi,
    AddUIExtended,
    AndI,
    CeilDivSI,
    CeilDivUI,
    Cmpf,
    Cmpi,
    Constant,
    Divf,
    DivSI,
    DivUI,
    ExtFOp,
    ExtSIOp,
    ExtUIOp,
    FastMathFlagsAttr,
    FloatingPointLikeBinaryOperation,
    FloorDivSI,
    FPToSIOp,
    IndexCastOp,
    Maximumf,
    MaxSI,
    MaxUI,
    Minimumf,
    MinSI,
    MinUI,
    Mulf,
    MulSIExtended,
    MulUIExtended,
    Negf,
    OrI,
    RemSI,
    RemUI,
    Select,
    ShLI,
    ShRSI,
    ShRUI,
    SignlessIntegerBinaryOperation,
    SIToFPOp,
    Subf,
    Subi,
    TruncFOp,
    TruncIOp,
    XOrI,
)
from xdsl.dialects.builtin import (
    AnyTensorType,
    AnyVectorType,
    FloatAttr,
    IndexType,
    IntegerType,
    TensorType,
    VectorType,
    f32,
    f64,
    i1,
    i32,
    i64,
)
from xdsl.ir import Attribute
from xdsl.irdl import base
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.isattr import isattr
from xdsl.utils.test_value import TestSSAValue

_BinOpArgT = TypeVar("_BinOpArgT", bound=Attribute)


class Test_integer_arith_construction:
    operand_type = i32
    a = Constant.from_int_and_width(1, operand_type)
    b = Constant.from_int_and_width(1, operand_type)

    @pytest.mark.parametrize(
        "OpClass",
        [
            Addi,
            Subi,
            DivUI,
            DivSI,
            FloorDivSI,
            CeilDivSI,
            CeilDivUI,
            RemUI,
            RemSI,
            MinUI,
            MinSI,
            MaxUI,
            MaxSI,
            AndI,
            OrI,
            XOrI,
            ShLI,
            ShRUI,
            ShRSI,
        ],
    )
    @pytest.mark.parametrize("return_type", [None, operand_type])
    def test_arith_ops_init(
        self,
        OpClass: type[SignlessIntegerBinaryOperation],
        return_type: Attribute,
    ):
        op = OpClass(self.a, self.b)

        assert isinstance(op, OpClass)
        assert op.lhs.owner is self.a
        assert op.rhs.owner is self.b
        assert op.result.type == self.operand_type

    def test_Cmpi(self):
        _ = Cmpi(self.a, self.b, 2)

    @pytest.mark.parametrize(
        "input",
        ["eq", "ne", "slt", "sle", "ult", "ule", "ugt", "uge"],
    )
    def test_Cmpi_from_mnemonic(self, input: str):
        _ = Cmpi(self.a, self.b, input)


@pytest.mark.parametrize(
    "lhs_type, rhs_type, sum_type, is_correct",
    [
        (i32, i32, None, True),
        (i32, i32, i32, True),
        (i32, i32, i64, False),
        (i32, i64, None, False),
        (VectorType(i32, [4]), VectorType(i32, [4]), None, True),
        (VectorType(i32, [4]), VectorType(i32, [5]), None, False),
        (VectorType(i32, [4]), VectorType(i64, [5]), None, False),
        (VectorType(i32, [4]), VectorType(i32, [4]), VectorType(i32, [4]), True),
        (VectorType(i32, [4]), VectorType(i32, [4]), VectorType(i64, [4]), False),
        (TensorType(i32, [4]), TensorType(i32, [4]), None, True),
        (TensorType(i32, [4]), TensorType(i32, [5]), None, False),
        (TensorType(i32, [4]), TensorType(i64, [5]), None, False),
        (TensorType(i32, [4]), TensorType(i32, [4]), TensorType(i32, [4]), True),
        (TensorType(i32, [4]), TensorType(i32, [4]), TensorType(i1, [4]), False),
        (VectorType(i32, [4]), TensorType(i32, [4]), None, False),
        (VectorType(i32, [4]), TensorType(i32, [4]), TensorType(i32, [4]), False),
    ],
)
def test_addui_extend(
    lhs_type: Attribute,
    rhs_type: Attribute,
    sum_type: Attribute | None,
    is_correct: bool,
):
    lhs = TestSSAValue(lhs_type)
    rhs = TestSSAValue(rhs_type)

    attributes = {"foo": i32}

    if is_correct:
        op = AddUIExtended(lhs, rhs, attributes, sum_type)
        op.verify()
        assert op.lhs == lhs
        assert op.rhs == rhs
        assert op.attributes == attributes
        if sum_type:
            assert op.sum.type == sum_type
        assert op.overflow.type == AddUIExtended.infer_overflow_type(lhs_type)
        if isattr(
            container_type := op.overflow.type,
            base(AnyVectorType) | base(AnyTensorType),
        ):
            assert container_type.element_type == i1
        else:
            assert op.overflow.type == i1
    else:
        with pytest.raises((VerifyException, ValueError)):
            op = AddUIExtended(lhs, rhs, attributes, sum_type)
            op.verify()


@pytest.mark.parametrize("op_type", [MulSIExtended, MulUIExtended])
def test_mul_extended(op_type: type[MulSIExtended | MulUIExtended]):
    lhs = TestSSAValue(i32)
    rhs = TestSSAValue(i32)

    op = op_type(lhs, rhs)

    assert op.lhs == lhs
    assert op.rhs == rhs
    assert op.low.type == i32
    assert op.high.type == i32

    op2 = op_type(lhs, rhs, i64)

    assert op2.lhs == lhs
    assert op2.rhs == rhs
    assert op2.low.type == i64
    assert op2.high.type == i64


class Test_float_arith_construction:
    a = Constant(FloatAttr(1.1, f32))
    b = Constant(FloatAttr(2.2, f32))

    @pytest.mark.parametrize(
        "func",
        [Addf, Subf, Mulf, Divf, Maximumf, Minimumf],
    )
    @pytest.mark.parametrize(
        "flags", [FastMathFlagsAttr("none"), FastMathFlagsAttr("fast"), None]
    )
    def test_arith_ops(
        self,
        func: type[FloatingPointLikeBinaryOperation],
        flags: FastMathFlagsAttr | None,
    ):
        op = func(self.a, self.b, flags)
        assert op.operands[0].owner is self.a
        assert op.operands[1].owner is self.b
        assert op.fastmath == flags


def test_select_op():
    t = Constant.from_int_and_width(1, IntegerType(1))
    f = Constant.from_int_and_width(0, IntegerType(1))
    select_t_op = Select(t, t, f)
    select_f_op = Select(f, t, f)
    select_t_op.verify_()
    select_f_op.verify_()

    # wanting to verify it actually selected the correct operand, but not sure if in correct scope
    assert select_t_op.result.type == t.result.type
    assert select_f_op.result.type == f.result.type


def test_index_cast_op():
    a = Constant.from_int_and_width(0, 32)
    cast = IndexCastOp(a, IndexType())

    assert cast.result.type == IndexType()
    assert cast.input.type == i32
    assert cast.input.owner == a


def test_cast_fp_and_si_ops():
    a = Constant.from_int_and_width(0, 32)
    fp = SIToFPOp(a, f32)
    si = FPToSIOp(fp, i32)

    assert fp.input == a.result
    assert fp.result == si.input
    assert isinstance(si.result.type, IntegerType)
    assert fp.result.type == f32


def test_negf_op():
    a = Constant(FloatAttr(1.0, f32))
    neg_a = Negf(a)

    b = Constant(FloatAttr(1.0, f64))
    neg_b = Negf(b)

    assert neg_a.result.type == f32
    assert neg_b.result.type == f64


def test_extend_truncate_fpops():
    a = Constant(FloatAttr(1.0, f32))
    b = Constant(FloatAttr(2.0, f64))
    ext_op = ExtFOp(a, f64)
    trunc_op = TruncFOp(b, f32)

    assert ext_op.input == a.result
    assert ext_op.result.type == f64
    assert trunc_op.input == b.result
    assert trunc_op.result.type == f32


def test_cmpf_from_mnemonic():
    a = Constant(FloatAttr(1.0, f64))
    b = Constant(FloatAttr(2.0, f64))
    operations = [
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
    ]
    cmpf_ops = [Cmpf(a, b, operations[i]) for i in range(len(operations))]

    for index, op in enumerate(cmpf_ops):
        assert op.lhs.type == f64
        assert op.rhs.type == f64
        assert op.predicate.value.data == index


def test_cmpf_get():
    a = Constant(FloatAttr(1.0, f32))
    b = Constant(FloatAttr(2.0, f32))

    cmpf_op = Cmpf(a, b, 1)

    assert cmpf_op.lhs.type == f32
    assert cmpf_op.rhs.type == f32
    assert cmpf_op.predicate.value.data == 1


def test_cmpf_missmatch_type():
    a = Constant(FloatAttr(1.0, f32))
    b = Constant(FloatAttr(2.0, f64))

    with pytest.raises(TypeError) as e:
        _cmpf_op = Cmpf(a, b, 1)
    assert (
        e.value.args[0]
        == "Comparison operands must have same type, but provided f32 and f64"
    )


def test_cmpi_mismatch_type():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i64)

    with pytest.raises(TypeError) as e:
        _cmpi_op = Cmpi(a, b, 1)
    assert (
        e.value.args[0]
        == "Comparison operands must have same type, but provided i32 and i64"
    )


def test_cmpf_incorrect_comparison():
    a = Constant(FloatAttr(1.0, f32))
    b = Constant(FloatAttr(2.0, f32))

    with pytest.raises(VerifyException) as e:
        # 'eq' is a comparison op for cmpi but not cmpf
        _cmpf_op = Cmpf(a, b, "eq")
    assert e.value.args[0] == "Unknown comparison mnemonic: eq"


def test_cmpi_incorrect_comparison():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)

    with pytest.raises(VerifyException) as e:
        # 'oeq' is a comparison op for cmpf but not cmpi
        _cmpi_op = Cmpi(a, b, "oeq")
    assert e.value.args[0] == "Unknown comparison mnemonic: oeq"


def test_cmpi_index_type():
    a = Constant.from_int_and_width(1, IndexType())
    b = Constant.from_int_and_width(2, IndexType())
    Cmpi(a, b, "eq").verify()


def test_extend_truncate_iops():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i64)
    exts_op = ExtSIOp(a, i64)
    extu_op = ExtUIOp(a, i64)
    trunc_op = TruncIOp(b, i32)
    exts_op.verify()
    extu_op.verify()
    trunc_op.verify()

    assert exts_op.input == a.result
    assert exts_op.result.type == i64
    assert extu_op.input == a.result
    assert extu_op.result.type == i64
    assert trunc_op.input == b.result
    assert trunc_op.result.type == i32


def test_trunci_incorrect_bitwidth():
    a = Constant.from_int_and_width(1, 16)
    # bitwidth of b has to be smaller than the one of a
    with pytest.raises(VerifyException):
        _trunci_op = TruncIOp(a, i32).verify()


def test_extui_incorrect_bitwidth():
    a = Constant.from_int_and_width(1, 64)
    # bitwidth of b has to be larger than the one of a
    with pytest.raises(VerifyException):
        _extui_op = ExtUIOp(a, i32).verify()
