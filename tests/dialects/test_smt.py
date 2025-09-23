import pytest

from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.dialects.smt import (
    AndOp,
    ApplyFuncOp,
    AssertOp,
    BinaryBVOp,
    BitVectorAttr,
    BitVectorType,
    BoolType,
    BVAddOp,
    BVAndOp,
    BVAShrOp,
    BvConstantOp,
    BVLShrOp,
    BVMulOp,
    BVNegOp,
    BVNotOp,
    BVOrOp,
    BVSDivOp,
    BVShlOp,
    BVSModOp,
    BVSRemOp,
    BVUDivOp,
    BVURemOp,
    BVXOrOp,
    ConstantBoolOp,
    DeclareFunOp,
    DistinctOp,
    EqOp,
    ExistsOp,
    ForallOp,
    FuncType,
    ImpliesOp,
    IteOp,
    NotOp,
    OrOp,
    QuantifierOp,
    UnaryBVOp,
    VariadicBoolOp,
    XOrOp,
    YieldOp,
)
from xdsl.ir import Block, Region
from xdsl.traits import ConstantLike
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


def test_constant_bool():
    op = ConstantBoolOp(True)
    assert op.value is True
    assert op.value_attr == IntegerAttr(-1, 1)
    constantlike = op.get_trait(ConstantLike)
    assert constantlike is not None
    assert constantlike.get_constant_value(op) == IntegerAttr(-1, 1)

    op = ConstantBoolOp(False)
    assert op.value is False
    assert op.value_attr == IntegerAttr(0, 1)
    constantlike = op.get_trait(ConstantLike)
    assert constantlike is not None
    assert constantlike.get_constant_value(op) == IntegerAttr(0, 1)


def test_bv_type():
    bv_type = BitVectorType(32)
    assert bv_type.width.data == 32
    assert bv_type.value_range() == (0, 2**32)

    bv_type = BitVectorType(3)
    assert bv_type.width.data == 3
    assert bv_type.value_range() == (0, 2**3)

    with pytest.raises(
        VerifyException,
        match="BitVectorType width must be strictly greater than zero, got 0",
    ):
        BitVectorType(0)


def test_function_type():
    func_type = FuncType([BoolType(), BitVectorType(32)], BoolType())
    assert list(func_type.domain_types) == [BoolType(), BitVectorType(32)]
    assert func_type.range_type == BoolType()


def test_not_op():
    arg = create_ssa_value(BoolType())
    op = NotOp(arg)
    assert op.result.type == BoolType()
    assert op.input == arg


@pytest.mark.parametrize("op_type", [AndOp, OrOp, XOrOp])
def test_variadic_bool_op(op_type: type[VariadicBoolOp]):
    arg1 = create_ssa_value(BoolType())
    arg2 = create_ssa_value(BoolType())
    arg3 = create_ssa_value(BoolType())
    op = op_type(arg1, arg2, arg3)
    assert op.result.type == BoolType()
    assert list(op.inputs) == [arg1, arg2, arg3]


@pytest.mark.parametrize("op_type", [EqOp, DistinctOp])
@pytest.mark.parametrize("type", [BoolType(), BitVectorType(32)])
def test_eq_distinct_op(op_type: type[VariadicBoolOp], type: BoolType | BitVectorType):
    arg1 = create_ssa_value(type)
    arg2 = create_ssa_value(type)
    arg3 = create_ssa_value(type)
    op = op_type(arg1, arg2, arg3)
    assert op.result.type == BoolType()
    assert list(op.inputs) == [arg1, arg2, arg3]


def test_implies_op():
    arg1 = create_ssa_value(BoolType())
    arg2 = create_ssa_value(BoolType())
    op = ImpliesOp(arg1, arg2)
    assert op.result.type == BoolType()
    assert op.lhs == arg1
    assert op.rhs == arg2


@pytest.mark.parametrize("op_type", [ExistsOp, ForallOp])
def test_quantifier_op(op_type: type[QuantifierOp]):
    arg1 = create_ssa_value(BoolType())
    region = Region([Block()])
    region.block.add_op(YieldOp(arg1))
    op = op_type(body=region)
    assert op.returned_value == arg1


def test_declare_fun():
    op = DeclareFunOp(BoolType(), "foo")
    assert op.name_prefix == StringAttr("foo")
    assert op.result.type == BoolType()

    op = DeclareFunOp(BoolType())
    assert op.name_prefix is None
    assert op.result.type == BoolType()

    op = DeclareFunOp(BitVectorType(32))
    assert op.name_prefix is None
    assert op.result.type == BitVectorType(32)


def test_apply_func():
    func = create_ssa_value(FuncType([BoolType(), BitVectorType(32)], BoolType()))
    arg1 = create_ssa_value(BoolType())
    arg2 = create_ssa_value(BitVectorType(32))
    op = ApplyFuncOp(func, arg1, arg2)

    assert op.result.type == BoolType()


def test_ite():
    arg1 = create_ssa_value(BoolType())
    arg2 = create_ssa_value(BitVectorType(32))
    arg3 = create_ssa_value(BitVectorType(32))
    op = IteOp(arg1, arg2, arg3)
    assert op.result.type == BitVectorType(32)
    assert op.cond == arg1
    assert op.then_value == arg2
    assert op.else_value == arg3


def test_assert_op():
    arg1 = create_ssa_value(BoolType())
    assert_op = AssertOp(arg1)
    assert assert_op.input == arg1


def test_bv_attr():
    bv_attr = BitVectorAttr(0, 32)
    assert bv_attr.value.data == 0
    assert bv_attr.type == BitVectorType(32)

    with pytest.raises(VerifyException, match="is out of range"):
        bv_attr = BitVectorAttr(-1, 32)

    with pytest.raises(VerifyException, match="is out of range"):
        bv_attr = BitVectorAttr(2**32, 32)


def test_bv_constant_op():
    bv_attr = BitVectorAttr(42, 32)
    op = BvConstantOp(bv_attr)
    assert op.value == bv_attr
    assert op.result.type == BitVectorType(32)
    constantlike = op.get_trait(ConstantLike)
    assert constantlike is not None
    assert constantlike.get_constant_value(op) == bv_attr

    op2 = BvConstantOp(42, 32)
    assert op2.value == bv_attr
    assert op2.result.type == BitVectorType(32)
    constantlike2 = op2.get_trait(ConstantLike)
    assert constantlike2 is not None
    assert constantlike2.get_constant_value(op2) == bv_attr

    op3 = BvConstantOp(42, BitVectorType(32))
    assert op3.value == bv_attr
    assert op3.result.type == BitVectorType(32)
    constantlike3 = op3.get_trait(ConstantLike)
    assert constantlike3 is not None
    assert constantlike3.get_constant_value(op3) == bv_attr


@pytest.mark.parametrize("op_type", [BVNotOp, BVNegOp])
def test_bv_unary_op(op_type: type[UnaryBVOp]):
    arg = create_ssa_value(BitVectorType(32))
    op = op_type(arg)
    assert op.input == arg
    assert op.result.type == arg.type


@pytest.mark.parametrize(
    "op_type",
    [
        BVAndOp,
        BVOrOp,
        BVXOrOp,
        BVAddOp,
        BVMulOp,
        BVUDivOp,
        BVSDivOp,
        BVURemOp,
        BVSRemOp,
        BVSModOp,
        BVShlOp,
        BVLShrOp,
        BVAShrOp,
    ],
)
def test_bv_binary_op(op_type: type[BinaryBVOp]):
    arg1 = create_ssa_value(BitVectorType(32))
    arg2 = create_ssa_value(BitVectorType(32))
    op = op_type(arg1, arg2)
    assert op.lhs == arg1
    assert op.rhs == arg2
    assert op.result.type == BitVectorType(32)
