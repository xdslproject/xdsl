import pytest

from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.dialects.smt import (
    AndOp,
    ApplyFuncOp,
    AssertOp,
    BoolType,
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
    VariadicBoolOp,
    XOrOp,
    YieldOp,
)
from xdsl.ir import Block, Region
from xdsl.utils.test_value import create_ssa_value


def test_constant_bool():
    op = ConstantBoolOp(True)
    assert op.value is True
    assert op.value_attr == IntegerAttr(-1, 1)

    op = ConstantBoolOp(False)
    assert op.value is False
    assert op.value_attr == IntegerAttr(0, 1)


def test_function_type():
    func_type = FuncType([BoolType(), BoolType()], BoolType())
    assert list(func_type.domain_types) == [BoolType(), BoolType()]
    assert func_type.range_type == BoolType()


def test_not_op():
    arg = create_ssa_value(BoolType())
    op = NotOp(arg)
    assert op.result.type == BoolType()
    assert op.input == arg


@pytest.mark.parametrize("op_type", [AndOp, OrOp, XOrOp, EqOp, DistinctOp])
def test_variadic_bool_op(op_type: type[VariadicBoolOp]):
    arg1 = create_ssa_value(BoolType())
    arg2 = create_ssa_value(BoolType())
    arg3 = create_ssa_value(BoolType())
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


def test_apply_func():
    func = create_ssa_value(FuncType([BoolType(), BoolType()], BoolType()))
    arg1 = create_ssa_value(BoolType())
    arg2 = create_ssa_value(BoolType())
    op = ApplyFuncOp(func, arg1, arg2)

    assert op.result.type == BoolType()


def test_ite():
    arg1 = create_ssa_value(BoolType())
    arg2 = create_ssa_value(BoolType())
    arg3 = create_ssa_value(BoolType())
    op = IteOp(arg1, arg2, arg3)
    assert op.result.type == BoolType()
    assert op.cond == arg1
    assert op.then_value == arg2
    assert op.else_value == arg3


def test_assert_op():
    arg1 = create_ssa_value(BoolType())
    assert_op = AssertOp(arg1)
    assert assert_op.input == arg1
