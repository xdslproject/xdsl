import pytest

from xdsl.dialects.builtin import StringAttr, SymbolRefAttr, i32
from xdsl.dialects.irdl import (
    AllOfOp,
    AnyOfOp,
    AnyOp,
    AttributeOp,
    AttributeType,
    DialectOp,
    IsOp,
    OperandsOp,
    OperationOp,
    ParametersOp,
    ParametricOp,
    ResultsOp,
    TypeOp,
)
from xdsl.ir import Block, Region
from xdsl.utils.test_value import TestSSAValue


@pytest.mark.parametrize("op_type", [DialectOp, TypeOp, AttributeOp, OperationOp])
def test_named_region_op_init(
    op_type: type[DialectOp | TypeOp | AttributeOp | OperationOp],
):
    """
    Test __init__ of DialectOp, TypeOp, AttributeOp, OperationOp.
    """
    op = op_type("cmath", Region(Block()))
    op2 = op_type(StringAttr("cmath"), Region(Block()))
    op3 = op_type.create(
        attributes={"sym_name": StringAttr("cmath")}, regions=[Region(Block())]
    )

    assert op.is_structurally_equivalent(op2)
    assert op2.is_structurally_equivalent(op3)

    assert op.sym_name == StringAttr("cmath")
    assert len(op.body.blocks) == 1


@pytest.mark.parametrize("op_type", [ParametersOp, OperandsOp, ResultsOp])
def test_parameters_init(op_type: type[ParametersOp | OperandsOp | ResultsOp]):
    """
    Test __init__ of ParametersOp, OperandsOp, ResultsOp.
    """

    val1 = TestSSAValue(AttributeType())
    val2 = TestSSAValue(AttributeType())
    op = op_type([val1, val2])
    op2 = op_type.create(operands=[val1, val2])

    assert op.is_structurally_equivalent(op2)

    assert op.args == (val1, val2)


def test_is_init():
    """Test __init__ of IsOp."""
    op = IsOp(i32)
    op2 = IsOp.create(attributes={"expected": i32}, result_types=[AttributeType()])

    assert op.is_structurally_equivalent(op2)

    assert op.expected == i32
    assert op.output.type == AttributeType()


def test_parametric_init():
    """Test __init__ of ParametricOp."""
    val1 = TestSSAValue(AttributeType())
    val2 = TestSSAValue(AttributeType())

    op = ParametricOp("complex", [val1, val2])
    op2 = ParametricOp(StringAttr("complex"), [val1, val2])
    op3 = ParametricOp(SymbolRefAttr("complex"), [val1, val2])
    op4 = ParametricOp.create(
        attributes={"base_type": SymbolRefAttr("complex")},
        operands=[val1, val2],
        result_types=[AttributeType()],
    )

    assert op.is_structurally_equivalent(op2)
    assert op2.is_structurally_equivalent(op3)
    assert op3.is_structurally_equivalent(op4)

    assert op.base_type == SymbolRefAttr("complex")
    assert op.args == (val1, val2)
    assert op.output.type == AttributeType()


def test_any_init():
    """Test __init__ of AnyOp."""
    op = AnyOp()
    op2 = AnyOp.create(result_types=[AttributeType()])

    assert op.is_structurally_equivalent(op2)
    assert op.output.type == AttributeType()


@pytest.mark.parametrize("op_type", [AllOfOp, AnyOfOp])
def test_any_all_of_init(op_type: type[AllOfOp | AnyOfOp]):
    """Test __init__ of AnyOf and AllOf."""
    val1 = TestSSAValue(AttributeType())
    val2 = TestSSAValue(AttributeType())
    op = op_type((val1, val2))
    op2 = op_type.create(operands=[val1, val2], result_types=[AttributeType()])

    assert op.is_structurally_equivalent(op2)

    assert op.args == (val1, val2)
    assert op.output.type == AttributeType()
