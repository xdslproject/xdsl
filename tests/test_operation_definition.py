from __future__ import annotations
import pytest

from xdsl.ir import Attribute, OpResult, Operation, Region
from xdsl.irdl import (Operand, irdl_op_definition, OperandDef, ResultDef,
                       AttributeDef, AnyAttr, OpDef, RegionDef, OpAttr)
from xdsl.utils.exceptions import PyRDLOpDefinitionError


@irdl_op_definition
class OpDefTestOp(Operation):
    name = "test.op_def_test"

    operand: Operand
    result: OpResult
    attr: OpAttr[Attribute]
    region: Region

    # Check that we can define methods in operation definitions
    def test(self):
        pass


def test_get_definition():
    """Test retrieval of an IRDL definition from an operation"""
    assert OpDefTestOp.irdl_definition == OpDef(
        "test.op_def_test",
        operands=[("operand", OperandDef(AnyAttr()))],
        results=[("result", ResultDef(AnyAttr()))],
        attributes={"attr": AttributeDef(AnyAttr())},
        regions=[("region", RegionDef())])


class InvalidTypedFieldTestOp(Operation):
    name = "test.invalid_typed_field"

    field: int


def test_invalid_typed_field():
    """Check that typed fields are not allowed"""
    with pytest.raises(PyRDLOpDefinitionError):
        irdl_op_definition(InvalidTypedFieldTestOp)


class InvalidFieldTestOp(Operation):
    name = "test.invalid_field"

    field = 2


def test_invalid_field():
    """Check that untyped fields are not allowed"""
    with pytest.raises(PyRDLOpDefinitionError):
        irdl_op_definition(InvalidFieldTestOp)
