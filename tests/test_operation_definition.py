from __future__ import annotations
import pytest
from xdsl.dialects.builtin import IntAttr, StringAttr

from xdsl.ir import Attribute, OpResult, Operation, Region
from xdsl.irdl import (Operand, irdl_op_definition, OperandDef, ResultDef,
                       AttributeDef, AnyAttr, OpDef, RegionDef, OpAttr)
from xdsl.utils.exceptions import PyRDLOpDefinitionError, VerifyException


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


def test_invalid_typed_field():
    """Check that typed fields are not allowed"""
    with pytest.raises(PyRDLOpDefinitionError):

        class InvalidTypedFieldTestOp(Operation):
            name = "test.invalid_typed_field"

            field: int


def test_invalid_field():
    """Check that untyped fields are not allowed"""
    with pytest.raises(PyRDLOpDefinitionError):

        class InvalidFieldTestOp(Operation):
            name = "test.invalid_field"

            field = 2


class AttrOp(Operation):
    name: str = "test.two_var_result_op"
    attr: OpAttr[StringAttr]


def test_attr_verify():
    op = AttrOp.create(attributes={"attr": IntAttr.from_int(1)})
    with pytest.raises(VerifyException) as e:
        op.verify()
    assert e.value.args[0] == "!int<1> should be of base attribute string"
