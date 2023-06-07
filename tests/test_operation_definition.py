from __future__ import annotations
from typing import Annotated, Generic, TypeVar

import pytest

from xdsl.dialects.builtin import IndexType, IntAttr, IntegerType, StringAttr, i32
from xdsl.dialects.test import TestType

from xdsl.utils.test_value import TestSSAValue
from xdsl.ir import Attribute, OpResult, Region
from xdsl.irdl import (
    AttrSizedOperandSegments,
    AttrSizedRegionSegments,
    AttrSizedResultSegments,
    ConstraintVar,
    Operand,
    OptOpAttr,
    OptOpResult,
    OptOperand,
    OptRegion,
    VarOpResult,
    VarOperand,
    VarRegion,
    irdl_op_definition,
    OperandDef,
    ResultDef,
    AttributeDef,
    AnyAttr,
    OpDef,
    RegionDef,
    OpAttr,
    IRDLOperation,
)
from xdsl.utils.exceptions import (
    DiagnosticException,
    PyRDLOpDefinitionError,
    VerifyException,
)

################################################################################
#                              IRDL definition                                 #
################################################################################


@irdl_op_definition
class OpDefTestOp(IRDLOperation):
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
        regions=[("region", RegionDef())],
    )


################################################################################
#                            Invalid definitions                               #
################################################################################


class InvalidTypedFieldTestOp(IRDLOperation):
    name = "test.invalid_typed_field"

    field: int


def test_invalid_typed_field():
    """Check that typed fields are not allowed"""
    with pytest.raises(PyRDLOpDefinitionError):
        irdl_op_definition(InvalidTypedFieldTestOp)


class InvalidFieldTestOp(IRDLOperation):
    name = "test.invalid_field"

    field = 2


def test_invalid_field():
    """Check that untyped fields are not allowed"""
    with pytest.raises(PyRDLOpDefinitionError):
        irdl_op_definition(InvalidFieldTestOp)


################################################################################
#                                  Verifiers                                   #
################################################################################


@irdl_op_definition
class AttrOp(IRDLOperation):
    name = "test.two_var_result_op"
    attr: OpAttr[StringAttr]


def test_attr_verify():
    op = AttrOp.create(attributes={"attr": IntAttr(1)})
    with pytest.raises(
        VerifyException, match="#int<1> should be of base attribute string"
    ):
        op.verify()


@irdl_op_definition
class ConstraintVarOp(IRDLOperation):
    name = "test.constraint_var_op"

    T = Annotated[IntegerType | IndexType, ConstraintVar("T")]

    operand: Annotated[Operand, T]
    result: Annotated[OpResult, T]
    attribute: OpAttr[T]


def test_constraint_var():
    i32_operand = TestSSAValue(i32)
    index_operand = TestSSAValue(IndexType())
    op = ConstraintVarOp.create(
        operands=[i32_operand], result_types=[i32], attributes={"attribute": i32}
    )
    op.verify()

    op2 = ConstraintVarOp.create(
        operands=[index_operand],
        result_types=[IndexType()],
        attributes={"attribute": IndexType()},
    )
    op2.verify()


def test_constraint_var_fail_non_equal():
    """Check that all uses of a constraint variable are of the same attribute."""
    i32_operand = TestSSAValue(i32)
    index_operand = TestSSAValue(IndexType())

    # Fail because of operand
    op = ConstraintVarOp.create(
        operands=[index_operand], result_types=[i32], attributes={"attribute": i32}
    )
    with pytest.raises(DiagnosticException):
        op.verify()

    # Fail because of result
    op2 = ConstraintVarOp.create(
        operands=[i32_operand],
        result_types=[IndexType()],
        attributes={"attribute": i32},
    )
    with pytest.raises(DiagnosticException):
        op2.verify()

    # Fail because of attribute
    op3 = ConstraintVarOp.create(
        operands=[i32_operand],
        result_types=[i32],
        attributes={"attribute": IndexType()},
    )
    with pytest.raises(DiagnosticException):
        op3.verify()


def test_constraint_var_fail_not_satisfy_constraint():
    """Check that all uses of a constraint variable are satisfying the constraint."""
    test_operand = TestSSAValue(TestType("foo"))
    op = ConstraintVarOp.create(
        operands=[test_operand],
        result_types=[TestType("foo")],
        attributes={"attribute": TestType("foo")},
    )
    with pytest.raises(DiagnosticException):
        op.verify()


################################################################################
#                                Accessors                                     #
################################################################################


@irdl_op_definition
class RegionOp(IRDLOperation):
    name = "test.region_op"

    irdl_options = [AttrSizedRegionSegments()]

    region: Region
    opt_region: OptRegion
    var_region: VarRegion


def test_region_accessors():
    """Test accessors for regions."""
    region1 = Region()
    region2 = Region()
    region3 = Region()
    region4 = Region()

    op = RegionOp.build(regions=[region1, [region2], [region3, region4]])
    assert op.region is op.regions[0]
    assert op.opt_region is op.regions[1]
    assert len(op.var_region) == 2
    assert op.var_region[0] is op.regions[2]
    assert op.var_region[1] is op.regions[3]

    region1 = Region()

    op = RegionOp.build(regions=[region1, [], []])
    assert op.opt_region is None
    assert len(op.var_region) == 0


@irdl_op_definition
class OperandOp(IRDLOperation):
    name = "test.operand_op"

    irdl_options = [AttrSizedOperandSegments()]

    operand: Operand
    opt_operand: OptOperand
    var_operand: VarOperand


def test_operand_accessors():
    """Test accessors for operands."""
    operand1 = OpResult(i32, None, None)  # type: ignore
    operand2 = OpResult(i32, None, None)  # type: ignore
    operand3 = OpResult(i32, None, None)  # type: ignore
    operand4 = OpResult(i32, None, None)  # type: ignore

    op = OperandOp.build(operands=[operand1, [operand2], [operand3, operand4]])
    assert op.operand is op.operands[0]
    assert op.opt_operand is op.operands[1]
    assert len(op.var_operand) == 2
    assert op.var_operand[0] is op.operands[2]
    assert op.var_operand[1] is op.operands[3]

    op = OperandOp.build(operands=[operand1, [], []])
    assert op.opt_operand is None
    assert len(op.var_operand) == 0


@irdl_op_definition
class OpResultOp(IRDLOperation):
    name = "test.op_result_op"

    irdl_options = [AttrSizedResultSegments()]

    result: OpResult
    opt_result: OptOpResult
    var_result: VarOpResult


def test_opresult_accessors():
    """Test accessors for results."""
    op = OpResultOp.build(result_types=[i32, [i32], [i32, i32]])
    assert op.result is op.results[0]
    assert op.opt_result is op.results[1]
    assert len(op.var_result) == 2
    assert op.var_result[0] is op.results[2]
    assert op.var_result[1] is op.results[3]

    op = OpResultOp.build(result_types=[i32, [], []])
    assert op.opt_result is None
    assert len(op.var_result) == 0


@irdl_op_definition
class AttributeOp(IRDLOperation):
    name = "test.attribute_op"

    attr: OpAttr[StringAttr]
    opt_attr: OptOpAttr[StringAttr]


def test_attribute_accessors():
    """Test accessors for attributes."""

    op = AttributeOp.create(
        attributes={"attr": StringAttr("test"), "opt_attr": StringAttr("opt_test")}
    )
    assert op.attr is op.attributes["attr"]
    assert op.opt_attr is op.attributes["opt_attr"]

    op = AttributeOp.create(attributes={"attr": StringAttr("test")})
    assert op.opt_attr is None


def test_attribute_setters():
    """Test setters for attributes."""

    op = AttributeOp.create(attributes={"attr": StringAttr("test")})

    op.attr = StringAttr("new_test")
    assert op.attr.data == "new_test"

    op.opt_attr = StringAttr("new_opt_test")
    assert op.opt_attr.data == "new_opt_test"

    op.opt_attr = None
    assert op.opt_attr is None


################################################################################
#                             Generic operation                                #
################################################################################

FooType = Annotated[TestType, TestType("foo")]
BarType = Annotated[TestType, TestType("bar")]


_Attr = TypeVar("_Attr", bound=StringAttr | IntAttr)
_Operand = TypeVar("_Operand", bound=FooType | BarType)
_Result = TypeVar("_Result", bound=FooType | BarType)


class GenericOp(Generic[_Attr, _Operand, _Result], IRDLOperation):
    name = "test.string_or_int_generic"

    attr: OpAttr[_Attr]
    operand: Annotated[Operand, _Operand]
    result: Annotated[OpResult, _Result]


@irdl_op_definition
class StringFooOp(GenericOp[StringAttr, FooType, FooType]):
    name = "test.string_specialized"


def test_generic_op():
    """Test generic operation."""
    FooOperand = TestSSAValue(TestType("foo"))
    BarOperand = TestSSAValue(TestType("bar"))
    FooResultType = TestType("foo")
    BarResultType = TestType("bar")

    op = StringFooOp(
        attributes={"attr": StringAttr("test")},
        operands=[FooOperand],
        result_types=[FooResultType],
    )
    op.verify()

    op_attr_fail = StringFooOp(
        attributes={"attr": IntAttr(1)},
        operands=[FooOperand],
        result_types=[FooResultType],
    )
    with pytest.raises(DiagnosticException):
        op_attr_fail.verify()

    op_operand_fail = StringFooOp(
        attributes={"attr": StringAttr("test")},
        operands=[BarOperand],
        result_types=[FooResultType],
    )
    with pytest.raises(DiagnosticException):
        op_operand_fail.verify()

    op_result_fail = StringFooOp(
        attributes={"attr": StringAttr("test")},
        operands=[FooOperand],
        result_types=[BarResultType],
    )
    with pytest.raises(DiagnosticException):
        op_result_fail.verify()
