from __future__ import annotations

from typing import Annotated, Generic, TypeVar

import pytest

from xdsl.dialects.builtin import (
    DenseArrayBase,
    IndexType,
    IntAttr,
    IntegerType,
    StringAttr,
    i32,
)
from xdsl.dialects.test import TestType
from xdsl.ir import Attribute, OpResult, Region
from xdsl.irdl import (
    AnyAttr,
    AttributeDef,
    AttrSizedOperandSegments,
    AttrSizedRegionSegments,
    AttrSizedResultSegments,
    BaseAttr,
    ConstraintVar,
    IRDLOperation,
    OpDef,
    Operand,
    OperandDef,
    OptOperand,
    OptOpResult,
    OptRegion,
    PropertyDef,
    RegionDef,
    ResultDef,
    VarOperand,
    VarOpResult,
    VarRegion,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_prop_def,
    opt_region_def,
    opt_result_def,
    prop_def,
    region_def,
    result_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)
from xdsl.utils.exceptions import (
    DiagnosticException,
    PyRDLOpDefinitionError,
    VerifyException,
)
from xdsl.utils.test_value import TestSSAValue

################################################################################
#                              IRDL definition                                 #
################################################################################


@irdl_op_definition
class OpDefTestOp(IRDLOperation):
    name = "test.op_def_test"

    irdl_options = [AttrSizedOperandSegments()]

    operand: Operand = operand_def()
    result: OpResult = result_def()
    prop: Attribute = prop_def(Attribute)
    attr: Attribute = attr_def(Attribute)
    region: Region = region_def()

    # Check that we can define methods in operation definitions
    def test(self):
        pass


def test_get_definition():
    """Test retrieval of an IRDL definition from an operation"""
    assert OpDefTestOp.get_irdl_definition() == OpDef(
        "test.op_def_test",
        operands=[("operand", OperandDef(AnyAttr()))],
        results=[("result", ResultDef(AnyAttr()))],
        attributes={
            "attr": AttributeDef(AnyAttr()),
            "operandSegmentSizes": AttributeDef(BaseAttr(DenseArrayBase)),
        },
        properties={"prop": PropertyDef(AnyAttr())},
        regions=[("region", RegionDef())],
        accessor_names={"attr": ("attr", "attribute"), "prop": ("prop", "property")},
        options=[AttrSizedOperandSegments()],
    )


@irdl_op_definition
class PropOptionOp(IRDLOperation):
    name = "test.prop_option_test"

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


def test_property_option():
    """Test retrieval of an IRDL definition from an operation"""
    assert PropOptionOp.get_irdl_definition() == OpDef(
        "test.prop_option_test",
        properties={"operandSegmentSizes": PropertyDef(BaseAttr(DenseArrayBase))},
        options=[AttrSizedOperandSegments(as_property=True)],
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
    attr: StringAttr = attr_def(StringAttr)


def test_attr_verify():
    op = AttrOp.create(attributes={"attr": IntAttr(1)})
    with pytest.raises(
        VerifyException, match="#builtin.int<1> should be of base attribute string"
    ):
        op.verify()


@irdl_op_definition
class ConstraintVarOp(IRDLOperation):
    name = "test.constraint_var_op"

    T = Annotated[IntegerType | IndexType, ConstraintVar("T")]

    operand: Operand = operand_def(T)
    result: OpResult = result_def(T)
    attribute: T = attr_def(T)


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


@irdl_op_definition
class OperationWithoutProperty(IRDLOperation):
    name = "test.op_without_prop"

    prop1: Attribute = prop_def(Attribute)


# Check that an operation cannot accept properties that are not defined
def test_unknown_property():
    op = OperationWithoutProperty.create(properties={"prop1": i32, "prop2": i32})
    with pytest.raises(
        VerifyException, match="property 'prop2' is not defined by the operation"
    ):
        op.verify()


################################################################################
#                                Accessors                                     #
################################################################################


@irdl_op_definition
class RegionOp(IRDLOperation):
    name = "test.region_op"

    irdl_options = [AttrSizedRegionSegments()]

    region: Region = region_def()
    opt_region: OptRegion = opt_region_def()
    var_region: VarRegion = var_region_def()


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

    operand: Operand = operand_def()
    opt_operand: OptOperand = opt_operand_def()
    var_operand: VarOperand = var_operand_def()


def test_operand_accessors():
    """Test accessors for operands."""
    operand1 = TestSSAValue(i32)
    operand2 = TestSSAValue(i32)
    operand3 = TestSSAValue(i32)
    operand4 = TestSSAValue(i32)

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

    result: OpResult = result_def()
    opt_result: OptOpResult = opt_result_def()
    var_result: VarOpResult = var_result_def()


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

    attr: StringAttr = attr_def(StringAttr)
    opt_attr: StringAttr | None = opt_attr_def(StringAttr)


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


@irdl_op_definition
class PropertyOp(IRDLOperation):
    name = "test.attribute_op"

    attr: StringAttr = prop_def(StringAttr)
    opt_attr: StringAttr | None = opt_prop_def(StringAttr)


def test_property_accessors():
    """Test accessors for properties."""

    op = PropertyOp.create(
        properties={"attr": StringAttr("test"), "opt_attr": StringAttr("opt_test")}
    )
    assert op.attr is op.properties["attr"]
    assert op.opt_attr is op.properties["opt_attr"]

    op = PropertyOp.create(properties={"attr": StringAttr("test")})
    assert op.opt_attr is None


def test_property_setters():
    """Test setters for properties."""

    op = PropertyOp.create(properties={"attr": StringAttr("test")})

    op.attr = StringAttr("new_test")
    assert op.attr.data == "new_test"

    op.opt_attr = StringAttr("new_opt_test")
    assert op.opt_attr.data == "new_opt_test"

    op.opt_attr = None
    assert op.opt_attr is None


################################################################################
#                            Renamed attributes                                #
################################################################################

# These tests check that operations that have attributes with names that differ
# from the attribute accessors can be defined.


@irdl_op_definition
class RenamedAttributeOp(IRDLOperation):
    """
    An operation that has attributes with different names than the attribute
    accessors.
    """

    name = "test.renamed_attribute_op"

    accessor: StringAttr = attr_def(StringAttr, attr_name="attr_name")
    opt_accessor: StringAttr | None = opt_attr_def(
        StringAttr, attr_name="opt_attr_name"
    )


def test_renamed_attributes_verify():
    op = RenamedAttributeOp.create(
        attributes={
            "attr_name": StringAttr("test"),
            "opt_attr_name": StringAttr("test_opt"),
        }
    )
    op.verify()

    op = RenamedAttributeOp.create(
        attributes={
            "accessor": StringAttr("test"),
        }
    )
    with pytest.raises(VerifyException, match="attribute attr_name expected"):
        op.verify()

    op = RenamedAttributeOp.create(
        attributes={
            "attr_name": StringAttr("test"),
            "opt_attr_name": i32,
        }
    )
    with pytest.raises(VerifyException, match="i32 should be of base attribute string"):
        op.verify()


def test_renamed_attributes_accessors():
    op = RenamedAttributeOp.create(
        attributes={
            "attr_name": StringAttr("test"),
            "opt_attr_name": StringAttr("test_opt"),
        }
    )

    assert op.accessor is op.attributes["attr_name"]
    assert op.opt_accessor is op.attributes["opt_attr_name"]


@irdl_op_definition
class RenamedPropertyOp(IRDLOperation):
    """
    An operation that has properties with different names than the properties
    accessors.
    """

    name = "test.renamed_property_op"

    accessor: StringAttr = prop_def(StringAttr, prop_name="prop_name")
    opt_accessor: StringAttr | None = opt_prop_def(
        StringAttr, prop_name="opt_prop_name"
    )


def test_renamed_properties_verify():
    op = RenamedPropertyOp.create(
        properties={
            "prop_name": StringAttr("test"),
            "opt_prop_name": StringAttr("test_opt"),
        }
    )
    op.verify()

    op = RenamedPropertyOp.create(
        properties={
            "accessor": StringAttr("test"),
        }
    )
    with pytest.raises(VerifyException, match="property prop_name expected"):
        op.verify()

    op = RenamedPropertyOp.create(
        properties={
            "prop_name": StringAttr("test"),
            "opt_prop_name": i32,
        }
    )
    with pytest.raises(VerifyException, match="i32 should be of base attribute string"):
        op.verify()


def test_renamed_properties_accessors():
    op = RenamedPropertyOp.create(
        properties={
            "prop_name": StringAttr("test"),
            "opt_prop_name": StringAttr("test_opt"),
        }
    )

    assert op.accessor is op.properties["prop_name"]
    assert op.opt_accessor is op.properties["opt_prop_name"]


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

    attr: _Attr = attr_def(_Attr)
    operand: Operand = operand_def(_Operand)
    result: OpResult = result_def(_Result)


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


class OtherParentOp(IRDLOperation):
    other_attr = attr_def(Attribute)


@irdl_op_definition
class OtherStringFooOp(GenericOp[StringAttr, FooType, FooType], OtherParentOp):
    name = "test.string_specialized"


def test_multiple_inheritance_op():
    """Test generic operation."""
    FooOperand = TestSSAValue(TestType("foo"))
    FooResultType = TestType("foo")

    op = OtherStringFooOp(
        attributes={"attr": StringAttr("test"), "other_attr": StringAttr("test")},
        operands=[FooOperand],
        result_types=[FooResultType],
    )
    op.verify()

    op_attr_fail = OtherStringFooOp(
        attributes={"attr": IntAttr(1), "other_attr": StringAttr("test")},
        operands=[FooOperand],
        result_types=[FooResultType],
    )
    with pytest.raises(DiagnosticException):
        op_attr_fail.verify()

    op = OtherStringFooOp(
        attributes={"attr": StringAttr("test")},
        operands=[FooOperand],
        result_types=[FooResultType],
    )
    with pytest.raises(DiagnosticException):
        op_attr_fail.verify()
