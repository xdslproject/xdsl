from __future__ import annotations

from typing import Annotated, ClassVar, Generic

import pytest
from typing_extensions import TypeVar

from xdsl.context import Context
from xdsl.dialects.builtin import (
    BoolAttr,
    DenseArrayBase,
    IndexType,
    IntAttr,
    IntegerType,
    StringAttr,
    i32,
    i64,
)
from xdsl.dialects.test import TestType
from xdsl.ir import Block, Region
from xdsl.irdl import (
    AnyAttr,
    AnyInt,
    AnyOf,
    AttributeDef,
    AttrSizedOperandSegments,
    AttrSizedRegionSegments,
    AttrSizedResultSegments,
    BaseAttr,
    ConstraintVar,
    EqAttrConstraint,
    IntVarConstraint,
    IRDLOperation,
    OpDef,
    OperandDef,
    PropertyDef,
    RangeOf,
    RangeVarConstraint,
    RegionDef,
    ResultDef,
    VarConstraint,
    attr_def,
    base,
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
    traits_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.traits import NoTerminator
from xdsl.utils.exceptions import (
    DiagnosticException,
    PyRDLOpDefinitionError,
    VerifyException,
)
from xdsl.utils.test_value import create_ssa_value

################################################################################
#                              IRDL definition                                 #
################################################################################


@irdl_op_definition
class OpDefTestOp(IRDLOperation):
    name = "test.op_def_test"

    irdl_options = [AttrSizedOperandSegments()]

    operand = operand_def()
    result = result_def()
    prop = prop_def()
    attr = attr_def()
    region = region_def()

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


class InvalidIRDLOpts(IRDLOperation):
    name = "test.invalid_field"
    irdl_options = [42]


def test_invalid_irdl_options():
    """Check that irdl_options only contains IRDLOptions"""
    with pytest.raises(
        PyRDLOpDefinitionError,
        match="All values in irdl_options should inherit IRDLOption",
    ):
        irdl_op_definition(InvalidIRDLOpts)


################################################################################
#                                  Verifiers                                   #
################################################################################


@irdl_op_definition
class AttrOp(IRDLOperation):
    name = "test.two_var_result_op"
    attr = attr_def(StringAttr)


def test_attr_verify():
    op = AttrOp.create(attributes={"attr": IntAttr(1)})
    with pytest.raises(
        VerifyException, match="#builtin.int<1> should be of base attribute string"
    ):
        op.verify()


with pytest.deprecated_call():
    # TODO: remove this test once the Annotated API is deprecated
    @irdl_op_definition
    class ConstraintVarOp(IRDLOperation):
        name = "test.constraint_var_op"

        T = Annotated[IntegerType | IndexType, ConstraintVar("T")]

        operand = operand_def(T)
        result = result_def(T)
        attribute = attr_def(T)


def test_constraint_var():
    i32_operand = create_ssa_value(i32)
    index_operand = create_ssa_value(IndexType())
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
    i32_operand = create_ssa_value(i32)
    index_operand = create_ssa_value(IndexType())

    # Fail because of operand
    op = ConstraintVarOp.create(
        operands=[index_operand], result_types=[i32], attributes={"attribute": i32}
    )
    with pytest.raises(
        DiagnosticException,
        match="Operation does not verify: result 'result' at position 0 does not verify",
    ):
        op.verify()

    # Fail because of result
    op2 = ConstraintVarOp.create(
        operands=[i32_operand],
        result_types=[IndexType()],
        attributes={"attribute": i32},
    )
    with pytest.raises(
        DiagnosticException,
        match="Operation does not verify: result 'result' at position 0 does not verify",
    ):
        op2.verify()

    # Fail because of attribute
    op3 = ConstraintVarOp.create(
        operands=[i32_operand],
        result_types=[i32],
        attributes={"attribute": IndexType()},
    )
    with pytest.raises(
        DiagnosticException,
        match="Operation does not verify: attribute i32 expected from variable 'T', but got index",
    ):
        op3.verify()


def test_constraint_var_fail_not_satisfy_constraint():
    """Check that all uses of a constraint variable are satisfying the constraint."""
    test_operand = create_ssa_value(TestType("foo"))
    op = ConstraintVarOp.create(
        operands=[test_operand],
        result_types=[TestType("foo")],
        attributes={"attribute": TestType("foo")},
    )
    with pytest.raises(
        DiagnosticException,
        match="Operation does not verify: operand 'operand' at position 0 does not verify",
    ):
        op.verify()


@irdl_op_definition
class GenericConstraintVarOp(IRDLOperation):
    name = "test.constraint_var_op"

    T: ClassVar = VarConstraint("T", base(IntegerType) | base(IndexType))

    operand = operand_def(T)
    result = result_def(T)
    attribute = attr_def(T)


def test_generic_constraint_var():
    i32_operand = create_ssa_value(i32)
    index_operand = create_ssa_value(IndexType())
    op = GenericConstraintVarOp.create(
        operands=[i32_operand], result_types=[i32], attributes={"attribute": i32}
    )
    op.verify()

    op2 = GenericConstraintVarOp.create(
        operands=[index_operand],
        result_types=[IndexType()],
        attributes={"attribute": IndexType()},
    )
    op2.verify()


def test_generic_constraint_var_fail_non_equal():
    """Check that all uses of a constraint variable are of the same attribute."""
    i32_operand = create_ssa_value(i32)
    index_operand = create_ssa_value(IndexType())

    # Fail because of operand
    op = GenericConstraintVarOp.create(
        operands=[index_operand], result_types=[i32], attributes={"attribute": i32}
    )
    with pytest.raises(
        DiagnosticException,
        match="Operation does not verify: result 'result' at position 0 does not verify",
    ):
        op.verify()

    # Fail because of result
    op2 = GenericConstraintVarOp.create(
        operands=[i32_operand],
        result_types=[IndexType()],
        attributes={"attribute": i32},
    )
    with pytest.raises(
        DiagnosticException,
        match="Operation does not verify: result 'result' at position 0 does not verify",
    ):
        op2.verify()

    # Fail because of attribute
    op3 = GenericConstraintVarOp.create(
        operands=[i32_operand],
        result_types=[i32],
        attributes={"attribute": IndexType()},
    )
    with pytest.raises(
        DiagnosticException,
        match="Operation does not verify: attribute i32 expected from variable 'T', but got index",
    ):
        op3.verify()


def test_generic_constraint_var_fail_not_satisfy_constraint():
    """Check that all uses of a constraint variable are satisfying the constraint."""
    test_operand = create_ssa_value(TestType("foo"))
    op = GenericConstraintVarOp.create(
        operands=[test_operand],
        result_types=[TestType("foo")],
        attributes={"attribute": TestType("foo")},
    )
    with pytest.raises(
        DiagnosticException,
        match="Operation does not verify: operand 'operand' at position 0 does not verify",
    ):
        op.verify()


@irdl_op_definition
class ConstraintRangeVarOp(IRDLOperation):
    name = "test.constraint_range_var"

    operand = var_operand_def(RangeVarConstraint("T", RangeOf(AnyOf((i32, IndexType)))))
    result = var_result_def(RangeVarConstraint("T", RangeOf(AnyOf((i32, IndexType)))))


def test_range_var():
    i32_operand = create_ssa_value(i32)
    index_operand = create_ssa_value(IndexType())
    op = ConstraintRangeVarOp.create(operands=[], result_types=[])
    op.verify()
    op = ConstraintRangeVarOp.create(operands=[i32_operand], result_types=[i32])
    op.verify()
    op = ConstraintRangeVarOp.create(
        operands=[i32_operand, i32_operand], result_types=[i32, i32]
    )
    op.verify()

    op2 = ConstraintRangeVarOp.create(
        operands=[index_operand], result_types=[IndexType()]
    )
    op2.verify()


def test_range_var_fail_non_equal():
    """Check that all uses of a range variable are of the same attribute."""
    i32_operand = create_ssa_value(i32)
    index_operand = create_ssa_value(IndexType())

    op = ConstraintRangeVarOp.create(operands=[index_operand], result_types=[i32])
    with pytest.raises(
        VerifyException,
        match=r"attributes \('index',\) expected from range variable 'T', but got \('i32',\)",
    ):
        op.verify()

    op2 = ConstraintRangeVarOp.create(
        operands=[i32_operand], result_types=[IndexType()]
    )
    with pytest.raises(
        VerifyException,
        match=r"attributes \('i32',\) expected from range variable 'T', but got \('index',\)",
    ):
        op2.verify()

    op2 = ConstraintRangeVarOp.create(operands=[i32_operand], result_types=[i32, i32])
    with pytest.raises(
        VerifyException,
        match=r"attributes \('i32',\) expected from range variable 'T', but got \('i32', 'i32'\)",
    ):
        op2.verify()

    op2 = ConstraintRangeVarOp.create(operands=[i32_operand], result_types=[])
    with pytest.raises(
        VerifyException,
        match=r"attributes \('i32',\) expected from range variable 'T', but got \(\)",
    ):
        op2.verify()


def test_range_var_fail_not_satisfy_constraint():
    """Check that all uses of a range variable are satisfying the constraint."""
    test_operand = create_ssa_value(TestType("foo"))
    op = ConstraintRangeVarOp.create(
        operands=[test_operand], result_types=[TestType("foo")]
    )
    with pytest.raises(VerifyException, match='Unexpected attribute !test.type<"foo">'):
        op.verify()

    op = ConstraintRangeVarOp.create(
        operands=[test_operand, test_operand],
        result_types=[TestType("foo"), TestType("foo")],
    )
    with pytest.raises(VerifyException, match='Unexpected attribute !test.type<"foo">'):
        op.verify()


@irdl_op_definition
class SameLengthOp(IRDLOperation):
    """
    An operation that has the same number of results and operands.
    """

    name = "test.same_length"

    LENGTH: ClassVar = IntVarConstraint("length", AnyInt())
    operand = var_operand_def(RangeOf(AnyAttr()).of_length(LENGTH))
    result = var_result_def(RangeOf(AnyAttr()).of_length(LENGTH))


def test_same_length_op():
    operand1 = create_ssa_value(i32)
    operand2 = create_ssa_value(i32)

    op1 = SameLengthOp.create(operands=[operand1, operand2], result_types=[i32, i32])
    op1.verify()

    with pytest.raises(
        VerifyException,
        match="incorrect length for range variable:\ninteger 2 expected from int variable 'length', but got 1",
    ):
        op2 = SameLengthOp.create(operands=[operand1, operand2], result_types=[i32])
        op2.verify()

    with pytest.raises(
        VerifyException,
        match="incorrect length for range variable:\ninteger 1 expected from int variable 'length', but got 2",
    ):
        op3 = SameLengthOp.create(operands=[operand1], result_types=[i32, i32])
        op3.verify()

    with pytest.raises(
        VerifyException,
        match=(
            "result 'result' expected at position 0 does not verify:\n"
            "incorrect length for range variable:\ninteger 1 expected from int variable 'length', but got 0"
        ),
    ):
        op3 = SameLengthOp.create(operands=[operand1], result_types=[])
        op3.verify()


@irdl_op_definition
class WithoutPropOp(IRDLOperation):
    name = "test.op_without_prop"

    prop1 = prop_def()


# Check that an operation cannot accept properties that are not defined
def test_unknown_property():
    op = WithoutPropOp.create(properties={"prop1": i32, "prop2": i32})
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

    region = region_def()
    opt_region = opt_region_def()
    var_region = var_region_def()


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

    operand = operand_def()
    opt_operand = opt_operand_def()
    var_operand = var_operand_def()


def test_operand_accessors():
    """Test accessors for operands."""
    operand1 = create_ssa_value(i32)
    operand2 = create_ssa_value(i32)
    operand3 = create_ssa_value(i32)
    operand4 = create_ssa_value(i32)

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

    result = result_def()
    opt_result = opt_result_def()
    var_result = var_result_def()


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

    attr = attr_def(StringAttr)
    opt_attr = opt_attr_def(StringAttr)


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


def test_attribute_missing():
    """Test verification raises error when an attribute is missing"""

    op = AttributeOp.create(attributes={})

    with pytest.raises(
        VerifyException,
        match="attribute 'attr' expected in operation 'test.attribute_op'",
    ):
        op.verify()


@irdl_op_definition
class PropertyOp(IRDLOperation):
    name = "test.attribute_op"

    attr = prop_def(StringAttr)
    opt_attr = opt_prop_def(StringAttr)


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


def test_property_missing():
    """Test verification raises error when a property is missing"""

    op = PropertyOp.create(properties={})

    with pytest.raises(
        VerifyException,
        match="property 'attr' expected in operation 'test.attribute_op'",
    ):
        op.verify()


def test_undefined_property():
    """Test verification raises error when a property is missing"""

    op = PropertyOp.create(
        properties={"attr": StringAttr("test"), "unknown": StringAttr("test")}
    )

    with pytest.raises(
        VerifyException,
        match="property 'unknown' is not defined by the operation 'test.attribute_op'."
        + " Use the dictionary attribute to add arbitrary information to the operation.",
    ):
        op.verify()


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

    accessor = attr_def(StringAttr, attr_name="attr_name")
    opt_accessor = opt_attr_def(StringAttr, attr_name="opt_attr_name")


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
    with pytest.raises(
        VerifyException,
        match="attribute 'attr_name' expected in operation 'test.renamed_attribute_op'",
    ):
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

    accessor = prop_def(StringAttr, prop_name="prop_name")
    opt_accessor = opt_prop_def(StringAttr, prop_name="opt_prop_name")


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
    with pytest.raises(
        VerifyException,
        match="property 'prop_name' expected in operation 'test.renamed_property_op'",
    ):
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
_Operand = TypeVar("_Operand", bound=TestType)
_Result = TypeVar("_Result", bound=TestType)


class GenericOp(IRDLOperation, Generic[_Attr, _Operand, _Result]):
    name = "test.string_or_int_generic"

    attr: _Attr = attr_def(_Attr)
    operand = operand_def(_Operand)
    result = result_def(_Result)


@irdl_op_definition
class StringFooOp(GenericOp[StringAttr, FooType, FooType]):
    name = "test.string_specialized"


class Generic2Op(GenericOp[StringAttr, _Operand, FooType], Generic[_Operand]): ...


@irdl_op_definition
class StringFoo2Op(Generic2Op[FooType]):
    name = "test.string_specialized_2"


@pytest.mark.parametrize("cls", [StringFooOp, StringFoo2Op])
def test_generic_op(cls: type[StringFooOp | StringFoo2Op]):
    """Test generic operation."""
    FooOperand = create_ssa_value(TestType("foo"))
    BarOperand = create_ssa_value(TestType("bar"))
    FooResultType = TestType("foo")
    BarResultType = TestType("bar")

    op = cls(
        attributes={"attr": StringAttr("test")},
        operands=[FooOperand],
        result_types=[FooResultType],
    )
    op.verify()

    op_attr_fail = cls(
        attributes={"attr": IntAttr(1)},
        operands=[FooOperand],
        result_types=[FooResultType],
    )
    with pytest.raises(DiagnosticException):
        op_attr_fail.verify()

    op_operand_fail = cls(
        attributes={"attr": StringAttr("test")},
        operands=[BarOperand],
        result_types=[FooResultType],
    )
    with pytest.raises(DiagnosticException):
        op_operand_fail.verify()

    op_result_fail = cls(
        attributes={"attr": StringAttr("test")},
        operands=[FooOperand],
        result_types=[BarResultType],
    )
    with pytest.raises(DiagnosticException):
        op_result_fail.verify()


class OtherParentOp(IRDLOperation):
    other_attr = attr_def()


@irdl_op_definition
class OtherStringFooOp(GenericOp[StringAttr, FooType, FooType], OtherParentOp):
    name = "test.string_specialized"


def test_multiple_inheritance_op():
    """Test generic operation."""
    FooOperand = create_ssa_value(TestType("foo"))
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


@irdl_op_definition
class EntryArgsOp(IRDLOperation):
    name = "test.entry_args"
    body = opt_region_def(entry_args=RangeOf(EqAttrConstraint(i32)))

    traits = traits_def(NoTerminator())


def test_entry_args_op():
    op = EntryArgsOp.create()
    op.verify()

    op = EntryArgsOp.create(regions=[Region(Block(arg_types=[]))])
    op.verify()
    op = EntryArgsOp.create(regions=[Region(Block(arg_types=[i32]))])
    op.verify()
    op = EntryArgsOp.create(regions=[Region(Block(arg_types=[i32, i32]))])
    op.verify()

    op = EntryArgsOp.create(regions=[Region(Block(arg_types=[i64]))])
    with pytest.raises(
        VerifyException,
        match="""\
Operation does not verify: region #0 entry arguments do not verify:
.*Expected attribute i32 but got i64""",
    ):
        op.verify()

    op = EntryArgsOp.create(regions=[Region(Block(arg_types=[i64, i32]))])
    with pytest.raises(
        VerifyException,
        match="""\
Operation does not verify: region #0 entry arguments do not verify:
.*Expected attribute i32 but got i64""",
    ):
        op.verify()


class OptionlessMultipleVarOp(IRDLOperation):
    name = "test.multiple_var_op"

    optional = opt_operand_def()
    variadic = var_operand_def()


def test_no_multiple_var_option():
    with pytest.raises(
        PyRDLOpDefinitionError,
        match="Operation test.multiple_var_op defines more than two variadic operands, "
        "but do not define any of SameVariadicOperandSize or AttrSizedOperandSegments "
        "PyRDL options.",
    ):
        irdl_op_definition(OptionlessMultipleVarOp)


@irdl_op_definition
class DefaultOp(IRDLOperation):
    name = "test.default"

    prop = prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    opt_prop = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(True))

    attr = attr_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    opt_attr = opt_attr_def(BoolAttr, default_value=BoolAttr.from_bool(True))

    assembly_format = "(`prop` $prop^)? (`opt_prop` $opt_prop^)? (`attr` $attr^)? (`opt_attr` $opt_attr^)? attr-dict"


def test_default_accessors():
    ctx = Context()
    ctx.load_op(DefaultOp)

    parsed = Parser(ctx, "test.default").parse_operation()

    assert isinstance(parsed, DefaultOp)

    assert not parsed.prop.value.data

    assert parsed.properties.get("opt_prop") is None

    assert parsed.opt_prop.value.data

    assert not parsed.attr.value.data

    assert parsed.attributes.get("opt_attr") is None

    assert parsed.opt_attr.value.data


def test_generic_accessors():
    ctx = Context()
    ctx.load_op(DefaultOp)

    parsed = Parser(
        ctx, '"test.default"() <{ "prop" = false }> {"attr" = false} : () -> ()'
    ).parse_operation()

    assert isinstance(parsed, DefaultOp)

    assert not parsed.prop.value.data

    assert parsed.properties.get("opt_prop") is None

    assert parsed.opt_prop.value.data

    assert not parsed.attr.value.data

    assert parsed.attributes.get("opt_attr") is None

    assert parsed.opt_attr.value.data
