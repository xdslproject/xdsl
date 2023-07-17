"""
Test the definition of attributes and their constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Annotated, Any, Generic, TypeAlias, TypeVar, cast

import pytest

from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType, Signedness
from xdsl.ir import Attribute, Data, ParametrizedAttribute
from xdsl.irdl import (
    AnyAttr,
    AttrConstraint,
    BaseAttr,
    ConstraintVar,
    GenericData,
    ParamAttrDef,
    ParameterDef,
    irdl_attr_definition,
    irdl_to_attr_constraint,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import PyRDLAttrDefinitionError, VerifyException

################################################################################
# Data attributes
################################################################################


@irdl_attr_definition
class BoolData(Data[bool]):
    """An attribute holding a boolean value."""

    name = "bool"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> bool:
        if parser.parse_optional_keyword("True"):
            return True
        if parser.parse_optional_keyword("False"):
            return False
        parser.raise_error("Expected True or False literal")

    def print_parameter(self, printer: Printer):
        printer.print_string(str(self.data))


@irdl_attr_definition
class IntData(Data[int]):
    """An attribute holding an integer value."""

    name = "int"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> int:
        return parser.parse_integer()

    def print_parameter(self, printer: Printer):
        printer.print_string(str(self.data))


@irdl_attr_definition
class StringData(Data[str]):
    """An attribute holding a string value."""

    name = "str"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        return parser.parse_str_literal()

    def print_parameter(self, printer: Printer):
        printer.print_string(self.data)


def test_simple_data():
    """Test that the definition of a data with a class parameter."""
    b = BoolData(True)
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(b)
    assert stream.getvalue() == "#bool<True>"


@irdl_attr_definition
class IntListData(Data[list[int]]):
    """
    An attribute holding a list of integers.
    """

    name = "int_list"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> list[int]:
        raise NotImplementedError()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string("[")
        printer.print_list(self.data, lambda x: printer.print_string(str(x)))
        printer.print_string("]")


def test_non_class_data():
    """Test the definition of a Data with a non-class parameter."""
    attr = IntListData([0, 1, 42])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#int_list<[0, 1, 42]>"


################################################################################
# IntegerAttr
################################################################################


def test_signed_integer_attr():
    """Test the verification of a signed integer attribute."""
    with pytest.raises(VerifyException):
        IntegerAttr(1 << 31, IntegerType(32, Signedness.SIGNED))

    with pytest.raises(VerifyException):
        IntegerAttr(-(1 << 31) - 1, IntegerType(32, Signedness.SIGNED))

    IntegerAttr((1 << 31) - 1, IntegerType(32, Signedness.SIGNED))
    IntegerAttr(-(1 << 31), IntegerType(32, Signedness.SIGNED))


def test_unsigned_integer_attr():
    """Test the verification of a unsigned integer attribute."""
    with pytest.raises(VerifyException):
        IntegerAttr(1 << 32, IntegerType(32, Signedness.UNSIGNED))

    with pytest.raises(VerifyException):
        IntegerAttr(-1, IntegerType(32, Signedness.UNSIGNED))

    IntegerAttr((1 << 32) - 1, IntegerType(32, Signedness.UNSIGNED))


def test_signless_integer_attr():
    """Test the verification of a signless integer attribute."""
    with pytest.raises(VerifyException):
        IntegerAttr((1 << 32) + 1, IntegerType(32, Signedness.SIGNLESS))

    with pytest.raises(VerifyException):
        IntegerAttr(-(1 << 32) - 1, IntegerType(32, Signedness.SIGNLESS))

    IntegerAttr(1 << 32, IntegerType(32, Signedness.SIGNLESS))
    IntegerAttr(-(1 << 32), IntegerType(32, Signedness.SIGNLESS))


################################################################################
# PyRDL Base constraints
################################################################################


@irdl_attr_definition
class BoolWrapperAttr(ParametrizedAttribute):
    name = "bool_wrapper"

    param: ParameterDef[BoolData]


def test_bose_constraint():
    """Test the verifier of a base attribute type constraint."""
    attr = BoolWrapperAttr([BoolData(True)])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#bool_wrapper<#bool<True>>"


def test_base_constraint_fail():
    """Test the verifier of a union constraint."""
    with pytest.raises(Exception) as e:
        BoolWrapperAttr([StringData("foo")])
    assert e.value.args[0] == "#str<foo> should be of base attribute bool"


################################################################################
# PyRDL union constraints
################################################################################


@irdl_attr_definition
class BoolOrIntParamAttr(ParametrizedAttribute):
    name = "bool_or_int"

    param: ParameterDef[BoolData | IntData]


def test_union_constraint_left():
    """Test the verifier of a union constraint."""
    attr = BoolOrIntParamAttr([BoolData(True)])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#bool_or_int<#bool<True>>"


def test_union_constraint_right():
    """Test the verifier of a union constraint."""
    attr = BoolOrIntParamAttr([IntData(42)])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#bool_or_int<#int<42>>"


def test_union_constraint_fail():
    """Test the verifier of a union constraint."""
    with pytest.raises(Exception) as e:
        BoolOrIntParamAttr([StringData("foo")])
    assert e.value.args[0] == "Unexpected attribute #str<foo>"


################################################################################
# PyRDL Annotated constraints
################################################################################


class PositiveIntConstr(AttrConstraint):
    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, IntData):
            raise VerifyException(
                f"Expected {IntData.name} attribute, but got {attr.name}."
            )
        if attr.data <= 0:
            raise VerifyException(f"Expected positive integer, got {attr.data}.")


@irdl_attr_definition
class PositiveIntAttr(ParametrizedAttribute):
    name = "positive_int"

    param: ParameterDef[Annotated[IntData, PositiveIntConstr()]]


def test_annotated_constraint():
    """Test the verifier of an annotated constraint."""
    attr = PositiveIntAttr([IntData(42)])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#positive_int<#int<42>>"


def test_annotated_constraint_fail():
    """Test that the verifier of an annotated constraint can fail."""
    with pytest.raises(Exception) as e:
        PositiveIntAttr([IntData(-42)])
    assert e.value.args[0] == "Expected positive integer, got -42."


################################################################################
# PyRDL Generic constraints
################################################################################

_T = TypeVar("_T", bound=BoolData | IntData)


@irdl_attr_definition
class ParamWrapperAttr(Generic[_T], ParametrizedAttribute):
    name = "int_or_bool_generic"

    param: ParameterDef[_T]


def test_typevar_attribute_int():
    """Test the verifier of a generic attribute."""
    attr = ParamWrapperAttr[IntData]([IntData(42)])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#int_or_bool_generic<#int<42>>"


def test_typevar_attribute_bool():
    """Test the verifier of a generic attribute."""
    attr = ParamWrapperAttr[BoolData]([BoolData(True)])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#int_or_bool_generic<#bool<True>>"


def test_typevar_attribute_fail():
    """Test that the verifier of an generic attribute can fail."""
    with pytest.raises(Exception) as e:
        ParamWrapperAttr([StringData("foo")])
    assert e.value.args[0] == "Unexpected attribute #str<foo>"


@irdl_attr_definition
class ParamConstrAttr(ParametrizedAttribute):
    name = "param_constr"

    param: ParameterDef[ParamWrapperAttr[IntData]]


def test_param_attr_constraint():
    """Test the verifier of an attribute with a parametric constraint."""
    attr = ParamConstrAttr([ParamWrapperAttr([IntData(42)])])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#param_constr<#int_or_bool_generic<#int<42>>>"


def test_param_attr_constraint_fail():
    """
    Test that the verifier of an attribute with
    a parametric constraint can fail.
    """
    with pytest.raises(Exception) as e:
        ParamConstrAttr([ParamWrapperAttr([BoolData(True)])])
    assert e.value.args[0] == "#bool<True> should be of base attribute int"


_U = TypeVar("_U", bound=IntData)


@irdl_attr_definition
class NestedParamWrapperAttr(Generic[_U], ParametrizedAttribute):
    name = "nested_param_wrapper"

    param: ParameterDef[ParamWrapperAttr[_U]]


def test_nested_generic_constraint():
    """
    Test the verifier of an attribute with a generic
    constraint used in a parametric constraint.
    """
    attr = NestedParamWrapperAttr[IntData]([ParamWrapperAttr([IntData(42)])])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#nested_param_wrapper<#int_or_bool_generic<#int<42>>>"


def test_nested_generic_constraint_fail():
    """
    Test that the verifier of an attribute with
    a parametric constraint can fail.
    """
    with pytest.raises(Exception) as e:
        NestedParamWrapperAttr([ParamWrapperAttr([BoolData(True)])])
    assert e.value.args[0] == "#bool<True> should be of base attribute int"


@irdl_attr_definition
class NestedParamConstrAttr(ParametrizedAttribute):
    name = "nested_param_constr"

    param: ParameterDef[NestedParamWrapperAttr[Annotated[IntData, PositiveIntConstr()]]]


def test_nested_param_attr_constraint():
    """
    Test the verifier of a nested parametric constraint.
    """
    attr = NestedParamConstrAttr(
        [NestedParamWrapperAttr([ParamWrapperAttr([IntData(42)])])]
    )
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert (
        stream.getvalue()
        == "#nested_param_constr<#nested_param_wrapper<#int_or_bool_generic<#int<42>>>>"
    )


def test_nested_param_attr_constraint_fail():
    """
    Test that the verifier of a nested parametric constraint can fail.
    """
    with pytest.raises(Exception) as e:
        NestedParamConstrAttr(
            [NestedParamWrapperAttr([ParamWrapperAttr([IntData(-42)])])]
        )
    assert e.value.args[0] == "Expected positive integer, got -42."


################################################################################
# GenericData definition
################################################################################

_MissingGenericDataData = TypeVar("_MissingGenericDataData")


@irdl_attr_definition
class MissingGenericDataData(Data[_MissingGenericDataData]):
    name = "missing_genericdata"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> _MissingGenericDataData:
        raise NotImplementedError()

    def print_parameter(self, printer: Printer) -> None:
        raise NotImplementedError()

    def verify(self) -> None:
        return


class MissingGenericDataDataWrapper(ParametrizedAttribute):
    name = "missing_genericdata_wrapper"

    param: ParameterDef[MissingGenericDataData[int]]


def test_data_with_generic_missing_generic_data_failure():
    """
    Test error message when a generic data is used in constraints
    without implementing GenericData.
    """
    with pytest.raises(Exception) as e:
        irdl_attr_definition(MissingGenericDataDataWrapper)
    assert e.value.args[0] == (
        "Generic `Data` type 'missing_genericdata' cannot be converted to "
        "an attribute constraint. Consider making it inherit from "
        "`GenericData` instead of `Data`."
    )


A = TypeVar("A", bound=Attribute)


@dataclass
class DataListAttr(AttrConstraint):
    """
    A constraint that enforces that the elements of a ListData all respect
    a constraint.
    """

    elem_constr: AttrConstraint

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        attr = cast(ListData[Any], attr)
        for e in attr.data:
            self.elem_constr.verify(e, constraint_vars)


@irdl_attr_definition
class ListData(GenericData[list[A]]):
    name = "list"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> list[A]:
        raise NotImplementedError()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string("[")
        printer.print_list(self.data, printer.print_attribute)
        printer.print_string("]")

    @staticmethod
    def generic_constraint_coercion(args: tuple[Any]) -> AttrConstraint:
        assert len(args) == 1
        return DataListAttr(irdl_to_attr_constraint(args[0]))

    @staticmethod
    def from_list(data: list[A]) -> ListData[A]:
        return ListData(data)

    def verify(self) -> None:
        for idx, val in enumerate(self.data):
            if not isinstance(val, Attribute):
                raise VerifyException(
                    f"{self.name} data expects attribute list, but element "
                    f"{idx} is of type {type(val)}."
                )


AnyListData: TypeAlias = ListData[Attribute]


class Test_generic_data_verifier:
    def test_generic_data_verifier(self):
        """
        Test that a GenericData can be created.
        """
        attr = ListData([BoolData(True), ListData([BoolData(False)])])
        stream = StringIO()
        p = Printer(stream=stream)
        p.print_attribute(attr)
        assert stream.getvalue() == "#list<[#bool<True>, #list<[#bool<False>]>]>"


@irdl_attr_definition
class ListDataWrapper(ParametrizedAttribute):
    name = "list_wrapper"

    val: ParameterDef[ListData[BoolData]]


def test_generic_data_wrapper_verifier():
    """
    Test that a GenericData used in constraints pass the verifier when correct.
    """
    attr = ListDataWrapper([ListData([BoolData(True), BoolData(False)])])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#list_wrapper<#list<[#bool<True>, #bool<False>]>>"


def test_generic_data_wrapper_verifier_failure():
    """
    Test that a GenericData used in constraints fails
    the verifier when constraints are not satisfied.
    """
    with pytest.raises(VerifyException) as e:
        ListDataWrapper([ListData([BoolData(True), ListData([BoolData(False)])])])
    assert e.value.args[0] == "#list<[#bool<False>]> should be of base attribute bool"


@irdl_attr_definition
class ListDataNoGenericsWrapper(ParametrizedAttribute):
    name = "list_no_generics_wrapper"

    val: ParameterDef[AnyListData]


def test_generic_data_no_generics_wrapper_verifier():
    """
    Test that GenericType can be used in constraints without a parameter.
    """
    attr = ListDataNoGenericsWrapper(
        [ListData([BoolData(True), ListData([BoolData(False)])])]
    )
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert (
        stream.getvalue()
        == "#list_no_generics_wrapper<#list<[#bool<True>, #list<[#bool<False>]>]>>"
    )


################################################################################
# Parametrized attribute definition
################################################################################


@irdl_attr_definition
class ParamAttrDefAttr(ParametrizedAttribute):
    name = "test.param_attr_def_attr"

    arg1: ParameterDef[Attribute]
    arg2: ParameterDef[BoolData]

    # Check that we can define methods in attribute definition
    def test(self):
        pass


def test_irdl_definition():
    """Test that we can get the IRDL definition of a parametrized attribute."""

    assert ParamAttrDefAttr.irdl_definition == ParamAttrDef(
        "test.param_attr_def_attr", [("arg1", AnyAttr()), ("arg2", BaseAttr(BoolData))]
    )


class InvalidTypedFieldTestAttr(ParametrizedAttribute):
    name = "test.invalid_typed_field"

    x: int


def test_invalid_typed_field():
    """Check that typed fields are not allowed."""
    with pytest.raises(PyRDLAttrDefinitionError):
        irdl_attr_definition(InvalidTypedFieldTestAttr)


class InvalidUntypedFieldTestAttr(ParametrizedAttribute):
    name = "test.invalid_untyped_field"

    x = 2


def test_invalid_field():
    """Check that untyped fields are not allowed."""
    with pytest.raises(PyRDLAttrDefinitionError):
        irdl_attr_definition(InvalidUntypedFieldTestAttr)


@irdl_attr_definition
class OveriddenInitAttr(ParametrizedAttribute):
    name = "test.overidden_init"

    param: ParameterDef[Attribute]

    def __init__(self, param: int | str):
        match param:
            case int():
                super().__init__([IntData(param)])
            case str():
                super().__init__([StringData(param)])


def test_generic_constructor():
    """Test the generic constructor of a ParametrizedAttribute."""

    param = IntData(42)
    attr = OveriddenInitAttr.new([param])

    assert isinstance(attr, OveriddenInitAttr)
    assert attr.param == param


def test_custom_constructor():
    """Test the use of custom constructors in ParametrizedAttribute."""

    assert OveriddenInitAttr.new([IntData(42)]) == OveriddenInitAttr(42)
    assert OveriddenInitAttr.new([StringData("17")]) == OveriddenInitAttr("17")


################################################################################
# ConstraintVar
################################################################################


@irdl_attr_definition
class ConstraintVarAttr(ParametrizedAttribute):
    name = "test.constraint_var"

    T = Annotated[IntegerType, ConstraintVar("T")]

    param1: ParameterDef[IntegerAttr[T]]
    param2: ParameterDef[IntegerAttr[T]]


def test_constraint_var():
    """Test that ConstraintVar can be used in attributes."""
    ConstraintVarAttr.new([IntegerAttr(42, 32), IntegerAttr(17, 32)])
    ConstraintVarAttr.new([IntegerAttr(42, 64), IntegerAttr(17, 64)])


def test_constraint_var_fail_non_equal():
    """Test that constraint variables must be equal."""
    with pytest.raises(VerifyException):
        ConstraintVarAttr.new([IntegerAttr(42, 32), IntegerAttr(42, 64)])
    with pytest.raises(VerifyException):
        ConstraintVarAttr.new([IntegerAttr(42, 64), IntegerAttr(42, 32)])


def test_constraint_var_fail_not_satisfy_constraint():
    """Test that constraint variables must satisfy the underlying constraint."""
    with pytest.raises(VerifyException):
        ConstraintVarAttr.new(
            [IntegerAttr(42, IndexType()), IntegerAttr(17, IndexType())]
        )
