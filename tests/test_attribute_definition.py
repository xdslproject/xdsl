"""
Test the definition of attributes and their constraints.
"""

from __future__ import annotations
from dataclasses import dataclass
from io import StringIO
from typing import Any, TypeVar, cast, Annotated, Generic, TypeAlias

import pytest

from xdsl.ir import Attribute, Data, ParametrizedAttribute
from xdsl.irdl import (
    AttrConstraint,
    GenericData,
    ParameterDef,
    irdl_attr_definition,
    irdl_to_attr_constraint,
    AnyAttr,
    BaseAttr,
    ParamAttrDef,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import PyRDLAttrDefinitionError, VerifyException

################################################################################
# Data attributes
################################################################################


@irdl_attr_definition
class BoolData(Data[bool]):
    """An attribute holding a boolean value."""

    name = "bool"

    @staticmethod
    def parse_parameter(parser: Parser) -> bool:
        val = parser.tokenizer.next_token_of_pattern("(True|False)")
        if val is None or val.text not in ("True", "False"):
            parser.raise_error("Expected True or False literal")
        if val.text == "True":
            return True
        return False

    def print_parameter(self, printer: Printer):
        printer.print_string(str(self.data))


@irdl_attr_definition
class IntData(Data[int]):
    """An attribute holding an integer value."""

    name = "int"

    @staticmethod
    def parse_parameter(parser: Parser) -> int:
        return parser.parse_integer()

    def print_parameter(self, printer: Printer):
        printer.print_string(str(self.data))


@irdl_attr_definition
class StringData(Data[str]):
    """An attribute holding a string value."""

    name = "str"

    @staticmethod
    def parse_parameter(parser: Parser) -> str:
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


def test_simple_data_verifier_failure():
    """
    Test that the verifier of a data with a class parameter fails when given
    a parameter of the wrong type.
    """
    with pytest.raises(VerifyException) as e:
        BoolData(2)  # type: ignore
    assert e.value.args[0] == (
        "bool data attribute expected type " "<class 'bool'>, but <class 'int'> given."
    )


class IntListMissingVerifierData(Data[list[int]]):
    """
    An attribute holding a list of integers.
    The definition should fail, since no verifier is provided, and the Data
    type parameter is not a class.
    """

    name = "missing_verifier_data"

    @staticmethod
    def parse_parameter(parser: Parser) -> list[int]:
        raise NotImplementedError()

    def print_parameter(self, printer: Printer) -> None:
        raise NotImplementedError()


def test_data_with_non_class_param_missing_verifier_failure():
    """
    Test that a non-class Data parameter requires the definition of a verifier.
    """
    with pytest.raises(Exception) as e:
        irdl_attr_definition(IntListMissingVerifierData)

    # Python 3.10 and 3.11 have different error messages
    assert e.value.args[0] in [
        "In IntListMissingVerifierData definition: "
        'Cannot infer "verify" method. Type parameter of Data has type GenericAlias.',
        "In IntListMissingVerifierData definition: "
        'Cannot infer "verify" method. Type parameter of Data is not a class.',
    ]


@irdl_attr_definition
class IntListData(Data[list[int]]):
    """
    An attribute holding a list of integers.
    """

    name = "int_list"

    @staticmethod
    def parse_parameter(parser: Parser) -> list[int]:
        raise NotImplementedError()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string("[")
        printer.print_list(self.data, lambda x: printer.print_string(str(x)))
        printer.print_string("]")

    def verify(self) -> None:
        if not isinstance(self.data, list):
            raise VerifyException("int_list data should hold a list.")
        for elem in self.data:
            if not isinstance(elem, int):
                raise VerifyException("int_list list elements should be integers.")


def test_non_class_data():
    """Test the definition of a Data with a non-class parameter."""
    attr = IntListData([0, 1, 42])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#int_list<[0, 1, 42]>"


def test_simple_data_constructor_failure():
    """
    Test that the verifier of a Data with a non-class parameter fails when
    given wrong arguments.
    """
    with pytest.raises(VerifyException) as e:
        IntListData([0, 1, 42, ""])  # type: ignore
    assert e.value.args[0] == "int_list list elements should be integers."


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
    def verify(self, attr: Attribute) -> None:
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

    @staticmethod
    def parse_parameter(parser: Parser) -> _MissingGenericDataData:
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

    def verify(self, attr: Attribute) -> None:
        attr = cast(ListData[Any], attr)
        for e in attr.data:
            self.elem_constr.verify(e)


@irdl_attr_definition
class ListData(GenericData[list[A]]):
    name = "list"

    @staticmethod
    def parse_parameter(parser: Parser) -> list[A]:
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
        if not isinstance(self.data, list):
            raise VerifyException(
                f"Wrong type given to attribute {self.name}: got"
                f" {type(self.data)}, but expected list of"
                " attributes."
            )
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

    def test_generic_data_verifier_fail(self):
        """
        Test that a GenericData verifier fails when given wrong parameters.
        """
        with pytest.raises(VerifyException) as e:
            ListData([0])  # type: ignore
        assert e.value.args[0] == (
            "list data expects attribute list, but"
            " element 0 is of type <class 'int'>."
        )

    def test_generic_data_verifier_fail_II(self):
        """
        Test that a GenericData verifier fails when given wrong parameters.
        """
        with pytest.raises(VerifyException) as e:
            ListData((0))  # type: ignore
        assert e.value.args[0] == (
            "Wrong type given to attribute list: "
            "got <class 'int'>, but expected list of attributes."
        )


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
        if isinstance(param, int):
            super().__init__([IntData(param)])
        elif isinstance(param, str):
            super().__init__([StringData(param)])
        else:
            raise TypeError(
                "Expected `int` or `str` type in " "OveriddenInitAttr constructor"
            )


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
