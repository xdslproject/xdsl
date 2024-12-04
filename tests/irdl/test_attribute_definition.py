"""
Test the definition of attributes and their constraints.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import auto
from io import StringIO
from typing import Annotated, Any, Generic, TypeAlias, TypeVar, cast

import pytest

from xdsl.dialects.builtin import (
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    NoneAttr,
    Signedness,
)
from xdsl.ir import (
    Attribute,
    BitEnumAttribute,
    Data,
    EnumAttribute,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    StrEnum,
    TypedAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AttrConstraint,
    BaseAttr,
    ConstraintContext,
    ConstraintVar,
    GenericData,
    MessageConstraint,
    ParamAttrDef,
    ParameterDef,
    irdl_attr_definition,
    irdl_to_attr_constraint,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import PyRDLAttrDefinitionError, VerifyException


def test_wrong_attribute_type():
    with pytest.raises(
        TypeError,
        match="Class AbstractAttribute should either be a subclass of 'Data' or 'ParametrizedAttribute'",
    ):

        @irdl_attr_definition
        class AbstractAttribute(Attribute):  # pyright: ignore[reportUnusedClass]
            name = "test.wrong"
            pass


################################################################################
# Data attributes
################################################################################


@irdl_attr_definition
class BoolData(Data[bool]):
    """An attribute holding a boolean value."""

    name = "test.bool"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> bool:
        with parser.in_angle_brackets():
            if parser.parse_optional_keyword("True"):
                return True
            if parser.parse_optional_keyword("False"):
                return False
            parser.raise_error("Expected True or False literal")

    def print_parameter(self, printer: Printer):
        with printer.in_angle_brackets():
            printer.print_string(str(self.data))


@irdl_attr_definition
class IntData(Data[int]):
    """An attribute holding an integer value."""

    name = "test.int"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> int:
        with parser.in_angle_brackets():
            return parser.parse_integer()

    def print_parameter(self, printer: Printer):
        with printer.in_angle_brackets():
            printer.print_string(str(self.data))


@irdl_attr_definition
class StringData(Data[str]):
    """An attribute holding a string value."""

    name = "test.str"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            return parser.parse_str_literal()

    def print_parameter(self, printer: Printer):
        with printer.in_angle_brackets():
            printer.print_string(self.data)


def test_simple_data():
    """Test that the definition of a data with a class parameter."""
    b = BoolData(True)
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(b)
    assert stream.getvalue() == "#test.bool<True>"


@irdl_attr_definition
class IntListData(Data[tuple[int, ...]]):
    """
    An attribute holding a list of integers.
    """

    name = "test.int_list"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> tuple[int]:
        raise NotImplementedError()

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("[")
            printer.print_list(self.data, lambda x: printer.print_string(str(x)))
            printer.print_string("]")


def test_non_class_data():
    """Test the definition of a Data with a non-class parameter."""
    attr = IntListData((0, 1, 42))
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#test.int_list<[0, 1, 42]>"


class TestEnum(StrEnum):
    Yes = auto()
    No = auto()
    Maybe = auto()


class TestNonIdentifierEnum(StrEnum):
    """
    The value defined by this StrEnum is not parsable as an identifier, because of the
    contained space.
    While valid as a StrEnum, it is thus invalid to use it in an EnumAttribute.
    """

    Spaced = "left right"


@irdl_attr_definition
class EnumData(EnumAttribute[TestEnum], SpacedOpaqueSyntaxAttribute):
    name = "test.enum"


def test_enum_attribute():
    """Test the definition of an EnumAttribute."""
    attr = EnumData(TestEnum.No)
    assert str(attr) == "#test<enum no>"


def test_indirect_enum_guard():
    EnumType = TypeVar("EnumType", bound=StrEnum)
    with pytest.raises(
        TypeError, match="Only direct inheritance from EnumAttribute is allowed."
    ):

        class IndirectEnumData(  # pyright: ignore[reportUnusedClass]
            EnumAttribute[EnumType]
        ):
            name = "test.indirect_enum"


def test_identifier_enum_guard():
    with pytest.raises(
        ValueError,
        match="All StrEnum values of an EnumAttribute must be parsable as an identifer.",
    ):

        class IndirectEnumData(  # pyright: ignore[reportUnusedClass]
            EnumAttribute[TestNonIdentifierEnum]
        ):
            name = "test.non_identifier_enum"


@irdl_attr_definition
class BitEnumData(BitEnumAttribute[TestEnum]):
    name = "test.bitenum"
    all_value = "all"
    none_value = "none"


@pytest.mark.parametrize(
    "input,output",
    [
        (None, "#test.bitenum<none>"),
        ([], "#test.bitenum<none>"),
        ([TestEnum.No], "#test.bitenum<no>"),
        ([TestEnum.Yes], "#test.bitenum<yes>"),
        ([TestEnum.No, TestEnum.Yes], "#test.bitenum<yes,no>"),
        ([TestEnum.No, TestEnum.Yes, TestEnum.Maybe], "#test.bitenum<all>"),
        ("all", "#test.bitenum<all>"),
        ("none", "#test.bitenum<none>"),
    ],
)
def test_bit_enum_attribute(input: Sequence[TestEnum] | str, output: str):
    attr = BitEnumData(input)
    assert str(attr) == output


def test_bit_enum_invalid_str():
    with pytest.raises(
        TypeError,
        match="expected string parameter to be one of none or all, got helloworld",
    ):
        BitEnumData("helloworld")


################################################################################
# Typed Attribute
################################################################################


def test_typed_attribute():
    with pytest.raises(
        PyRDLAttrDefinitionError,
        match="TypedAttribute TypedAttr should have a 'type' parameter.",
    ):

        @irdl_attr_definition
        class TypedAttr(  # pyright: ignore[reportUnusedClass]
            TypedAttribute
        ):
            name = "test.typed"


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
        IntegerAttr((1 << 32), IntegerType(32, Signedness.SIGNLESS))

    with pytest.raises(VerifyException):
        IntegerAttr(-(1 << 32) - 1, IntegerType(32, Signedness.SIGNLESS))

    IntegerAttr(1 << 32 - 1, IntegerType(32, Signedness.SIGNLESS))
    IntegerAttr(-(1 << 31), IntegerType(32, Signedness.SIGNLESS))


################################################################################
# PyRDL Base constraints
################################################################################


@irdl_attr_definition
class BoolWrapperAttr(ParametrizedAttribute):
    name = "test.bool_wrapper"

    param: ParameterDef[BoolData]


def test_bose_constraint():
    """Test the verifier of a base attribute type constraint."""
    attr = BoolWrapperAttr((BoolData(True),))
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#test.bool_wrapper<#test.bool<True>>"


def test_base_constraint_fail():
    """Test the verifier of a union constraint."""
    with pytest.raises(Exception) as e:
        BoolWrapperAttr((StringData("foo"),))
    assert e.value.args[0] == "#test.str<foo> should be of base attribute test.bool"


################################################################################
# PyRDL union constraints
################################################################################


@irdl_attr_definition
class BoolOrIntParamAttr(ParametrizedAttribute):
    name = "test.bool_or_int"

    param: ParameterDef[BoolData | IntData]


def test_union_constraint_left():
    """Test the verifier of a union constraint."""
    attr = BoolOrIntParamAttr((BoolData(True),))
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#test.bool_or_int<#test.bool<True>>"


def test_union_constraint_right():
    """Test the verifier of a union constraint."""
    attr = BoolOrIntParamAttr((IntData(42),))
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#test.bool_or_int<#test.int<42>>"


def test_union_constraint_fail():
    """Test the verifier of a union constraint."""
    with pytest.raises(Exception) as e:
        BoolOrIntParamAttr((StringData("foo"),))
    assert e.value.args[0] == "Unexpected attribute #test.str<foo>"


################################################################################
# PyRDL Annotated constraints
################################################################################


class PositiveIntConstr(AttrConstraint):
    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if not isinstance(attr, IntData):
            raise VerifyException(
                f"Expected {IntData.name} attribute, but got {attr.name}."
            )
        if attr.data <= 0:
            raise VerifyException(f"Expected positive integer, got {attr.data}.")


@irdl_attr_definition
class PositiveIntAttr(ParametrizedAttribute):
    name = "test.positive_int"

    param: ParameterDef[Annotated[IntData, PositiveIntConstr()]]


def test_annotated_constraint():
    """Test the verifier of an annotated constraint."""
    attr = PositiveIntAttr((IntData(42),))
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#test.positive_int<#test.int<42>>"


def test_annotated_constraint_fail():
    """Test that the verifier of an annotated constraint can fail."""
    with pytest.raises(Exception) as e:
        PositiveIntAttr((IntData(-42),))
    assert e.value.args[0] == "Expected positive integer, got -42."


################################################################################
# PyRDL Generic constraints
################################################################################

_T = TypeVar("_T", bound=BoolData | IntData)


@irdl_attr_definition
class ParamWrapperAttr(Generic[_T], ParametrizedAttribute):
    name = "test.int_or_bool_generic"

    param: ParameterDef[_T]

    def __init__(self, param: _T):
        super().__init__((param,))


def test_typevar_attribute_int():
    """Test the verifier of a generic attribute."""
    attr = ParamWrapperAttr[IntData](IntData(42))
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#test.int_or_bool_generic<#test.int<42>>"


def test_typevar_attribute_bool():
    """Test the verifier of a generic attribute."""
    attr = ParamWrapperAttr[BoolData](BoolData(True))
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#test.int_or_bool_generic<#test.bool<True>>"


def test_typevar_attribute_fail():
    """Test that the verifier of an generic attribute can fail."""
    with pytest.raises(Exception) as e:
        ParamWrapperAttr(StringData("foo"))  # pyright: ignore
    assert e.value.args[0] == "Unexpected attribute #test.str<foo>"


@irdl_attr_definition
class ParamConstrAttr(ParametrizedAttribute):
    name = "test.param_constr"

    param: ParameterDef[ParamWrapperAttr[IntData]]

    def __init__(self, param: ParameterDef[ParamWrapperAttr[IntData]]):
        super().__init__((param,))


def test_param_attr_constraint():
    """Test the verifier of an attribute with a parametric constraint."""
    attr = ParamConstrAttr(ParamWrapperAttr(IntData(42)))
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert (
        stream.getvalue()
        == "#test.param_constr<#test.int_or_bool_generic<#test.int<42>>>"
    )


def test_param_attr_constraint_fail():
    """
    Test that the verifier of an attribute with
    a parametric constraint can fail.
    """
    with pytest.raises(Exception) as e:
        ParamConstrAttr(ParamWrapperAttr(BoolData(True)))  # pyright: ignore
    assert e.value.args[0] == "#test.bool<True> should be of base attribute test.int"


_U = TypeVar("_U", bound=IntData)


@irdl_attr_definition
class NestedParamWrapperAttr(Generic[_U], ParametrizedAttribute):
    name = "test.nested_param_wrapper"

    param: ParameterDef[ParamWrapperAttr[_U]]

    def __init__(self, param: ParameterDef[ParamWrapperAttr[_U]]):
        super().__init__((param,))


def test_nested_generic_constraint():
    """
    Test the verifier of an attribute with a generic
    constraint used in a parametric constraint.
    """
    attr = NestedParamWrapperAttr[IntData](ParamWrapperAttr(IntData(42)))
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert (
        stream.getvalue()
        == "#test.nested_param_wrapper<#test.int_or_bool_generic<#test.int<42>>>"
    )


def test_nested_generic_constraint_fail():
    """
    Test that the verifier of an attribute with
    a parametric constraint can fail.
    """
    with pytest.raises(Exception) as e:
        NestedParamWrapperAttr(ParamWrapperAttr(BoolData(True)))  # pyright: ignore
    assert e.value.args[0] == "#test.bool<True> should be of base attribute test.int"


@irdl_attr_definition
class NestedParamConstrAttr(ParametrizedAttribute):
    name = "test.nested_param_constr"

    param: ParameterDef[NestedParamWrapperAttr[Annotated[IntData, PositiveIntConstr()]]]


def test_nested_param_attr_constraint():
    """
    Test the verifier of a nested parametric constraint.
    """
    attr = NestedParamConstrAttr(
        (NestedParamWrapperAttr(ParamWrapperAttr(IntData(42))),)
    )
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert (
        stream.getvalue()
        == "#test.nested_param_constr<#test.nested_param_wrapper<#test.int_or_bool_generic<#test.int<42>>>>"
    )


def test_nested_param_attr_constraint_fail():
    """
    Test that the verifier of a nested parametric constraint can fail.
    """
    with pytest.raises(Exception) as e:
        NestedParamConstrAttr((NestedParamWrapperAttr(ParamWrapperAttr(IntData(-42))),))
    assert e.value.args[0] == "Expected positive integer, got -42."


################################################################################
# Message Constraint
################################################################################


@irdl_attr_definition
class InformativeAttr(ParametrizedAttribute):
    name = "test.informative"

    param: ParameterDef[
        Annotated[
            Attribute,
            MessageConstraint(
                NoneAttr,
                "Dear user, here's what this constraint means in your abstraction.",
            ),
        ]
    ]


def test_informative_attribute():
    okay = InformativeAttr((NoneAttr(),))
    okay.verify()

    with pytest.raises(
        VerifyException,
        match="Dear user, here's what this constraint means in your abstraction.\nUnderlying verification failure: #test.int<42> should be of base attribute none",
    ):
        InformativeAttr((IntData(42),))


def test_informative_constraint():
    """
    Test the verifier of an informative constraint.
    """
    constr = MessageConstraint(NoneAttr(), "User-enlightening message.")
    with pytest.raises(
        VerifyException,
        match="User-enlightening message.\nUnderlying verification failure: Expected attribute #none but got #builtin.int<1>",
    ):
        constr.verify(IntAttr(1), ConstraintContext())
    assert constr.can_infer(set())
    assert constr.get_unique_base() == NoneAttr


################################################################################
# GenericData definition
################################################################################

_MissingGenericDataData = TypeVar("_MissingGenericDataData")


@irdl_attr_definition
class MissingGenericDataData(Data[_MissingGenericDataData]):
    name = "test.missing_genericdata"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> _MissingGenericDataData:
        raise NotImplementedError()

    def print_parameter(self, printer: Printer) -> None:
        raise NotImplementedError()

    def verify(self) -> None:
        return


class MissingGenericDataDataWrapper(ParametrizedAttribute):
    name = "test.missing_genericdata_wrapper"

    param: ParameterDef[MissingGenericDataData[int]]


def test_data_with_generic_missing_generic_data_failure():
    """
    Test error message when a generic data is used in constraints
    without implementing GenericData.
    """
    with pytest.raises(Exception) as e:
        irdl_attr_definition(MissingGenericDataDataWrapper)
    assert e.value.args[0] == (
        "Generic `Data` type 'test.missing_genericdata' cannot be converted to "
        "an attribute constraint. Consider making it inherit from "
        "`GenericData` instead of `Data`."
    )


A = TypeVar("A", bound=Attribute)


@dataclass(frozen=True)
class DataListAttr(AttrConstraint):
    """
    A constraint that enforces that the elements of a ListData all respect
    a constraint.
    """

    elem_constr: AttrConstraint

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        attr = cast(ListData[Attribute], attr)
        for e in attr.data:
            self.elem_constr.verify(e, constraint_context)


@irdl_attr_definition
class ListData(GenericData[list[A]]):
    name = "test.list"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> list[A]:
        raise NotImplementedError()

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
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
        assert (
            stream.getvalue()
            == "#test.list<[#test.bool<True>, #test.list<[#test.bool<False>]>]>"
        )


@irdl_attr_definition
class ListDataWrapper(ParametrizedAttribute):
    name = "test.list_wrapper"

    val: ParameterDef[ListData[BoolData]]


def test_generic_data_wrapper_verifier():
    """
    Test that a GenericData used in constraints pass the verifier when correct.
    """
    attr = ListDataWrapper((ListData([BoolData(True), BoolData(False)]),))
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert (
        stream.getvalue()
        == "#test.list_wrapper<#test.list<[#test.bool<True>, #test.bool<False>]>>"
    )


def test_generic_data_wrapper_verifier_failure():
    """
    Test that a GenericData used in constraints fails
    the verifier when constraints are not satisfied.
    """
    with pytest.raises(VerifyException) as e:
        ListDataWrapper((ListData([BoolData(True), ListData([BoolData(False)])]),))
    assert (
        e.value.args[0]
        == "#test.list<[#test.bool<False>]> should be of base attribute test.bool"
    )
    assert (
        e.value.args[0]
        == "#test.list<[#test.bool<False>]> should be of base attribute test.bool"
    )


@irdl_attr_definition
class ListDataNoGenericsWrapper(ParametrizedAttribute):
    name = "test.list_no_generics_wrapper"

    val: ParameterDef[AnyListData]


def test_generic_data_no_generics_wrapper_verifier():
    """
    Test that GenericType can be used in constraints without a parameter.
    """
    attr = ListDataNoGenericsWrapper(
        (ListData([BoolData(True), ListData([BoolData(False)])]),)
    )
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert (
        stream.getvalue()
        == "#test.list_no_generics_wrapper<#test.list<[#test.bool<True>, #test.list<[#test.bool<False>]>]>>"
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

    assert ParamAttrDefAttr.get_irdl_definition() == ParamAttrDef(
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
                super().__init__((IntData(param),))
            case str():
                super().__init__((StringData(param),))


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
