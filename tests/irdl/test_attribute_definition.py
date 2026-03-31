"""
Test the definition of attributes and their constraints.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import auto
from io import StringIO
from typing import Annotated, ClassVar, Generic, TypeAlias

import pytest
from typing_extensions import TypeVar, override

from xdsl.context import Context
from xdsl.dialects.builtin import (
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    NoneAttr,
    Signedness,
    StringAttr,
    i32,
)
from xdsl.ir import (
    Attribute,
    AttributeInvT,
    BitEnumAttribute,
    BuiltinAttribute,
    Data,
    EnumAttribute,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    StrEnum,
    TypedAttribute,
)
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AnyOf,
    AttrConstraint,
    BaseAttr,
    ConstraintContext,
    GenericData,
    IntConstraint,
    MessageConstraint,
    ParamAttrConstraint,
    ParamAttrDef,
    ParamDef,
    TypeVarConstraint,
    VarConstraint,
    base,
    irdl_attr_definition,
    param_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import PyRDLAttrDefinitionError, VerifyException
from xdsl.utils.hints import isa


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
            printer.print_int(self.data)


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
            with printer.in_square_brackets():
                printer.print_list(self.data, printer.print_int)


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


def test_attribute_def_with_non_identifier_enum():
    """
    Test the definition of an EnumAttribute with a non-identifier enum is
    allowed.
    """

    @irdl_attr_definition
    class TestNonIdentifierEnumAttr(  # pyright: ignore[reportUnusedClass]
        EnumAttribute[TestNonIdentifierEnum]
    ):
        name = "test.non_identifier_enum"


@irdl_attr_definition(init=False)
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


def test_typed_attribute_parsing_printing():
    """
    Test that non builtin TypedAttributes are parsed and printed correctly.
    """

    @irdl_attr_definition
    class TypedAttr(TypedAttribute):
        name = "test.typed"
        value: IntAttr
        type: IntegerType

        @classmethod
        def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
            with parser.in_angle_brackets():
                value = parser.parse_integer()
            parser.parse_punctuation(":")
            type = parser.parse_type()
            return (IntAttr(value), type)

        def print_parameters(self, printer: Printer) -> None:
            with printer.in_angle_brackets():
                printer.print_int(self.value.data)
            printer.print_string(" : ")
            printer.print_attribute(self.type)

        @classmethod
        def get_type_index(cls) -> int:
            return 1

        @staticmethod
        def parse_with_type(
            parser: AttrParser,
            type: Attribute,
        ) -> TypedAttribute:
            raise NotImplementedError()

        def print_without_type(self, printer: Printer) -> None:
            raise NotImplementedError()

    ctx = Context()
    ctx.load_attr_or_type(TypedAttr)
    attr = Parser(ctx, "#test.typed<42> : i32").parse_attribute()
    assert attr == TypedAttr(IntAttr(42), i32)

    assert str(attr) == "#test.typed<42> : i32"


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

    param: BoolData


def test_bose_constraint():
    """Test the verifier of a base attribute type constraint."""
    attr = BoolWrapperAttr(BoolData(True))
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#test.bool_wrapper<#test.bool<True>>"


def test_base_constraint_fail():
    """Test the verifier of a union constraint."""
    with pytest.raises(
        VerifyException, match="#test.str<foo> should be of base attribute test.bool"
    ):
        BoolWrapperAttr(StringData("foo"))  # pyright: ignore[reportArgumentType]


################################################################################
# PyRDL union constraints
################################################################################


@irdl_attr_definition
class BoolOrIntParamAttr(ParametrizedAttribute):
    name = "test.bool_or_int"

    param: BoolData | IntData


def test_union_constraint_left():
    """Test the verifier of a union constraint."""
    attr = BoolOrIntParamAttr(BoolData(True))
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#test.bool_or_int<#test.bool<True>>"


def test_union_constraint_right():
    """Test the verifier of a union constraint."""
    attr = BoolOrIntParamAttr(IntData(42))
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#test.bool_or_int<#test.int<42>>"


def test_union_constraint_fail():
    """Test the verifier of a union constraint."""
    with pytest.raises(VerifyException, match="Unexpected attribute #test.str<foo>"):
        BoolOrIntParamAttr(StringData("foo"))  # pyright: ignore[reportArgumentType]


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

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> PositiveIntConstr:
        return self


@irdl_attr_definition
class PositiveIntAttr(ParametrizedAttribute):
    name = "test.positive_int"

    param: Attribute = param_def(PositiveIntConstr())


def test_annotated_constraint():
    """Test the verifier of an annotated constraint."""
    attr = PositiveIntAttr(IntData(42))
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "#test.positive_int<#test.int<42>>"


def test_annotated_constraint_fail():
    """Test that the verifier of an annotated constraint can fail."""
    with pytest.raises(VerifyException, match="Expected positive integer, got -42."):
        PositiveIntAttr(IntData(-42))


################################################################################
# PyRDL Generic constraints
################################################################################

_T = TypeVar("_T", bound=BoolData | IntData)


@irdl_attr_definition
class ParamWrapperAttr(ParametrizedAttribute, Generic[_T]):
    name = "test.int_or_bool_generic"

    param: _T


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
    with pytest.raises(VerifyException, match="Unexpected attribute #test.str<foo>"):
        ParamWrapperAttr(StringData("foo"))  # pyright: ignore


@irdl_attr_definition
class ParamConstrAttr(ParametrizedAttribute):
    name = "test.param_constr"

    param: ParamWrapperAttr[IntData]


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
    with pytest.raises(
        VerifyException, match="#test.bool<True> should be of base attribute test.int"
    ):
        ParamConstrAttr(ParamWrapperAttr(BoolData(True)))  # pyright: ignore


_U = TypeVar("_U", bound=IntData)


@irdl_attr_definition
class NestedParamWrapperAttr(ParametrizedAttribute, Generic[_U]):
    name = "test.nested_param_wrapper"

    param: ParamWrapperAttr[_U]


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
    with pytest.raises(
        VerifyException, match="#test.bool<True> should be of base attribute test.int"
    ):
        NestedParamWrapperAttr(ParamWrapperAttr(BoolData(True)))  # pyright: ignore


@irdl_attr_definition
class NestedParamConstrAttr(ParametrizedAttribute):
    name = "test.nested_param_constr"

    param: NestedParamWrapperAttr[Annotated[IntData, PositiveIntConstr()]]


def test_nested_param_attr_constraint():
    """
    Test the verifier of a nested parametric constraint.
    """
    attr = NestedParamConstrAttr(NestedParamWrapperAttr(ParamWrapperAttr(IntData(42))))
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
    with pytest.raises(VerifyException, match="Expected positive integer, got -42."):
        NestedParamConstrAttr(NestedParamWrapperAttr(ParamWrapperAttr(IntData(-42))))


################################################################################
# Message Constraint
################################################################################


@irdl_attr_definition
class InformativeAttr(ParametrizedAttribute):
    name = "test.informative"

    param: Attribute = param_def(
        MessageConstraint(
            NoneAttr,
            "Dear user, here's what this constraint means in your abstraction.",
        )
    )


def test_informative_attribute():
    okay = InformativeAttr(NoneAttr())
    okay.verify()

    with pytest.raises(
        VerifyException,
        match="Dear user, here's what this constraint means in your abstraction.\nUnderlying verification failure: #test.int<42> should be of base attribute none",
    ):
        InformativeAttr(IntData(42))


def test_informative_constraint():
    """
    Test the verifier of an informative constraint.
    """
    constr = MessageConstraint(NoneAttr(), "User-enlightening message.")
    with pytest.raises(
        VerifyException,
        match="User-enlightening message.\nUnderlying verification failure: Expected attribute none but got #builtin.int<1>",
    ):
        constr.verify(IntAttr(1), ConstraintContext())
    assert constr.can_infer(set())
    assert constr.get_bases() == {NoneAttr}


################################################################################
# GenericData definition
################################################################################


@irdl_attr_definition
class ListData(GenericData[tuple[AttributeInvT, ...]], Generic[AttributeInvT]):
    name = "test.list"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> tuple[AttributeInvT, ...]:
        raise NotImplementedError()

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("[")
            printer.print_list(self.data, printer.print_attribute)
            printer.print_string("]")

    @staticmethod
    @override
    def constr() -> DataListAttr:
        return DataListAttr(TypeVarConstraint(AttributeInvT, AnyAttr()))

    @staticmethod
    def from_list(data: list[AttributeInvT]) -> ListData[AttributeInvT]:
        return ListData(tuple(data))


AnyListData: TypeAlias = ListData[Attribute]


@dataclass(frozen=True)
class DataListAttr(AttrConstraint[ListData[AttributeInvT]]):
    """
    A constraint that enforces that the elements of a ListData all respect
    a constraint.
    """

    elem_constr: AttrConstraint[AttributeInvT]

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        if not isa(attr, ListData):
            raise VerifyException(
                f"Expected {attr} to be instance of {ListData.__name__}"
            )
        for e in attr.data:
            self.elem_constr.verify(e, constraint_context)

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> DataListAttr[AttributeInvT]:
        return DataListAttr(self.elem_constr.mapping_type_vars(type_var_mapping))


class Test_generic_data_verifier:
    def test_generic_data_verifier(self):
        """
        Test that a GenericData can be created.
        """
        attr = ListData((BoolData(True), ListData((BoolData(False),))))
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

    val: ListData[BoolData]


def test_generic_data_wrapper_verifier():
    """
    Test that a GenericData used in constraints pass the verifier when correct.
    """
    attr = ListDataWrapper(ListData((BoolData(True), BoolData(False))))
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
    with pytest.raises(
        VerifyException,
        match=re.escape(
            "#test.list<[#test.bool<False>]> should be of base attribute test.bool"
        ),
    ):
        ListDataWrapper(ListData((BoolData(True), ListData((BoolData(False),)))))  # pyright: ignore[reportArgumentType]


@irdl_attr_definition
class ListDataNoGenericsWrapper(ParametrizedAttribute):
    name = "test.list_no_generics_wrapper"

    val: AnyListData


def test_generic_data_no_generics_wrapper_verifier():
    """
    Test that GenericType can be used in constraints without a parameter.
    """
    attr = ListDataNoGenericsWrapper(
        ListData((BoolData(True), ListData((BoolData(False),))))
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

    arg1: Attribute
    arg2: BoolData

    # Check that we can define methods in attribute definition
    def test(self):
        pass


def test_irdl_definition():
    """Test that we can get the IRDL definition of a parametrized attribute."""
    assert ParamAttrDefAttr.get_irdl_definition() == ParamAttrDef(
        "test.param_attr_def_attr",
        [("arg1", ParamDef(AnyAttr())), ("arg2", ParamDef(BaseAttr(BoolData)))],
    )


@irdl_attr_definition
class ParamAttrDefAttr2(ParametrizedAttribute):
    name = "test.param_attr_def_attr"

    arg1: Attribute = param_def(base(IntAttr))
    arg2: BoolData

    # Check that we can define methods in attribute definition
    def test(self):
        pass


def test_irdl_definition2():
    """Test that we can get the IRDL definition of a parametrized attribute."""

    assert ParamAttrDefAttr2.get_irdl_definition() == ParamAttrDef(
        "test.param_attr_def_attr",
        [
            ("arg1", ParamDef(AnyAttr() & BaseAttr(IntAttr))),
            ("arg2", ParamDef(BaseAttr(BoolData))),
        ],
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

    param: Attribute

    def __init__(self, param: int | str):
        match param:
            case int():
                super().__init__(IntData(param))
            case str():
                super().__init__(StringData(param))


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


@irdl_attr_definition
class GenericAttr(ParametrizedAttribute, Generic[AttributeInvT]):
    name = "test.generic_attr"

    param: AttributeInvT


def test_generic_attr():
    """Test the generic parameter of a ParametrizedAttribute."""

    assert GenericAttr.get_irdl_definition() == ParamAttrDef(
        "test.generic_attr",
        [
            (
                "param",
                ParamDef(
                    TypeVarConstraint(
                        type_var=AttributeInvT,
                        base_constraint=AnyAttr(),
                    )
                ),
            )
        ],
    )

    assert base(GenericAttr[IntAttr]) == ParamAttrConstraint(
        GenericAttr, (BaseAttr(IntAttr),)
    )


################################################################################
# ConstraintVar
################################################################################


@irdl_attr_definition
class ConstraintVarAttr(ParametrizedAttribute):
    name = "test.constraint_var"

    T: ClassVar = VarConstraint("T", BaseAttr(IntegerType))

    param1: IntegerAttr = param_def(IntegerAttr.constr(type=T))
    param2: IntegerAttr = param_def(IntegerAttr.constr(type=T))


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


################################################################################
# Names
################################################################################


def test_non_builtin_name_fail():
    """
    Test that the name of an attribute is properly checked
    when it is not a builtin attribute.
    """
    with pytest.raises(PyRDLAttrDefinitionError, match="is not a valid attribute name"):

        @irdl_attr_definition
        class NonBuiltinNameAttr(  # pyright: ignore[reportUnusedClass]
            ParametrizedAttribute
        ):
            name = "vector"


def test_non_builtin_name():
    """
    Test that the name of an attribute is properly checked
    when it is not a builtin attribute.
    """

    @irdl_attr_definition
    class NonBuiltinNameAttr(  # pyright: ignore[reportUnusedClass]
        ParametrizedAttribute
    ):
        name = "test.vector"


def test_builtin_name():
    """
    Test that builtin attribute names are not checked.
    """

    @irdl_attr_definition
    class BuiltinNameAttr(  # pyright: ignore[reportUnusedClass]
        ParametrizedAttribute, BuiltinAttribute
    ):
        name = "builtin.vector"


################################################################################
# Mapping Type Var
################################################################################


@irdl_attr_definition
class A(Data[int]):
    name = "test.a"


@irdl_attr_definition
class B(Data[int]):
    name = "test.b"


@irdl_attr_definition
class C(Data[int]):
    name = "test.c"


_A = TypeVar("_A", bound=Attribute)


def test_var_constraint():
    var_constraint = VarConstraint("var", TypeVarConstraint(_A, BaseAttr(A)))

    with pytest.raises(KeyError, match="Mapping value missing for type var ~_A"):
        var_constraint.mapping_type_vars({})

    assert var_constraint.mapping_type_vars({_A: BaseAttr(B)}) == VarConstraint(
        "var", BaseAttr(B)
    )


def test_typevar_constraint():
    typevar_constraint = TypeVarConstraint(_A, BaseAttr(A))

    with pytest.raises(
        KeyError, match=re.escape("Mapping value missing for type var ~_A")
    ):
        assert typevar_constraint.mapping_type_vars({})
    assert typevar_constraint.mapping_type_vars({_A: BaseAttr(B)}) == BaseAttr(B)


def test_message_constraint():
    message_constraint = MessageConstraint(
        TypeVarConstraint(_A, BaseAttr(A)), "test message"
    )

    assert message_constraint.mapping_type_vars({_A: BaseAttr(B)}) == MessageConstraint(
        BaseAttr(B), "test message"
    )


def test_anyof_constraint():
    anyof_constraint = AnyOf((TypeVarConstraint(_A, BaseAttr(A)), BaseAttr(B)))

    assert anyof_constraint.mapping_type_vars({_A: BaseAttr(C)}) == AnyOf(
        (BaseAttr(C), BaseAttr(B))
    )


def test_allof_constraint():
    allof_constraint = AllOf((TypeVarConstraint(_A, BaseAttr(A)), BaseAttr(B)))

    assert allof_constraint.mapping_type_vars({_A: BaseAttr(C)}) == AllOf(
        (BaseAttr(C), BaseAttr(B))
    )


################################################################################
# Constant ClassVar
################################################################################


def test_class_var_pass():
    """Test that ClassVar constants are allowed in attribute definitions."""

    @irdl_attr_definition
    class ClassVarAttr(ParametrizedAttribute):  # pyright: ignore[reportUnusedClass]
        name = "test.class_var"
        CONSTANT: ClassVar[int]
        param: IntData

    @irdl_attr_definition
    class ClassVarAttr2(ParametrizedAttribute):  # pyright: ignore[reportUnusedClass]
        name = "test.class_var"
        CONSTANT: ClassVar[int] = 2
        param: IntData


def test_class_var_fail():
    """Test that lowercase ClassVar fields are not allowed."""
    with pytest.raises(
        PyRDLAttrDefinitionError,
        match='Invalid ClassVar name "constant", must be uppercase.',
    ):

        @irdl_attr_definition
        class InvalidClassVarAttr(ParametrizedAttribute):  # pyright: ignore[reportUnusedClass]
            name = "test.invalid_class_var"
            constant: ClassVar[int]  # Should be uppercase
            param: IntData


################################################################################
# Converters
################################################################################


@irdl_attr_definition
class ConverterAttr(ParametrizedAttribute):
    name = "test.converters"

    string: StringAttr = param_def(converter=StringAttr.get)

    i: IntAttr = param_def(converter=IntAttr.get)


def test_converters():
    string = "My string"
    string_attr = StringAttr(string)

    i = 2
    i_attr = IntAttr(i)

    attr_no_convertion = ConverterAttr(string_attr, i_attr)
    assert attr_no_convertion.i == i_attr
    assert attr_no_convertion.string == string_attr

    attr_some_convertion = ConverterAttr(string_attr, i)
    assert attr_some_convertion.i == i_attr
    assert attr_some_convertion.string == string_attr
    assert attr_no_convertion == attr_some_convertion

    attr_all_convertion = ConverterAttr(string, i)
    assert attr_all_convertion.i == i_attr
    assert attr_all_convertion.string == string_attr
    assert attr_no_convertion == attr_all_convertion
