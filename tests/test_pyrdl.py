"""Unit tests for IRDL."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import auto
from typing import Literal

import pytest
from typing_extensions import Self, TypeVar

from xdsl.dialects.builtin import (
    I32,
    IntAttr,
    IntAttrConstraint,
    IntegerAttr,
    IntegerType,
    Signedness,
    SignednessAttr,
)
from xdsl.ir import Attribute, Data, ParametrizedAttribute
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AnyInt,
    AnyOf,
    AttrConstraint,
    BaseAttr,
    ConstraintContext,
    ConstraintConvertible,
    EqAttrConstraint,
    EqIntConstraint,
    IntConstraint,
    IntSetConstraint,
    ParamAttrConstraint,
    VarConstraint,
    eq,
    get_constraint,
    irdl_attr_definition,
    irdl_to_attr_constraint,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import PyRDLTypeError, VerifyException
from xdsl.utils.str_enum import StrEnum


@irdl_attr_definition
class BoolData(Data[bool]):
    """An attribute holding a boolean value."""

    name = "test.bool"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> bool:
        raise NotImplementedError()

    def print_parameter(self, printer: Printer):
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
class FloatData(Data[float]):
    """An attribute holding a float value."""

    name = "test.float"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> float:
        with parser.in_angle_brackets():
            return parser.parse_float()

    def print_parameter(self, printer: Printer):
        with printer.in_angle_brackets():
            printer.print_string(str(self.data))


@irdl_attr_definition
class DoubleParamAttr(ParametrizedAttribute):
    """An attribute with two unbounded attribute parameters."""

    name = "test.param"

    param1: Attribute
    param2: Attribute


def test_eq_attr_verify():
    """Check that an EqAttrConstraint verifies the expected attribute"""
    bool_true = BoolData(True)
    eq_true_constraint = EqAttrConstraint(bool_true)
    eq_true_constraint.verify(bool_true, ConstraintContext())


def test_eq_attr_verify_wrong_parameters_fail():
    """
    Check that an EqAttrConstraint fails to verify an attribute with different
    parameters.
    """
    bool_true = BoolData(True)
    bool_false = BoolData(False)
    eq_true_constraint = EqAttrConstraint(bool_true)
    with pytest.raises(
        VerifyException, match=f"Expected attribute {bool_true} but got {bool_false}"
    ):
        eq_true_constraint.verify(bool_false, ConstraintContext())


def test_eq_attr_verify_wrong_base_fail():
    """
    Check that an EqAttrConstraint fails to verify an attribute with a
    different base attribute.
    """
    bool_true = BoolData(True)
    int_zero = IntData(0)
    eq_true_constraint = EqAttrConstraint(bool_true)
    with pytest.raises(
        VerifyException, match=f"Expected attribute {bool_true} but got {int_zero}"
    ):
        eq_true_constraint.verify(int_zero, ConstraintContext())


def test_base_attr_verify():
    """
    Check that a BaseAttr constraint verifies an attribute with the expected
    base attribute.
    """
    eq_true_constraint = BaseAttr(BoolData)
    eq_true_constraint.verify(BoolData(True), ConstraintContext())
    eq_true_constraint.verify(BoolData(False), ConstraintContext())


def test_base_attr_verify_wrong_base_fail():
    """
    Check that a BaseAttr constraint fails to verify an attribute with a
    different base attribute.
    """
    eq_true_constraint = BaseAttr(BoolData)
    int_zero = IntData(0)
    with pytest.raises(
        VerifyException, match=f"{int_zero} should be of base attribute {BoolData.name}"
    ):
        eq_true_constraint.verify(int_zero, ConstraintContext())


def test_any_attr_verify():
    """Check that an AnyAttr verifies any attribute."""
    any_constraint = AnyAttr()
    any_constraint.verify(BoolData(True), ConstraintContext())
    any_constraint.verify(BoolData(False), ConstraintContext())
    any_constraint.verify(IntData(0), ConstraintContext())


@dataclass(frozen=True)
class LessThan(AttrConstraint):
    bound: int

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        if not isinstance(attr, IntData):
            raise VerifyException(f"{attr} should be of base attribute {IntData.name}")
        if attr.data >= self.bound:
            raise VerifyException(f"{attr} should hold a value less than {self.bound}")

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> Self:
        return self


@dataclass(frozen=True)
class GreaterThan(AttrConstraint):
    bound: int

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if not isinstance(attr, IntData):
            raise VerifyException(f"{attr} should be of base attribute {IntData.name}")
        if attr.data <= self.bound:
            raise VerifyException(
                f"{attr} should hold a value greater than {self.bound}"
            )

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> Self:
        return self


def test_anyof_verify():
    """
    Check that an AnyOf constraint verifies if one of the constraints
    verify.
    """
    constraint = BaseAttr(BoolData) | BaseAttr(IntData)
    constraint.verify(IntData(1), ConstraintContext())
    constraint.verify(IntData(-10), ConstraintContext())
    constraint.verify(BoolData(True), ConstraintContext())
    constraint.verify(BoolData(False), ConstraintContext())


def test_anyof_verify_fail():
    """
    Check that an AnyOf constraint fails to verify if none of the constraints
    verify.
    """
    constraint = BaseAttr(BoolData) | BaseAttr(IntData)

    f = FloatData(10.0)

    with pytest.raises(
        VerifyException, match=r"Unexpected attribute #test.float<10.0>"
    ):
        constraint.verify(f, ConstraintContext())


def test_allof_verify():
    """
    Check that an AllOf constraint verifies if all of the constraints
    verify.
    """
    constraint = AllOf((LessThan(10), GreaterThan(0)))
    constraint.verify(IntData(1), ConstraintContext())
    constraint.verify(IntData(9), ConstraintContext())
    constraint.verify(IntData(5), ConstraintContext())


def test_allof_verify_fail():
    """
    Check that an AllOf constraint fails to verify if one of the constraints
    fails to verify.
    """
    constraint = AllOf((LessThan(10), GreaterThan(0)))

    with pytest.raises(
        VerifyException, match=f"{IntData(10)} should hold a value less than 10"
    ):
        constraint.verify(IntData(10), ConstraintContext())

    with pytest.raises(
        VerifyException, match=f"{IntData(0)} should hold a value greater than 0"
    ):
        constraint.verify(IntData(0), ConstraintContext())


def test_allof_verify_multiple_failures():
    """
    Check that an AllOf constraint provides verification info for all related constraints
    even when one of them fails.
    """
    constraint = AllOf((LessThan(5), GreaterThan(8)))

    with pytest.raises(
        VerifyException,
        match=f"The following constraints were not satisfied:\n{IntData(7)} should "
        f"hold a value less than 5\n{IntData(7)} should hold a value greater than 8",
    ):
        constraint.verify(IntData(7), ConstraintContext())


def test_param_attr_verify():
    bool_true = BoolData(True)
    constraint = ParamAttrConstraint(
        DoubleParamAttr, [EqAttrConstraint(bool_true), BaseAttr(IntData)]
    )
    constraint.verify(DoubleParamAttr(bool_true, IntData(0)), ConstraintContext())
    constraint.verify(DoubleParamAttr(bool_true, IntData(42)), ConstraintContext())


def test_param_attr_verify_base_fail():
    bool_true = BoolData(True)
    constraint = ParamAttrConstraint(
        DoubleParamAttr, [EqAttrConstraint(bool_true), BaseAttr(IntData)]
    )
    with pytest.raises(
        VerifyException,
        match=f"{bool_true} should be of base attribute {DoubleParamAttr.name}",
    ):
        constraint.verify(bool_true, ConstraintContext())


def test_param_attr_verify_params_num_params_fail():
    bool_true = BoolData(True)
    constraint = ParamAttrConstraint(DoubleParamAttr, [EqAttrConstraint(bool_true)])
    attr = DoubleParamAttr(bool_true, IntData(0))
    with pytest.raises(VerifyException, match="1 parameters expected, but got 2"):
        constraint.verify(attr, ConstraintContext())


def test_param_attr_verify_params_fail():
    bool_true = BoolData(True)
    bool_false = BoolData(False)
    constraint = ParamAttrConstraint(
        DoubleParamAttr, [EqAttrConstraint(bool_true), BaseAttr(IntData)]
    )

    with pytest.raises(
        VerifyException,
        match=f"{bool_false} should be of base attribute {IntData.name}",
    ):
        constraint.verify(DoubleParamAttr(bool_true, bool_false), ConstraintContext())

    with pytest.raises(
        VerifyException, match=f"Expected attribute {bool_true} but got {bool_false}"
    ):
        constraint.verify(DoubleParamAttr(bool_false, IntData(0)), ConstraintContext())


def test_constraint_vars_success():
    """Test that VarConstraint verifier succeed when given the same attributes."""

    constraint = VarConstraint("T", eq(BoolData(False)) | eq(IntData(0)))

    constraint_context = ConstraintContext()
    constraint.verify(BoolData(False), constraint_context)
    constraint.verify(BoolData(False), constraint_context)

    constraint_context = ConstraintContext()
    constraint.verify(IntData(0), constraint_context)
    constraint.verify(IntData(0), constraint_context)


def test_constraint_vars_fail_different():
    """Test that VarConstraint verifier fails when given different attributes."""

    constraint = VarConstraint("T", eq(BoolData(False)) | eq(IntData(0)))

    constraint_context = ConstraintContext()
    constraint.verify(IntData(0), constraint_context)

    with pytest.raises(VerifyException):
        constraint.verify(BoolData(False), constraint_context)


def test_constraint_vars_fail_underlying_constraint():
    """
    Test that VarConstraint verifier fails when given
    attributes that fail the underlying constraint.
    """

    constraint = VarConstraint("T", eq(BoolData(False)) | eq(IntData(0)))

    with pytest.raises(VerifyException):
        constraint.verify(IntData(1), ConstraintContext())


# region: irdl_to_attr_constraint

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


def test_data_with_generic_missing_generic_data_failure():
    """
    Test error message when a generic data is used in constraints
    without implementing GenericData.
    """
    with pytest.raises(
        PyRDLTypeError,
        match=(
            "Generic `Data` type 'test.missing_genericdata' cannot be converted to an "
            "attribute constraint. Consider making it inherit from `GenericData` "
            "instead of `Data`."
        ),
    ):
        irdl_to_attr_constraint(MissingGenericDataData[int])


class TestEnum(ConstraintConvertible["TestEnumAttr"], StrEnum):
    A = auto()
    B = auto()

    @staticmethod
    def base_constr() -> AttrConstraint["TestEnumAttr"]:
        return BaseAttr(TestEnumAttr)

    def constr(self) -> AttrConstraint["TestEnumAttr"]:
        return EqAttrConstraint(TestEnumAttr(self))


class TestEnumAttr(Data[TestEnum]):
    name = "test.enum"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> TestEnum: ...
    def print_parameter(self, printer: Printer) -> None: ...


def test_irdl_to_attr_constraint():
    with pytest.raises(
        PyRDLTypeError, match="Unexpected irdl constraint: <class 'int'>"
    ):
        irdl_to_attr_constraint(int)  # pyright: ignore[reportArgumentType]

    assert irdl_to_attr_constraint(TestEnumAttr) == BaseAttr(TestEnumAttr)
    assert irdl_to_attr_constraint(TestEnumAttr(TestEnum.A)) == EqAttrConstraint(
        TestEnumAttr(TestEnum.A)
    )
    assert irdl_to_attr_constraint(TestEnum) == BaseAttr(TestEnumAttr)
    assert irdl_to_attr_constraint(TestEnum.A) == EqAttrConstraint(
        TestEnumAttr(TestEnum.A)
    )
    assert irdl_to_attr_constraint(Literal[TestEnum.A]) == EqAttrConstraint(
        TestEnumAttr(TestEnum.A)
    )
    assert irdl_to_attr_constraint(Literal[TestEnum.A, TestEnum.B]) == AnyOf(  # pyright: ignore[reportArgumentType]
        (
            EqAttrConstraint(TestEnumAttr(TestEnum.A)),
            EqAttrConstraint(TestEnumAttr(TestEnum.B)),
        )
    )
    assert irdl_to_attr_constraint(IntAttr) == BaseAttr(IntAttr)
    assert irdl_to_attr_constraint(IntAttr[int]) == IntAttrConstraint(
        int_constraint=AnyInt()
    )
    assert irdl_to_attr_constraint(IntAttr[Literal[1]]) == IntAttrConstraint(
        int_constraint=EqIntConstraint(value=1)
    )
    assert irdl_to_attr_constraint(IntAttr[2]) == IntAttrConstraint(
        int_constraint=EqIntConstraint(value=2)
    )

    assert irdl_to_attr_constraint(IntegerType) == BaseAttr(IntegerType)
    # With one type arg, second default
    assert irdl_to_attr_constraint(IntegerType[int]) == ParamAttrConstraint(
        IntegerType,
        (IntAttrConstraint(AnyInt()), BaseAttr(SignednessAttr)),
    )
    # With both type args
    assert irdl_to_attr_constraint(IntegerType[int, Signedness]) == ParamAttrConstraint(
        IntegerType,
        (IntAttrConstraint(AnyInt()), BaseAttr(SignednessAttr)),
    )
    assert irdl_to_attr_constraint(IntegerType[32]) == ParamAttrConstraint(
        IntegerType,
        (IntAttrConstraint(EqIntConstraint(32)), BaseAttr(SignednessAttr)),
    )
    assert irdl_to_attr_constraint(IntegerType[Literal[32, 64]]) == ParamAttrConstraint(
        IntegerType,
        (
            IntAttrConstraint(IntSetConstraint(frozenset((32, 64)))),
            BaseAttr(SignednessAttr),
        ),
    )

    assert irdl_to_attr_constraint(Signedness.SIGNED) == EqAttrConstraint(
        SignednessAttr(Signedness.SIGNED)
    )

    assert irdl_to_attr_constraint(Signedness) == BaseAttr(SignednessAttr)
    assert irdl_to_attr_constraint(SignednessAttr) == BaseAttr(SignednessAttr)
    assert irdl_to_attr_constraint(
        SignednessAttr[Signedness.SIGNED]
    ) == EqAttrConstraint(SignednessAttr(Signedness.SIGNED))

    assert irdl_to_attr_constraint(IntegerAttr[I32]) == ParamAttrConstraint(
        IntegerAttr,
        (
            BaseAttr(IntAttr),
            ParamAttrConstraint(
                IntegerType,
                (
                    IntAttrConstraint(EqIntConstraint(32)),
                    EqAttrConstraint(SignednessAttr(Signedness.SIGNLESS)),
                ),
            ),
        ),
    )
    assert irdl_to_attr_constraint(IntegerAttr[IntegerType[32]]) == ParamAttrConstraint(
        IntegerAttr,
        (
            BaseAttr(IntAttr),
            ParamAttrConstraint(
                IntegerType,
                (IntAttrConstraint(EqIntConstraint(32)), BaseAttr(SignednessAttr)),
            ),
        ),
    )


def test_get_constraint():
    assert get_constraint(int) == AnyInt()
    assert get_constraint(Literal[1]) == EqIntConstraint(1)
    assert get_constraint(Literal[2]) == EqIntConstraint(2)
    assert get_constraint(Literal[2, 3]) == IntSetConstraint(frozenset((2, 3)))

    assert get_constraint(
        Literal[2] | Literal[3]  # noqa
    ) == IntSetConstraint(frozenset((2, 3)))

    assert get_constraint(
        Literal[1] | Literal[2, 3]  # noqa
    ) == IntSetConstraint(frozenset((1, 2, 3)))

    assert get_constraint(TestEnumAttr) == BaseAttr(TestEnumAttr)
    assert get_constraint(TestEnumAttr(TestEnum.A)) == EqAttrConstraint(
        TestEnumAttr(TestEnum.A)
    )
    assert get_constraint(TestEnum) == BaseAttr(TestEnumAttr)
    assert get_constraint(TestEnum.A) == EqAttrConstraint(TestEnumAttr(TestEnum.A))
    assert get_constraint(Literal[TestEnum.A]) == EqAttrConstraint(
        TestEnumAttr(TestEnum.A)
    )
    assert get_constraint(Literal[TestEnum.A, TestEnum.B]) == AnyOf(
        (
            EqAttrConstraint(TestEnumAttr(TestEnum.A)),
            EqAttrConstraint(TestEnumAttr(TestEnum.B)),
        )
    )

    with pytest.raises(
        PyRDLTypeError, match="Unexpected irdl constraint: <class 'str'>"
    ):
        get_constraint(str)  # pyright: ignore[reportCallIssue, reportArgumentType]


# endregion
