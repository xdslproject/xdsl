"""Unit tests for IRDL."""
from dataclasses import dataclass

import pytest

from xdsl.ir import Attribute, Data, ParametrizedAttribute
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AnyOf,
    AttrConstraint,
    BaseAttr,
    EqAttrConstraint,
    ParamAttrConstraint,
    ParameterDef,
    VarConstraint,
    irdl_attr_definition,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class BoolData(Data[bool]):
    """An attribute holding a boolean value."""

    name = "bool"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> bool:
        raise NotImplementedError()

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
class DoubleParamAttr(ParametrizedAttribute):
    """An attribute with two unbounded attribute parameters."""

    name = "param"

    param1: ParameterDef[Attribute]
    param2: ParameterDef[Attribute]


def test_eq_attr_verify():
    """Check that an EqAttrConstraint verifies the expected attribute"""
    bool_true = BoolData(True)
    eq_true_constraint = EqAttrConstraint(bool_true)
    eq_true_constraint.verify(bool_true, {})


def test_eq_attr_verify_wrong_parameters_fail():
    """
    Check that an EqAttrConstraint fails to verify an attribute with different
    parameters.
    """
    bool_true = BoolData(True)
    bool_false = BoolData(False)
    eq_true_constraint = EqAttrConstraint(bool_true)
    with pytest.raises(VerifyException) as e:
        eq_true_constraint.verify(bool_false, {})
    assert e.value.args[0] == (f"Expected attribute {bool_true} but got {bool_false}")


def test_eq_attr_verify_wrong_base_fail():
    """
    Check that an EqAttrConstraint fails to verify an attribute with a
    different base attribute.
    """
    bool_true = BoolData(True)
    int_zero = IntData(0)
    eq_true_constraint = EqAttrConstraint(bool_true)
    with pytest.raises(VerifyException) as e:
        eq_true_constraint.verify(int_zero, {})
    assert e.value.args[0] == (f"Expected attribute {bool_true} but got {int_zero}")


def test_base_attr_verify():
    """
    Check that a BaseAttr constraint verifies an attribute with the expected
    base attribute.
    """
    eq_true_constraint = BaseAttr(BoolData)
    eq_true_constraint.verify(BoolData(True), {})
    eq_true_constraint.verify(BoolData(False), {})


def test_base_attr_verify_wrong_base_fail():
    """
    Check that a BaseAttr constraint fails to verify an attribute with a
    different base attribute.
    """
    eq_true_constraint = BaseAttr(BoolData)
    int_zero = IntData(0)
    with pytest.raises(VerifyException) as e:
        eq_true_constraint.verify(int_zero, {})
    assert e.value.args[0] == (
        f"{int_zero} should be of base attribute {BoolData.name}"
    )


def test_any_attr_verify():
    """Check that an AnyAttr verifies any attribute."""
    any_constraint = AnyAttr()
    any_constraint.verify(BoolData(True), {})
    any_constraint.verify(BoolData(False), {})
    any_constraint.verify(IntData(0), {})


@dataclass
class LessThan(AttrConstraint):
    bound: int

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, IntData):
            raise VerifyException(f"{attr} should be of base attribute {IntData.name}")
        if attr.data >= self.bound:
            raise VerifyException(f"{attr} should hold a value less than {self.bound}")


@dataclass
class GreaterThan(AttrConstraint):
    bound: int

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, IntData):
            raise VerifyException(f"{attr} should be of base attribute {IntData.name}")
        if attr.data <= self.bound:
            raise VerifyException(
                f"{attr} should hold a value greater than {self.bound}"
            )


def test_anyof_verify():
    """
    Check that an AnyOf constraint verifies if one of the constraints
    verify.
    """
    constraint = AnyOf([LessThan(0), GreaterThan(10)])
    constraint.verify(IntData(-1), {})
    constraint.verify(IntData(-10), {})
    constraint.verify(IntData(11), {})
    constraint.verify(IntData(100), {})


def test_anyof_verify_fail():
    """
    Check that an AnyOf constraint fails to verify if none of the constraints
    verify.
    """
    constraint = AnyOf([LessThan(0), GreaterThan(10)])

    zero = IntData(0)
    ten = IntData(10)

    with pytest.raises(VerifyException) as e:
        constraint.verify(zero, {})
    assert e.value.args[0] == f"Unexpected attribute {zero}"

    with pytest.raises(VerifyException) as e:
        constraint.verify(ten, {})
    assert e.value.args[0] == f"Unexpected attribute {ten}"


def test_allof_verify():
    """
    Check that an AllOf constraint verifies if all of the constraints
    verify.
    """
    constraint = AllOf([LessThan(10), GreaterThan(0)])
    constraint.verify(IntData(1), {})
    constraint.verify(IntData(9), {})
    constraint.verify(IntData(5), {})


def test_allof_verify_fail():
    """
    Check that an AllOf constraint fails to verify if one of the constraints
    fails to verify.
    """
    constraint = AllOf([LessThan(10), GreaterThan(0)])

    with pytest.raises(VerifyException) as e:
        constraint.verify(IntData(10), {})
    assert e.value.args[0] == f"{IntData(10)} should hold a value less than 10"

    with pytest.raises(VerifyException) as e:
        constraint.verify(IntData(0), {})
    assert e.value.args[0] == f"{IntData(0)} should hold a value greater than 0"


def test_allof_verify_multiple_failures():
    """
    Check that an AllOf constraint provides verification info for all related constraints
    even when one of them fails.
    """
    constraint = AllOf([LessThan(5), GreaterThan(8)])

    with pytest.raises(VerifyException) as e:
        constraint.verify(IntData(7), {})
    assert (
        e.value.args[0]
        == f"The following constraints were not satisfied:\n{IntData(7)} should hold a value less than 5\n{IntData(7)} should hold a value greater than 8"
    )


def test_param_attr_verify():
    bool_true = BoolData(True)
    constraint = ParamAttrConstraint(
        DoubleParamAttr, [EqAttrConstraint(bool_true), BaseAttr(IntData)]
    )
    constraint.verify(DoubleParamAttr([bool_true, IntData(0)]), {})
    constraint.verify(DoubleParamAttr([bool_true, IntData(42)]), {})


def test_param_attr_verify_base_fail():
    bool_true = BoolData(True)
    constraint = ParamAttrConstraint(
        DoubleParamAttr, [EqAttrConstraint(bool_true), BaseAttr(IntData)]
    )
    with pytest.raises(VerifyException) as e:
        constraint.verify(bool_true, {})
    assert e.value.args[0] == (
        f"{bool_true} should be of base attribute {DoubleParamAttr.name}"
    )


def test_param_attr_verify_params_num_params_fail():
    bool_true = BoolData(True)
    constraint = ParamAttrConstraint(DoubleParamAttr, [EqAttrConstraint(bool_true)])
    attr = DoubleParamAttr([bool_true, IntData(0)])
    with pytest.raises(VerifyException) as e:
        constraint.verify(attr, {})
    assert e.value.args[0] == (f"1 parameters expected, but got 2")


def test_param_attr_verify_params_fail():
    bool_true = BoolData(True)
    bool_false = BoolData(False)
    constraint = ParamAttrConstraint(
        DoubleParamAttr, [EqAttrConstraint(bool_true), BaseAttr(IntData)]
    )

    with pytest.raises(VerifyException) as e:
        constraint.verify(DoubleParamAttr([bool_true, bool_false]), {})
    assert e.value.args[0] == (
        f"{bool_false} should be of base attribute {IntData.name}"
    )

    with pytest.raises(VerifyException) as e:
        constraint.verify(DoubleParamAttr([bool_false, IntData(0)]), {})
    assert e.value.args[0] == (f"Expected attribute {bool_true} but got {bool_false}")


def test_constraint_vars_success():
    """Test that VarConstraint verifier succeed when given the same attributes."""

    constraint = VarConstraint("T", AnyOf([BoolData(False), IntData(0)]))

    constraint_vars = {}
    constraint.verify(BoolData(False), constraint_vars)
    constraint.verify(BoolData(False), constraint_vars)

    constraint_vars = {}
    constraint.verify(IntData(0), constraint_vars)
    constraint.verify(IntData(0), constraint_vars)


def test_constraint_vars_fail_different():
    """Test that VarConstraint verifier fails when given different attributes."""

    constraint = VarConstraint("T", AnyOf([BoolData(False), IntData(0)]))

    constraint_vars = {}
    constraint.verify(IntData(0), constraint_vars)

    with pytest.raises(VerifyException):
        constraint.verify(BoolData(False), constraint_vars)


def test_constraint_vars_fail_underlying_constraint():
    """
    Test that VarConstraint verifier fails when given
    attributes that fail the underlying constraint.
    """

    constraint = VarConstraint("T", AnyOf([BoolData(False), IntData(0)]))

    with pytest.raises(VerifyException):
        constraint.verify(IntData(1), {})
