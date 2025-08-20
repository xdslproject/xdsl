import re
from typing import Literal

import pytest
from typing_extensions import TypeVar

from xdsl.irdl import (
    AnyInt,
    AtLeast,
    AtMost,
    ConstraintContext,
    EqIntConstraint,
    IntSetConstraint,
    IntTypeVarConstraint,
    get_int_constraint,
    get_optional_int_constraint,
)
from xdsl.utils.exceptions import PyRDLTypeError, VerifyException


def test_failing_inference():
    with pytest.raises(
        ValueError,
        match=re.escape(r"Cannot infer integer from constraint AtLeast(bound=3)"),
    ):
        AtLeast(3).infer(ConstraintContext())


def test_at_least():
    AtLeast(0).verify(0, ConstraintContext())

    AtLeast(1).verify(1, ConstraintContext())
    AtLeast(1).verify(2, ConstraintContext())
    with pytest.raises(VerifyException):
        AtLeast(1).verify(0, ConstraintContext())

    AtLeast(2).verify(2, ConstraintContext())
    AtLeast(2).verify(3, ConstraintContext())
    with pytest.raises(VerifyException):
        AtLeast(2).verify(1, ConstraintContext())
    with pytest.raises(VerifyException):
        AtLeast(2).verify(0, ConstraintContext())


def test_at_most():
    AtMost(0).verify(0, ConstraintContext())

    AtMost(1).verify(1, ConstraintContext())
    AtMost(1).verify(0, ConstraintContext())
    with pytest.raises(VerifyException, match="Expected integer <= 0, got 1"):
        AtMost(0).verify(1, ConstraintContext())

    AtMost(2).verify(2, ConstraintContext())
    AtMost(2).verify(1, ConstraintContext())
    with pytest.raises(VerifyException, match="Expected integer <= 2, got 3"):
        AtMost(2).verify(3, ConstraintContext())


def test_eq():
    one_constr = EqIntConstraint(1)
    one_constr.verify(1, ConstraintContext())
    with pytest.raises(VerifyException, match="Invalid value 2, expected 1"):
        one_constr.verify(2, ConstraintContext())
    assert one_constr.can_infer(set())
    assert one_constr.infer(ConstraintContext()) == 1


def test_set():
    empty_constr = IntSetConstraint(frozenset())
    assert not empty_constr.can_infer(set())
    with pytest.raises(
        VerifyException, match=re.escape("Invalid value 2, expected one of {}")
    ):
        empty_constr.verify(2, ConstraintContext())

    one_constr = IntSetConstraint(frozenset((0,)))
    one_constr.verify(0, ConstraintContext())
    assert one_constr.can_infer(set())
    assert one_constr.infer(ConstraintContext()) == 0
    with pytest.raises(
        VerifyException, match=re.escape("Invalid value 2, expected one of {0}")
    ):
        one_constr.verify(2, ConstraintContext())

    two_constr = IntSetConstraint(frozenset((0, 1)))
    two_constr.verify(0, ConstraintContext())
    assert not two_constr.can_infer(set())
    with pytest.raises(
        VerifyException, match=re.escape("Invalid value 2, expected one of {0, 1}")
    ):
        two_constr.verify(2, ConstraintContext())


def test_mapping_type_vars():
    _IntT = TypeVar("_IntT", bound=int, default=int)
    tv_constr = IntTypeVarConstraint(_IntT, AnyInt())
    my_constr = EqIntConstraint(1)
    assert tv_constr.mapping_type_vars({_IntT: my_constr}) is my_constr


def test_get_int_constr():
    assert get_int_constraint(int) == AnyInt()
    assert get_int_constraint(Literal[1]) == EqIntConstraint(1)
    assert get_int_constraint(Literal[2]) == EqIntConstraint(2)
    assert get_int_constraint(Literal[2, 3]) == IntSetConstraint(frozenset((2, 3)))

    assert get_int_constraint(
        Literal[2] | Literal[3]  # noqa
    ) == IntSetConstraint(frozenset((2, 3)))

    assert get_int_constraint(
        Literal[1] | Literal[2, 3]  # noqa
    ) == IntSetConstraint(frozenset((1, 2, 3)))
    assert get_int_constraint(
        (Literal[1] | Literal[2]) | Literal[3]  # noqa
    ) == IntSetConstraint(frozenset((1, 2, 3)))
    assert get_int_constraint(
        Literal[1] | (Literal[2] | Literal[3])  # noqa
    ) == IntSetConstraint(frozenset((1, 2, 3)))

    assert get_optional_int_constraint(str) is None

    with pytest.raises(PyRDLTypeError, match="Unexpected int type: <class 'str'>"):
        get_int_constraint(str)  # pyright: ignore[reportArgumentType]
