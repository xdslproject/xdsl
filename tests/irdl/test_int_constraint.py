import re
from typing import Literal

import pytest
from typing_extensions import TypeVar

from xdsl.irdl import (
    AnyInt,
    AtLeast,
    ConstraintContext,
    EqIntConstraint,
    IntSetConstraint,
    IntTypeVarConstraint,
    int_constr,
)
from xdsl.utils.exceptions import VerifyException


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


def test_int_constr():
    assert int_constr(int) == AnyInt()
    assert int_constr(Literal[1]) == EqIntConstraint(1)
    assert int_constr(Literal[2]) == EqIntConstraint(2)
    assert int_constr(Literal[2, 3]) == IntSetConstraint(frozenset((2, 3)))

    assert int_constr(
        Literal[2] | Literal[3]  # noqa
    ) == IntSetConstraint(frozenset((2, 3)))
