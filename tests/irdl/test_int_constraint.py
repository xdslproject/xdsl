import re

import pytest

from xdsl.irdl import AtLeast, ConstraintContext, EqIntConstraint
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
