import re

import pytest

from xdsl.irdl import AtLeast, ConstraintContext
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
