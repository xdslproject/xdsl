import re
from typing import Literal

import pytest
from typing_extensions import TypeVar

from xdsl.dialects.builtin import IntAttr, IntAttrConstraint
from xdsl.irdl import (
    AnyInt,
    AtLeast,
    AtMost,
    BaseAttr,
    ConstraintContext,
    EqAttrConstraint,
    EqIntConstraint,
    IntSetConstraint,
    IntTypeVarConstraint,
    IntVarConstraint,
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


def test_int_var_constraint_verify():
    IntVarConstraint("I", EqIntConstraint(1)).verify(1, ConstraintContext())
    IntVarConstraint("I", EqIntConstraint(2)).verify(2, ConstraintContext())
    IntVarConstraint("I", AnyInt()).verify(3, ConstraintContext())
    IntVarConstraint("I", AnyInt()).verify(4, ConstraintContext({}, {}, {"I": 4}))

    with pytest.raises(
        VerifyException, match="integer 2 expected from int variable 'I', but got 1"
    ):
        IntVarConstraint("I", AnyInt()).verify(1, ConstraintContext({}, {}, {"I": 2}))


@pytest.mark.parametrize(
    "constraint, context_dict, inferred",
    [
        (IntVarConstraint("I", AnyInt()), {}, None),
        (IntVarConstraint("I", AnyInt()), {"I": 2}, 2),
        (IntVarConstraint("I", EqIntConstraint(1)), {}, 1),
        (IntVarConstraint("I", EqIntConstraint(1)), {"I": 2}, 2),
    ],
)
def test_int_var_constraint_infer(
    constraint: IntVarConstraint,
    context_dict: dict[str, int],
    inferred: int | None,
) -> None:
    if inferred is None:
        assert not constraint.can_infer(context_dict.keys())
    else:
        assert constraint.can_infer(context_dict.keys())
        assert constraint.infer(ConstraintContext({}, {}, context_dict)) == inferred


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


def test_int_attr_get():
    assert IntAttrConstraint.get() == BaseAttr(IntAttr)
    assert IntAttrConstraint.get(int) == BaseAttr(IntAttr)
    assert IntAttrConstraint.get(1) == EqAttrConstraint(IntAttr(1))
    assert IntAttrConstraint.get(Literal[1, 2]) == IntAttrConstraint(
        IntSetConstraint(frozenset((1, 2)))
    )
    assert IntAttrConstraint.get(AnyInt()) == BaseAttr(IntAttr)
    assert IntAttrConstraint.get(
        IntSetConstraint(frozenset((1, 2)))
    ) == IntAttrConstraint(IntSetConstraint(frozenset((1, 2))))
    assert IntAttrConstraint.get(EqIntConstraint(1)) == EqAttrConstraint(IntAttr(1))
