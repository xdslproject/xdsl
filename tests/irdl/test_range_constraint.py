import re

import pytest
from typing_extensions import TypeVar

from xdsl.dialects.builtin import IntegerType, StringAttr, i32
from xdsl.ir import Attribute
from xdsl.irdl import (
    AnyAttr,
    AnyInt,
    AnyRange,
    BaseAttr,
    EqAttrConstraint,
    EqIntConstraint,
    IntTypeVarConstraint,
    IntVarConstraint,
    RangeLengthConstraint,
    RangeOf,
    RangeVarConstraint,
    VarConstraint,
    VerificationContext,
)
from xdsl.utils.exceptions import VerifyException


def test_failing_inference():
    with pytest.raises(
        ValueError,
        match=re.escape("Cannot infer range from constraint AnyRange()"),
    ):
        AnyRange().infer(VerificationContext(), length=None)


def test_range_of_variables():
    attr_constr = VarConstraint("ATTR", AnyAttr())
    range_constr = RangeOf(attr_constr)
    assert range_constr.variables() == {"ATTR"}
    len_constr = IntVarConstraint("LENGTH", AnyInt())
    assert range_constr.of_length(len_constr).variables() == {"ATTR", "LENGTH"}


def test_verify_range_length_constraint():
    hello = StringAttr("hello")
    world = StringAttr("world")
    attr_constr = VarConstraint("ATTR", BaseAttr(StringAttr))
    len_constr = IntVarConstraint("LENGTH", AnyInt())
    range_len_constr = RangeLengthConstraint(RangeOf(attr_constr), len_constr)
    with pytest.raises(
        VerifyException,
        match='''attribute "hello" expected from variable 'ATTR', but got "world"''',
    ):
        range_len_constr.verify((hello, world), VerificationContext())
    range_len_constr.verify((world, world), VerificationContext())

    with pytest.raises(
        VerifyException,
        match="incorrect length for range variable",
    ):
        range_len_constr.verify(
            (world, world, world), VerificationContext(_int_variables={"LENGTH": 2})
        )

    # verify_length
    range_len_constr.verify_length(2, VerificationContext())

    with pytest.raises(
        VerifyException,
        match="incorrect length for range variable",
    ):
        range_len_constr.verify_length(
            3, VerificationContext(_int_variables={"LENGTH": 2})
        )

    # variables
    assert range_len_constr.variables() == {"ATTR", "LENGTH"}

    # variables_from_length
    assert range_len_constr.variables_from_length() == {"LENGTH"}

    # can_infer
    assert not range_len_constr.can_infer(set(), length_known=True)
    assert not range_len_constr.can_infer(set(), length_known=False)
    assert not range_len_constr.can_infer({"ATTR"}, length_known=False)
    assert range_len_constr.can_infer({"ATTR"}, length_known=True)
    assert range_len_constr.can_infer({"ATTR", "LENGTH"}, length_known=False)

    # infer
    assert range_len_constr.infer(
        VerificationContext(_variables={"ATTR": world}), length=2
    ) == (
        world,
        world,
    )
    assert range_len_constr.infer(
        VerificationContext(_variables={"ATTR": world}, _int_variables={"LENGTH": 3}),
        length=None,
    ) == (world, world, world)


def test_of_length():
    range_constraint = AnyRange()

    range_len_1 = RangeLengthConstraint(range_constraint, EqIntConstraint(2))
    range_len_2 = range_constraint.of_length(EqIntConstraint(2))
    range_len_3 = range_constraint.of_length(2)

    assert range_len_1 == range_len_2
    assert range_len_1 == range_len_3


def test_mapping_type_vars():
    _IntT = TypeVar("_IntT", bound=int, default=int)
    tv_constr = IntTypeVarConstraint(_IntT, AnyInt())
    range_constr = RangeLengthConstraint(AnyRange(), tv_constr)
    my_constr = EqIntConstraint(1)
    assert range_constr.mapping_type_vars({_IntT: my_constr}) == RangeLengthConstraint(
        AnyRange(), my_constr
    )


def test_range_of_get():
    """Test that RangeOf.get(AnyAttr()) is simplified to AnyRange()."""
    # Test with AnyAttr instance
    range_constr = RangeOf.get(AnyAttr())
    assert isinstance(range_constr, AnyRange)

    range_constr3 = RangeOf.get(i32)
    assert isinstance(range_constr3, RangeOf)
    assert isinstance(range_constr3.constr, EqAttrConstraint)


def test_empty_range():
    constr = AnyRange().of_length(EqIntConstraint(0))

    assert constr.can_infer(set(), length_known=False)

    assert constr.infer(VerificationContext(), length=None) == ()


def test_range_var_constraint_verify():
    RangeVarConstraint("R", RangeOf(EqAttrConstraint(i32)).of_length(1)).verify(
        (i32,), VerificationContext()
    )
    RangeVarConstraint("R", AnyRange().of_length(1)).verify(
        (i32,), VerificationContext({}, {"R": (i32,)})
    )

    with pytest.raises(
        VerifyException,
        match=re.escape(
            "attributes ('i32',) expected from range variable 'R', but got ('i32', 'i32')"
        ),
    ):
        RangeVarConstraint("R", AnyRange().of_length(1)).verify(
            (i32, i32), VerificationContext({}, {"R": (i32,)})
        )


@pytest.mark.parametrize(
    "constraint, context_dict, length, inferred",
    [
        (RangeVarConstraint("R", AnyRange().of_length(1)), {}, None, None),
        (
            RangeVarConstraint("R", RangeOf(EqAttrConstraint(i32)).of_length(1)),
            {},
            True,
            (i32,),
        ),
        (RangeVarConstraint("R", RangeOf(EqAttrConstraint(i32))), {}, None, None),
        (RangeVarConstraint("R", RangeOf(EqAttrConstraint(i32))), {}, 2, (i32, i32)),
        (RangeVarConstraint.get("R"), {}, None, None),
        (
            RangeVarConstraint.get("R"),
            {"R": (i32, i32, i32)},
            None,
            (i32, i32, i32),
        ),
        (
            RangeVarConstraint.get("R"),
            {"R": (i32, i32, i32)},
            2,
            (i32, i32, i32),
        ),
        (
            RangeVarConstraint.get("R"),
            {"R": (i32, i32, i32)},
            3,
            (i32, i32, i32),
        ),
        (RangeVarConstraint.get("R"), {}, 3, None),
    ],
)
def test_range_var_constraint_infer(
    constraint: RangeVarConstraint,
    context_dict: dict[str, tuple[Attribute, ...]],
    length: int | None,
    inferred: tuple[Attribute, ...] | None,
) -> None:
    if inferred is None:
        assert not constraint.can_infer(
            context_dict.keys(), length_known=length is not None
        )
    else:
        assert constraint.can_infer(
            context_dict.keys(), length_known=length is not None
        )
        assert (
            constraint.infer(VerificationContext({}, context_dict), length=length)
            == inferred
        )


def test_range_var_constraint_get():
    assert RangeVarConstraint.get("R") == RangeVarConstraint("R", AnyRange())
    assert RangeVarConstraint.get("R", AnyRange()) == RangeVarConstraint(
        "R", AnyRange()
    )
    assert RangeVarConstraint.get("R", AnyRange().of_length(2)) == RangeVarConstraint(
        "R", AnyRange().of_length(2)
    )
    assert RangeVarConstraint.get("R", EqAttrConstraint(i32)) == RangeVarConstraint(
        "R", RangeOf(EqAttrConstraint(i32))
    )
    assert RangeVarConstraint.get("R", i32) == RangeVarConstraint(
        "R", RangeOf(EqAttrConstraint(i32))
    )


def test_range_constraint_get_bases():
    """Test get_bases() method on RangeConstraint subclasses."""
    # AnyRange returns None (any type possible)
    assert AnyRange().get_bases() is None

    # RangeOf delegates to inner constraint
    assert RangeOf(BaseAttr(StringAttr)).get_bases() == {StringAttr}
    assert RangeOf(EqAttrConstraint(i32)).get_bases() == {IntegerType}
    assert RangeOf(AnyAttr()).get_bases() is None

    # RangeLengthConstraint delegates to inner constraint
    range_len = RangeOf(BaseAttr(StringAttr)).of_length(2)
    assert range_len.get_bases() == {StringAttr}

    # RangeVarConstraint delegates to inner constraint
    range_var = RangeVarConstraint("R", RangeOf(BaseAttr(StringAttr)))
    assert range_var.get_bases() == {StringAttr}
