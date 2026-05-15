import re
from collections.abc import Mapping, Sequence

import pytest
from typing_extensions import TypeVar

from xdsl.dialects.builtin import StringAttr, i32
from xdsl.ir import Attribute
from xdsl.irdl import (
    AnyAttr,
    AnyInt,
    AttrConstraint,
    BaseAttr,
    ConstraintContext,
    EqAttrConstraint,
    EqIntConstraint,
    IntConstraint,
    IntTypeVarConstraint,
    IntVarConstraint,
    RangeConstraint,
    RangeLengthConstraint,
    RangeOf,
    RangeVarConstraint,
    SingleOf,
    VarConstraint,
)
from xdsl.utils.exceptions import VerifyException


class AnyRangeConstraint(RangeConstraint):
    """Constraint for testing default infer"""

    def verify(
        self, attrs: Sequence[Attribute], constraint_context: ConstraintContext
    ) -> None:
        return

    def verify_length(self, length: int, constraint_context: ConstraintContext) -> None:
        return

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> RangeConstraint:
        return self


def test_failing_inference():
    with pytest.raises(
        ValueError, match="Cannot infer range from constraint AnyRangeConstraint()"
    ):
        AnyRangeConstraint().infer(ConstraintContext(), length=None)


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
        range_len_constr.verify((hello, world), ConstraintContext())
    range_len_constr.verify((world, world), ConstraintContext())

    with pytest.raises(
        VerifyException,
        match="incorrect length for range variable",
    ):
        range_len_constr.verify(
            (world, world, world), ConstraintContext(_int_variables={"LENGTH": 2})
        )

    # verify_length
    range_len_constr.verify_length(2, ConstraintContext())

    with pytest.raises(
        VerifyException,
        match="incorrect length for range variable",
    ):
        range_len_constr.verify_length(
            3, ConstraintContext(_int_variables={"LENGTH": 2})
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
        ConstraintContext(_variables={"ATTR": world}), length=2
    ) == (
        world,
        world,
    )
    assert range_len_constr.infer(
        ConstraintContext(_variables={"ATTR": world}, _int_variables={"LENGTH": 3}),
        length=None,
    ) == (world, world, world)


def test_of_length():
    range_constraint = RangeOf(AnyAttr())

    range_len_1 = RangeLengthConstraint(range_constraint, EqIntConstraint(2))
    range_len_2 = range_constraint.of_length(EqIntConstraint(2))
    range_len_3 = range_constraint.of_length(2)

    assert range_len_1 == range_len_2
    assert range_len_1 == range_len_3


def test_mapping_type_vars():
    _IntT = TypeVar("_IntT", bound=int, default=int)
    tv_constr = IntTypeVarConstraint(_IntT, AnyInt())
    range_constr = RangeLengthConstraint(AnyRangeConstraint(), tv_constr)
    my_constr = EqIntConstraint(1)
    assert range_constr.mapping_type_vars({_IntT: my_constr}) == RangeLengthConstraint(
        AnyRangeConstraint(), my_constr
    )


def test_init_irdl_constraint():
    range_constr = RangeOf(Attribute)
    assert range_constr.constr == AnyAttr()


def test_empty_range():
    constr = AnyRangeConstraint().of_length(EqIntConstraint(0))

    assert constr.can_infer(set(), length_known=False)

    assert constr.infer(ConstraintContext(), length=None) == ()


def test_range_var_constraint():
    RangeVarConstraint("R", SingleOf(EqAttrConstraint(i32))).verify(
        (i32,), ConstraintContext()
    )
    RangeVarConstraint("R", SingleOf(AnyAttr())).verify(
        (i32,), ConstraintContext({}, {"R": (i32,)})
    )

    with pytest.raises(
        VerifyException,
        match=re.escape(
            "attributes ('i32',) expected from range variable 'R', but got ('i32', 'i32')"
        ),
    ):
        RangeVarConstraint("R", SingleOf(AnyAttr())).verify(
            (i32, i32), ConstraintContext({}, {"R": (i32,)})
        )

    assert (
        RangeVarConstraint("R", SingleOf(AnyAttr())).can_infer(
            set(), length_known=False
        )
        is False
    )
    assert (
        RangeVarConstraint("R", SingleOf(EqAttrConstraint(i32))).can_infer(
            set(), length_known=False
        )
        is True
    )
    assert (
        RangeVarConstraint("R", RangeOf(EqAttrConstraint(i32))).can_infer(
            set(), length_known=True
        )
        is True
    )
    assert (
        RangeVarConstraint("R", RangeOf(EqAttrConstraint(i32))).can_infer(
            set(), length_known=False
        )
        is False
    )
    assert (
        RangeVarConstraint("R", AnyRangeConstraint()).can_infer(
            {"R"}, length_known=True
        )
        is True
    )
    assert (
        RangeVarConstraint("R", AnyRangeConstraint()).can_infer(
            set(), length_known=True
        )
        is False
    )
    assert (
        RangeVarConstraint("R", AnyRangeConstraint()).can_infer(
            {"R"}, length_known=False
        )
        is True
    )
    assert (
        RangeVarConstraint("R", AnyRangeConstraint()).can_infer(
            set(), length_known=False
        )
        is False
    )

    assert RangeVarConstraint("R", SingleOf(EqAttrConstraint(i32))).infer(
        ConstraintContext(), length=None
    ) == (i32,)
    assert RangeVarConstraint("R", RangeOf(EqAttrConstraint(i32))).infer(
        ConstraintContext(), length=3
    ) == (i32, i32, i32)
    assert RangeVarConstraint("R", AnyRangeConstraint()).infer(
        ConstraintContext({}, {"R": (i32, i32)}), length=2
    ) == (i32, i32)
    assert RangeVarConstraint("R", AnyRangeConstraint()).infer(
        ConstraintContext({}, {"R": (i32, StringAttr("str"))}), length=None
    ) == (i32, StringAttr("str"))
