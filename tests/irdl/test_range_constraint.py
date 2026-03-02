from collections.abc import Mapping, Sequence

import pytest
from typing_extensions import TypeVar

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute
from xdsl.irdl import (
    AnyAttr,
    AnyInt,
    AttrConstraint,
    BaseAttr,
    ConstraintContext,
    EqIntConstraint,
    IntConstraint,
    IntTypeVarConstraint,
    IntVarConstraint,
    RangeConstraint,
    RangeLengthConstraint,
    RangeOf,
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
