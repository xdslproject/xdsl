from abc import ABC

import pytest

from xdsl.ir import Attribute, ParametrizedAttribute
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AttrConstraint,
    BaseAttr,
    EqAttrConstraint,
    ParamAttrConstraint,
    ParameterDef,
    VarConstraint,
    irdl_attr_definition,
)


class Base(ParametrizedAttribute, ABC):
    pass


@irdl_attr_definition
class AttrA(Base):
    name = "test.attr_a"


@irdl_attr_definition
class AttrB(Base):
    name = "test.attr_b"

    param: ParameterDef[AttrA]


@pytest.mark.parametrize(
    "constraint, expected",
    [
        (AnyAttr(), None),
        (EqAttrConstraint(AttrB([AttrA()])), AttrB),
        (BaseAttr(Base), None),
        (BaseAttr(AttrA), AttrA),
        (EqAttrConstraint(AttrB([AttrA()])) | AnyAttr(), None),
        (EqAttrConstraint(AttrB([AttrA()])) | BaseAttr(AttrA), None),
        (EqAttrConstraint(AttrB([AttrA()])) | BaseAttr(AttrB), AttrB),
        (AllOf((AnyAttr(), BaseAttr(Base))), None),
        (AllOf((AnyAttr(), BaseAttr(AttrA))), AttrA),
        (ParamAttrConstraint(AttrA, [BaseAttr(AttrB)]), AttrA),
        (ParamAttrConstraint(Base, [BaseAttr(AttrA)]), None),
        (VarConstraint("T", BaseAttr(Base)), None),
        (VarConstraint("T", BaseAttr(AttrA)), AttrA),
    ],
)
def test_attr_constraint_get_unique_base(
    constraint: AttrConstraint, expected: type[Attribute] | None
):
    assert constraint.get_unique_base() == expected
