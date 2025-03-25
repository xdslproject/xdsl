from abc import ABC

import pytest

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute, Data, ParametrizedAttribute
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AttrConstraint,
    BaseAttr,
    ConstraintContext,
    EqAttrConstraint,
    ParamAttrConstraint,
    ParameterDef,
    VarConstraint,
    eq,
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


def test_param_attr_constraint_inference():
    class BaseWrapAttr(ParametrizedAttribute):
        name = "wrap"

        inner: ParameterDef[Attribute]

    @irdl_attr_definition
    class WrapAttr(BaseWrapAttr): ...

    constr = ParamAttrConstraint(
        WrapAttr,
        (
            eq(
                StringAttr("Hello"),
            ),
        ),
    )

    assert constr.can_infer(set())
    assert constr.infer(ConstraintContext()) == WrapAttr((StringAttr("Hello"),))

    var_constr = ParamAttrConstraint(
        WrapAttr,
        (
            VarConstraint(
                "T",
                eq(
                    StringAttr("Hello"),
                ),
            ),
        ),
    )

    assert var_constr.can_infer({"T"})
    assert var_constr.infer(ConstraintContext({"T": StringAttr("Hello")})) == WrapAttr(
        (StringAttr("Hello"),)
    )

    base_constr = ParamAttrConstraint(
        BaseWrapAttr,
        (
            eq(
                StringAttr("Hello"),
            ),
        ),
    )
    assert not base_constr.can_infer(set())


def test_base_attr_constraint_inference():
    class BaseNoParamAttr(ParametrizedAttribute):
        name = "no_param"

    @irdl_attr_definition
    class WithParamAttr(ParametrizedAttribute):
        name = "with_param"

        inner: ParameterDef[Attribute]

    @irdl_attr_definition
    class DataAttr(Data[int]):
        name = "data"

    @irdl_attr_definition
    class NoParamAttr(BaseNoParamAttr): ...

    constr = BaseAttr(NoParamAttr)

    assert constr.can_infer(set())
    assert constr.infer(ConstraintContext()) == NoParamAttr()

    base_constr = BaseAttr(BaseNoParamAttr)
    assert not base_constr.can_infer(set())

    with_param_constr = BaseAttr(WithParamAttr)
    assert not with_param_constr.can_infer(set())

    data_constr = BaseAttr(DataAttr)
    assert not data_constr.can_infer(set())


@pytest.mark.parametrize(
    "constr, expected",
    [
        (BaseAttr(StringAttr), "BaseAttr(StringAttr)"),
        (
            ParamAttrConstraint(AttrB, (AnyAttr(),)),
            "ParamAttrConstraint(AttrB, (AnyAttr(),))",
        ),
    ],
)
def test_constraint_repr(constr: AttrConstraint, expected: str):
    assert repr(constr) == expected
    assert eval(repr(constr)) == constr
