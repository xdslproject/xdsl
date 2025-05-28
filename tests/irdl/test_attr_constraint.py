import re
from abc import ABC

import pytest

from xdsl.dialects.bufferization import TensorFromMemRefConstraint
from xdsl.dialects.builtin import (
    IndexType,
    IntegerType,
    MemRefType,
    StringAttr,
    TensorType,
    UnrankedMemRefType,
    UnrankedTensorType,
    i32,
)
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
    base,
    eq,
    irdl_attr_definition,
)


def test_failing_inference():
    with pytest.raises(
        ValueError, match="Cannot infer attribute from constraint AnyAttr()"
    ):
        AnyAttr().infer(ConstraintContext())

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"Cannot infer attribute from constraint AnyOf(attr_constrs=(BaseAttr(IntegerType), BaseAttr(IndexType)))"
        ),
    ):
        (base(IntegerType) | base(IndexType)).infer(ConstraintContext())


class Base(ParametrizedAttribute, ABC):
    pass


@irdl_attr_definition
class AttrA(Base):
    name = "test.attr_a"


@irdl_attr_definition
class AttrB(Base):
    name = "test.attr_b"

    param: ParameterDef[AttrA]


@irdl_attr_definition
class AttrC(Base):
    name = "test.attr_c"


@pytest.mark.parametrize(
    "constraint, expected",
    [
        (AnyAttr(), None),
        (EqAttrConstraint(AttrB([AttrA()])), {AttrB}),
        (BaseAttr(Base), None),
        (BaseAttr(AttrA), {AttrA}),
        (EqAttrConstraint(AttrB([AttrA()])) | AnyAttr(), None),
        (EqAttrConstraint(AttrB([AttrA()])) | BaseAttr(AttrA), {AttrA, AttrB}),
        (EqAttrConstraint(AttrB([AttrA()])) | BaseAttr(AttrB), {AttrB}),
        (AllOf((AnyAttr(), BaseAttr(Base))), None),
        (AllOf((AnyAttr(), BaseAttr(AttrA))), {AttrA}),
        (ParamAttrConstraint(AttrA, [BaseAttr(AttrB)]), {AttrA}),
        (ParamAttrConstraint(Base, [BaseAttr(AttrA)]), None),
        (VarConstraint("T", BaseAttr(Base)), None),
        (VarConstraint("T", BaseAttr(AttrA)), {AttrA}),
        (
            AllOf(
                (BaseAttr(AttrA) | BaseAttr(AttrB), BaseAttr(AttrB) | BaseAttr(AttrC))
            ),
            {AttrB},
        ),
    ],
)
def test_attr_constraint_get_bases(
    constraint: AttrConstraint, expected: set[type[Attribute]] | None
):
    assert constraint.get_bases() == expected


def test_param_attr_constraint_inference():
    class BaseWrapAttr(ParametrizedAttribute):
        name = "test.wrap"

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
        name = "test.no_param"

    @irdl_attr_definition
    class WithParamAttr(ParametrizedAttribute):
        name = "test.with_param"

        inner: ParameterDef[Attribute]

    @irdl_attr_definition
    class DataAttr(Data[int]):
        name = "test.data"

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


@pytest.mark.parametrize(
    "input, output",
    [
        (TensorType(i32, [2, 2]), MemRefType(i32, [2, 2])),
        (UnrankedTensorType(i32), UnrankedMemRefType.from_type(i32)),
    ],
)
def test_tensor_to_memref(
    input: TensorType | UnrankedTensorType, output: MemRefType | UnrankedMemRefType
):
    assert TensorFromMemRefConstraint.tensor_to_memref(input) == output


@pytest.mark.parametrize(
    "input, output",
    [
        (MemRefType(i32, [2, 2]), TensorType(i32, [2, 2])),
        (UnrankedMemRefType.from_type(i32), UnrankedTensorType(i32)),
    ],
)
def test_memref_to_tensor(
    input: MemRefType | UnrankedMemRefType, output: TensorType | UnrankedTensorType
):
    assert TensorFromMemRefConstraint.memref_to_tensor(input) == output
