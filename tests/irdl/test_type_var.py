"""Compatibility of native and backported type variables in IRDL."""

import sys
import typing
from collections.abc import Callable
from types import GenericAlias, new_class
from typing import Any, Generic, cast

import pytest
from typing_extensions import NoDefault, TypeForm, TypeVar

from xdsl.dialects.builtin import IndexType, IntegerType, StringAttr, i32
from xdsl.ir import Attribute, ParametrizedAttribute
from xdsl.irdl import (
    AnyAttr,
    AnyInt,
    AttrConstraint,
    BaseAttr,
    IntConstraint,
    IntTypeVarConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    TypeVarConstraint,
    VerificationContext,
    get_int_constraint,
    irdl_attr_definition,
    irdl_op_definition,
    irdl_to_attr_constraint,
    result_def,
)
from xdsl.utils.exceptions import PyRDLTypeError, VerifyException
from xdsl.utils.hints import (
    TypeVarType,
    get_type_var_default,
    get_type_var_mapping_with_defaults,
)


@pytest.fixture(params=[typing.TypeVar, TypeVar], ids=["typing", "extensions"])
def type_var_factory(request: pytest.FixtureRequest) -> Callable[..., typing.TypeVar]:
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            typing.TypeVar,
            marks=pytest.mark.skipif(
                sys.version_info < (3, 13),
                reason="typing.TypeVar defaults require Python 3.13",
            ),
            id="typing",
        ),
        pytest.param(TypeVar, id="extensions"),
    ]
)
def default_type_var_factory(
    request: pytest.FixtureRequest,
) -> Callable[..., typing.TypeVar]:
    return request.param


def generic_attr(*variables: typing.TypeVar) -> type[ParametrizedAttribute]:
    # Construct the Generic base dynamically to exercise both TypeVar providers.
    return irdl_attr_definition(
        new_class(
            "GenericAttr",
            (ParametrizedAttribute, cast(Any, Generic)[variables]),
            exec_body=lambda ns: ns.update(
                name="test.generic_attr",
                __annotations__={f"param{i}": var for i, var in enumerate(variables)},
            ),
        )
    )


def generic_op(variable: typing.TypeVar) -> type[IRDLOperation]:
    return new_class(
        "GenericOp",
        (IRDLOperation, cast(Any, Generic)[variable]),
        exec_body=lambda ns: ns.update(
            name="test.generic_op",
            result=result_def(cast(TypeForm[Attribute], variable)),
        ),
    )


def test_type_var_constraints(type_var_factory: Callable[..., typing.TypeVar]):
    variable = type_var_factory("T", bound=Attribute)
    # The factory constructs a type expression dynamically; the checker cannot
    # infer its bound from the TypeVar object alone.
    expression = cast(TypeForm[Attribute], variable)
    constraint = irdl_to_attr_constraint(expression, allow_type_var=True)
    assert constraint == TypeVarConstraint(variable, AnyAttr())
    assert constraint.mapping_type_vars({variable: BaseAttr(IndexType)}) == BaseAttr(
        IndexType
    )

    with pytest.raises(PyRDLTypeError, match="TypeVar in unexpected context"):
        irdl_to_attr_constraint(expression)
    with pytest.raises(PyRDLTypeError, match="expected to be bound"):
        irdl_to_attr_constraint(
            cast(TypeForm[Attribute], type_var_factory("Unbound")), allow_type_var=True
        )

    integer_variable = type_var_factory("I", bound=int)
    assert get_int_constraint(
        cast(TypeForm[int], integer_variable)
    ) == IntTypeVarConstraint(integer_variable, AnyInt())


def test_mixed_type_var_mapping():
    Native = typing.TypeVar("Native", bound=Attribute)
    Backport = TypeVar("Backport", bound=Attribute, default=IndexType)
    mapping: dict[TypeVarType, AttrConstraint | IntConstraint] = {
        Native: BaseAttr(IndexType),
        Backport: BaseAttr(StringAttr),
    }

    assert TypeVarConstraint(Native, AnyAttr()).mapping_type_vars(mapping) == BaseAttr(
        IndexType
    )
    assert TypeVarConstraint(Backport, AnyAttr()).mapping_type_vars(
        mapping
    ) == BaseAttr(StringAttr)
    assert get_type_var_default(Native) is NoDefault
    assert get_type_var_default(Backport) is IndexType


def test_generic_op_without_default(type_var_factory: Callable[..., typing.TypeVar]):
    variable = type_var_factory("T", bound=Attribute)
    base = generic_op(variable)
    assert get_type_var_mapping_with_defaults(base) == {}

    specialized = irdl_op_definition(
        cast(
            type[IRDLOperation],
            new_class("SpecializedOp", (cast(Any, base)[IndexType],)),
        )
    )
    assert get_type_var_mapping_with_defaults(specialized) == {variable: IndexType}
    specialized(result_types=[IndexType()]).verify()
    with pytest.raises(VerifyException):
        specialized(result_types=[i32]).verify()


def test_generic_op_default(
    default_type_var_factory: Callable[..., typing.TypeVar],
):
    variable = default_type_var_factory("T", bound=Attribute, default=IndexType)
    base = generic_op(variable)
    assert get_type_var_mapping_with_defaults(base) == {variable: IndexType}
    default_op = irdl_op_definition(
        cast(type[IRDLOperation], new_class("DefaultOp", (base,)))
    )
    default_op(result_types=[IndexType()]).verify()
    with pytest.raises(VerifyException):
        default_op(result_types=[i32]).verify()

    specialized = irdl_op_definition(
        cast(
            type[IRDLOperation],
            new_class("SpecializedOp", (cast(Any, base)[IntegerType],)),
        )
    )
    assert get_type_var_mapping_with_defaults(specialized) == {variable: IntegerType}
    specialized(result_types=[i32]).verify()
    with pytest.raises(VerifyException):
        specialized(result_types=[IndexType()]).verify()


def test_generic_attr_default(
    type_var_factory: Callable[..., typing.TypeVar],
    default_type_var_factory: Callable[..., typing.TypeVar],
):
    first = type_var_factory("First", bound=Attribute)
    second = default_type_var_factory("Second", bound=Attribute, default=IndexType)
    attr = generic_attr(first, second)
    # Construct the expression directly so Python does not fill the default
    # before IRDL's omitted-argument handling is exercised.
    constraint = irdl_to_attr_constraint(
        cast(TypeForm[ParametrizedAttribute], GenericAlias(attr, (StringAttr,)))
    )
    assert constraint == ParamAttrConstraint(
        attr, (BaseAttr(StringAttr), BaseAttr(IndexType))
    )
    constraint.verify(attr(StringAttr("x"), IndexType()), VerificationContext())
    with pytest.raises(VerifyException):
        constraint.verify(attr(StringAttr("x"), i32), VerificationContext())

    assert irdl_to_attr_constraint(
        cast(Any, attr)[StringAttr, IntegerType]
    ) == ParamAttrConstraint(attr, (BaseAttr(StringAttr), BaseAttr(IntegerType)))


def test_generic_attr_missing_argument(
    type_var_factory: Callable[..., typing.TypeVar],
):
    first = type_var_factory("First", bound=Attribute)
    second = type_var_factory("Second", bound=Attribute)
    attr = generic_attr(first, second)
    # Generic normally rejects missing arguments before IRDL sees them. Construct
    # the expression directly to test IRDL's own arity check.
    with pytest.raises(
        PyRDLTypeError, match=r"test\.generic_attr expects 2 parameters, got 1"
    ):
        irdl_to_attr_constraint(
            cast(TypeForm[ParametrizedAttribute], GenericAlias(attr, (StringAttr,)))
        )
