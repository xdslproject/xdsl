import re
from abc import ABC
from dataclasses import dataclass

import pytest
from typing_extensions import TypeVar

from xdsl.dialects.bufferization import TensorFromMemRefConstraint
from xdsl.dialects.builtin import (
    IndexType,
    IntAttr,
    IntAttrConstraint,
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
    AnyInt,
    AnyOf,
    AtLeast,
    AtMost,
    AttrConstraint,
    BaseAttr,
    ConstraintContext,
    EqAttrConstraint,
    EqIntConstraint,
    IntTypeVarConstraint,
    Not,
    ParamAttrConstraint,
    VarConstraint,
    base,
    eq,
    irdl_attr_definition,
)
from xdsl.utils.exceptions import PyRDLError, VerifyException


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


class Base2(ParametrizedAttribute, ABC):
    pass


@irdl_attr_definition
class AttrA(Base):
    name = "test.attr_a"


@irdl_attr_definition
class AttrB(Base):
    name = "test.attr_b"

    param: AttrA


@irdl_attr_definition
class AttrC(Base):
    name = "test.attr_c"


@irdl_attr_definition
class AttrD(Base):
    name = "test.attr_d"

    param: AttrA | AttrC


@pytest.mark.parametrize(
    "constraint, expected",
    [
        (AnyAttr(), None),
        (EqAttrConstraint(AttrB(AttrA())), {AttrB}),
        (BaseAttr(Base), None),
        (BaseAttr(AttrA), {AttrA}),
        (EqAttrConstraint(AttrB(AttrA())) | BaseAttr(AttrA), {AttrA, AttrB}),
        (
            EqAttrConstraint(AttrD(AttrA())) | EqAttrConstraint(AttrD(AttrC())),
            {AttrD},
        ),
        (AllOf((AnyAttr(), BaseAttr(Base))), None),
        (AllOf((AnyAttr(), BaseAttr(AttrA))), {AttrA}),
        (ParamAttrConstraint(AttrB, [BaseAttr(AttrA)]), {AttrB}),
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
    @dataclass(frozen=True)
    class BaseWrapAttr(ParametrizedAttribute):
        name = "test.wrap"

        inner: Attribute

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
    assert constr.infer(ConstraintContext()) == WrapAttr(StringAttr("Hello"))

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
        StringAttr("Hello")
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

        inner: Attribute

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


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        (
            BaseAttr(AttrA) | BaseAttr(AttrB) | BaseAttr(AttrC),
            AnyOf((BaseAttr(AttrA), BaseAttr(AttrB), BaseAttr(AttrC))),
        ),
        (
            # Note the [Attribute] to provide a supertype of AttrA and AttrB
            BaseAttr[Attribute](AttrA) & BaseAttr(AttrB) & BaseAttr(AttrC),
            AllOf((BaseAttr(AttrA), BaseAttr(AttrB), BaseAttr(AttrC))),
        ),
        (AnyAttr() | BaseAttr(AttrA), AnyAttr()),
        (BaseAttr(AttrA) | AnyAttr(), AnyAttr()),
        (AnyAttr() & BaseAttr(AttrA), BaseAttr(AttrA)),
        # Note the [Attribute] to provide a supertype of AttrA and Attribute
        (BaseAttr[Attribute](AttrA) & AnyAttr(), BaseAttr(AttrA)),
        (BaseAttr(AttrA) | BaseAttr(AttrA), BaseAttr(AttrA)),
        (BaseAttr(AttrA) | BaseAttr(AttrB), AnyOf((BaseAttr(AttrA), BaseAttr(AttrB)))),
        (
            ParamAttrConstraint(AttrB, (BaseAttr(AttrA),))
            | ParamAttrConstraint(AttrB, (BaseAttr(AttrA),)),
            ParamAttrConstraint(AttrB, (BaseAttr(AttrA),)),
        ),
        (
            ParamAttrConstraint(AttrD, (BaseAttr(AttrA),))
            | ParamAttrConstraint(AttrD, (BaseAttr(AttrC),)),
            ParamAttrConstraint(AttrD, (BaseAttr(AttrA) | BaseAttr(AttrC),)),
        ),
        (
            ParamAttrConstraint(AttrB, (BaseAttr(AttrA),))
            | ParamAttrConstraint(AttrA, (BaseAttr(AttrB),)),
            AnyOf(
                (
                    ParamAttrConstraint(AttrB, (BaseAttr(AttrA),)),
                    ParamAttrConstraint(AttrA, (BaseAttr(AttrB),)),
                )
            ),
        ),
    ],
)
def test_constraint_simplification(lhs: AttrConstraint, rhs: AttrConstraint):
    assert lhs == rhs


@pytest.mark.parametrize(
    "c1, c2, msg",
    [
        (
            AnyAttr(),
            BaseAttr(AttrA),
            re.escape(
                "Constraint in `AnyOf` without bases must be a `BaseAttr` of a non-final abstract attribute class, got AnyAttr() instead."
            ),
        ),
        (
            BaseAttr(AttrA) | BaseAttr(AttrB),
            BaseAttr(AttrA),
            re.escape(
                "Constraint BaseAttr(AttrA) shares a base with a non-equality constraint in {AnyOf(attr_constrs=(BaseAttr(AttrA), BaseAttr(AttrB)))} in `AnyOf` constraint."
            ),
        ),
        (
            BaseAttr(AttrA),
            EqAttrConstraint(AttrA()),
            re.escape(
                "Constraint EqAttrConstraint(attr=AttrA()) shares a base with a non-equality constraint in {BaseAttr(AttrA)} in `AnyOf` constraint."
            ),
        ),
        (
            EqAttrConstraint(AttrA()),
            BaseAttr(AttrA),
            re.escape(
                "Non-equality constraint BaseAttr(AttrA) shares a base with a constraint in {EqAttrConstraint(attr=AttrA())} in `AnyOf` constraint."
            ),
        ),
        (
            BaseAttr(Base),
            BaseAttr(Base2),
            re.escape(
                "Cannot form `AnyOf` constraint with both BaseAttr(Base2) and BaseAttr(Base), as they cannot be verified as disjoint."
            ),
        ),
        (
            BaseAttr(Base),
            BaseAttr(AttrA),
            re.escape(
                "Non-equality constraint BaseAttr(AttrA) overlaps with the constraint BaseAttr(Base) in `AnyOf` constraint."
            ),
        ),
        (
            BaseAttr(Base),
            EqAttrConstraint(AttrA()),
            re.escape(
                "Equality constraint EqAttrConstraint(attr=AttrA()) overlaps with the constraint BaseAttr(Base) in `AnyOf` constraint."
            ),
        ),
    ],
)
def test_any_of_overlapping(c1: AttrConstraint, c2: AttrConstraint, msg: str):
    with pytest.raises(PyRDLError, match=msg):
        AnyOf((c1, c2))


@pytest.mark.parametrize(
    "constrs",
    [
        (
            BaseAttr(AttrC),
            BaseAttr(AttrA),
        ),
        (
            EqAttrConstraint(AttrD(AttrA())),
            EqAttrConstraint(AttrD(AttrC())),
        ),
        (
            EqAttrConstraint(AttrD(AttrA())),
            BaseAttr(AttrA),
            BaseAttr(AttrC),
            EqAttrConstraint(AttrD(AttrC())),
        ),
    ],
)
def test_any_of_non_overlapping(constrs: tuple[AttrConstraint, ...]):
    AnyOf(constrs)


def test_mapping_type_vars():
    _IntT = TypeVar("_IntT", bound=int, default=int)
    tv_constr = IntTypeVarConstraint(_IntT, AnyInt())
    int_attr_constr = IntAttrConstraint(tv_constr)
    my_constr = EqIntConstraint(1)
    assert int_attr_constr.mapping_type_vars({_IntT: my_constr}) == IntAttrConstraint(
        my_constr
    )


def test_not_constraint_with_variables():
    """Test that Not constraint validates against containing variables."""

    # Creating a Not constraint with a simple constraint (no variables) should work
    not_five = Not(EqAttrConstraint(IntAttr(5)))

    # Test that it correctly verifies
    ctx = ConstraintContext()
    not_five.verify(IntAttr(3), ctx)  # Should pass - 3 is not 5

    with pytest.raises(VerifyException, match="expected attribute to not satisfy"):
        not_five.verify(IntAttr(5), ctx)  # Should fail - 5 is 5

    # Creating a Not constraint with a variable constraint should raise an error
    with pytest.raises(PyRDLError, match="Not constraint cannot contain variables"):
        Not(VarConstraint("x", IntAttrConstraint(AnyInt())))

    # Test with nested constraint that has variables
    var_constraint = VarConstraint("y", EqAttrConstraint(IntAttr(10)))
    with pytest.raises(PyRDLError, match="Not constraint cannot contain variables"):
        Not(var_constraint)


def test_not_constraint_no_side_effects():
    """
    Ensure Not does not leak context mutations from inner constraints that set
    variables before ultimately failing.
    """

    # Branch 1 (IntAttr): sets 'x' then fails via Eq mismatch
    branch1 = AllOf(
        (
            VarConstraint("x", BaseAttr(IntAttr)),
            EqAttrConstraint(IntAttr(10)),
        )
    )

    # Branch 2 (StringAttr): sets 'y' then fails via Eq mismatch
    branch2 = AllOf(
        (
            VarConstraint("y", BaseAttr(StringAttr)),
            EqAttrConstraint(StringAttr("abc")),
        )
    )

    # Intersection of variables across AnyOf branches is empty, so construction is allowed
    neg = Not(AnyOf((branch1, branch2)))

    ctx = ConstraintContext()

    # Verifying with IntAttr should select branch1, which fails -> Not succeeds
    neg.verify(IntAttr(5), ctx)
    assert ctx.get_variable("x") is None

    # Verifying with StringAttr should select branch2, which fails -> Not succeeds
    neg.verify(StringAttr("def"), ctx)
    assert ctx.get_variable("y") is None


def test_not_constraint_disallowed_in_anyof_bases():
    """Not has unknown bases and should not be allowed inside AnyOf."""

    with pytest.raises(
        PyRDLError,
        match=r"cannot appear in an `AnyOf` constraint as its bases aren't known",
    ):
        AnyOf((Not(EqAttrConstraint(IntAttr(1))), BaseAttr(IntAttr)))


def test_constraint_invert_operator():
    """Test the ~ operator for constraint negation."""

    not_five = ~EqAttrConstraint(IntAttr(5))
    ctx = ConstraintContext()

    # Should pass (3 != 5)
    not_five.verify(IntAttr(3), ctx)

    # Should fail (5 == 5)
    with pytest.raises(VerifyException, match="expected attribute to not satisfy"):
        not_five.verify(IntAttr(5), ctx)


# TODO: separate testcases might be nicer here
def test_not_constraint_use_cases():
    """Test some use cases for the Not constraint."""

    # Use case: Exclude a specific range of values using AtLeast/AtMost
    # "Not between 0 and 10" = "Not (>= 0 AND <= 10)"
    not_small_positive = Not(
        AllOf((IntAttrConstraint(AtLeast(0)), IntAttrConstraint(AtMost(10))))
    )
    ctx = ConstraintContext()

    # Values outside the range should pass
    not_small_positive.verify(IntAttr(-1), ctx)  # Negative is fine
    not_small_positive.verify(IntAttr(11), ctx)  # Above range is fine
    not_small_positive.verify(IntAttr(100), ctx)  # Much larger is fine

    # Values in the range should fail
    for i in range(0, 11):
        with pytest.raises(VerifyException):
            not_small_positive.verify(IntAttr(i), ctx)

    # Use case: Exclude specific attribute types
    not_string = Not(BaseAttr(StringAttr))

    # Non-strings should pass
    not_string.verify(IntAttr(42), ctx)
    not_string.verify(AttrA(), ctx)

    # Strings should fail
    with pytest.raises(VerifyException):
        not_string.verify(StringAttr("hello"), ctx)

    # Use case: Complex logical patterns with AllOf
    # "Must be an integer but not 0" - useful for division denominators
    non_zero_int = AllOf((BaseAttr(IntAttr), Not(EqAttrConstraint(IntAttr(0)))))

    # Non-zero integers should pass
    non_zero_int.verify(IntAttr(1), ctx)
    non_zero_int.verify(IntAttr(-5), ctx)
    non_zero_int.verify(IntAttr(100), ctx)

    # Zero should fail
    with pytest.raises(VerifyException):
        non_zero_int.verify(IntAttr(0), ctx)

    # Non-integers should also fail
    with pytest.raises(VerifyException):
        non_zero_int.verify(StringAttr("not an int"), ctx)
