import re
from abc import ABC
from dataclasses import dataclass
from typing import Generic

import pytest
from typing_extensions import TypeVar

from xdsl.dialects.bufferization import TensorFromMemRefConstraint
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseArrayBase,
    DenseIntElementsAttr,
    IndexType,
    IntAttr,
    IntAttrConstraint,
    IntegerType,
    MemRefType,
    Signedness,
    SignednessAttr,
    StringAttr,
    TensorType,
    UnrankedMemRefType,
    UnrankedTensorType,
    i32,
    i64,
)
from xdsl.ir import Attribute, Data, ParametrizedAttribute
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AnyInt,
    AnyOf,
    AtLeast,
    AttrConstraint,
    AttrSetConstraint,
    BaseAttr,
    ConstraintContext,
    EqAttrConstraint,
    EqIntConstraint,
    IntSetConstraint,
    IntTypeVarConstraint,
    IntVarConstraint,
    MessageConstraint,
    ParamAttrConstraint,
    SizedConstraint,
    VarConstraint,
    base,
    eq,
    irdl_attr_definition,
    irdl_to_attr_constraint,
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
        (ParamAttrConstraint(AttrB, (BaseAttr(AttrA),)), {AttrB}),
        (ParamAttrConstraint(Base, (BaseAttr(AttrA),)), None),
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
    "sized_attribute",
    [
        ArrayAttr((AttrA(), AttrC())),
        DenseIntElementsAttr.from_list(TensorType(i32, (2,)), (1, 2)),
        DenseArrayBase.from_list(i32, (1, 2)),
    ],
)
def test_sized_constraint(sized_attribute: Attribute):
    constr_passes = SizedConstraint(EqIntConstraint(2))

    constr_passes.verify(sized_attribute, ConstraintContext())

    constr_fails = SizedConstraint(AtLeast(3))

    with pytest.raises(VerifyException, match="expected integer >= 3, got 2"):
        constr_fails.verify(sized_attribute, ConstraintContext())


def test_sized_constraint_ops():
    sized_constraint = SizedConstraint(IntVarConstraint("I", AnyInt()))

    assert sized_constraint.variables() == {"I"}

    I = TypeVar("I")

    type_var_constraint = SizedConstraint(IntTypeVarConstraint(I, AnyInt()))

    assert type_var_constraint.mapping_type_vars(
        {I: EqIntConstraint(2)}
    ) == SizedConstraint(EqIntConstraint(2))


def test_not_sized_constraint():
    constr = SizedConstraint(AnyInt())

    with pytest.raises(VerifyException, match="Expected #test.attr_a to be sized"):
        constr.verify(AttrA(), ConstraintContext())


def test_attr_set_constraint():
    constr = AttrSetConstraint.get(AttrA(), AttrD(AttrA()), AttrD(AttrC()))

    context = ConstraintContext()

    constr.verify(AttrA(), context)
    constr.verify(AttrD(AttrA()), context)
    constr.verify(AttrD(AttrC()), context)

    with pytest.raises(
        VerifyException,
        match="Expected one of #test.attr_a, #test.attr_d<#test.attr_a>, #test.attr_d<#test.attr_c>, but got #test.attr_c",
    ):
        constr.verify(AttrC(), context)

    assert constr.get_bases() == {AttrA, AttrD}
    assert not constr.can_infer(set())
    assert not constr.variables()


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


_T = TypeVar("_T")


class AttrE(ParametrizedAttribute, Generic[_T]):
    param: _T


def test_param_instantiated_generic():
    with pytest.raises(PyRDLError):
        ParamAttrConstraint.get(AttrE[AttrB])


class AttrF(ParametrizedAttribute):
    param1: Attribute
    param2: Attribute


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
        (
            ParamAttrConstraint(AttrD, (BaseAttr(AttrA),))
            | BaseAttr(AttrA)
            | ParamAttrConstraint(AttrD, (BaseAttr(AttrC),)),
            AnyOf(
                (
                    ParamAttrConstraint(AttrD, (BaseAttr(AttrA) | BaseAttr(AttrC),)),
                    BaseAttr(AttrA),
                )
            ),
        ),
        (
            ParamAttrConstraint(AttrF, (BaseAttr(AttrA), BaseAttr(AttrA)))
            | ParamAttrConstraint(AttrF, (BaseAttr(AttrA), BaseAttr(AttrC))),
            ParamAttrConstraint(
                AttrF, (BaseAttr(AttrA), BaseAttr(AttrA) | BaseAttr(AttrC))
            ),
        ),
        (
            ParamAttrConstraint(AttrF, (BaseAttr(AttrA), BaseAttr(AttrA)))
            | ParamAttrConstraint(AttrF, (BaseAttr(AttrC), BaseAttr(AttrA))),
            ParamAttrConstraint(
                AttrF, (BaseAttr(AttrA) | BaseAttr(AttrC), BaseAttr(AttrA))
            ),
        ),
    ],
)
def test_constraint_simplification(lhs: AttrConstraint, rhs: AttrConstraint):
    assert lhs == rhs


def test_param_attr_merge_failure():
    # ParamAttrConstraints as below cannot be merged into a single constraint
    # Therefore the 'any_of' fails
    with pytest.raises(PyRDLError):
        _ = ParamAttrConstraint(
            AttrB, (BaseAttr(AttrA), BaseAttr(AttrA))
        ) | ParamAttrConstraint(AttrB, (BaseAttr(AttrC), BaseAttr(AttrC)))


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
    assert int_attr_constr.mapping_type_vars({_IntT: my_constr}) == EqAttrConstraint(
        IntAttr(1)
    )
    my_constr_2 = IntSetConstraint(frozenset((1, 2)))
    assert int_attr_constr.mapping_type_vars({_IntT: my_constr_2}) == IntAttrConstraint(
        my_constr_2
    )


@pytest.mark.parametrize(
    "constr, expected",
    [
        (ParamAttrConstraint.get(AttrA), EqAttrConstraint(AttrA())),
        (ParamAttrConstraint.get(AttrB, AttrA()), EqAttrConstraint(AttrB(AttrA()))),
        (
            ParamAttrConstraint.get(AttrB, AnyAttr()),
            BaseAttr(AttrB),
        ),
        (
            ParamAttrConstraint.get(AttrF, None, None),
            BaseAttr(AttrF),
        ),
        (
            ParamAttrConstraint.get(AttrF, AttrA, AttrB),
            ParamAttrConstraint(
                AttrF, (irdl_to_attr_constraint(AttrA), irdl_to_attr_constraint(AttrB))
            ),
        ),
        (
            ParamAttrConstraint.get(
                AttrF, None, ParamAttrConstraint.get(AttrF, AttrA, None)
            ),
            ParamAttrConstraint(
                AttrF,
                (AnyAttr(), ParamAttrConstraint(AttrF, (BaseAttr(AttrA), AnyAttr()))),
            ),
        ),
        (
            ParamAttrConstraint.get(
                AttrF, None, ParamAttrConstraint.get(AttrF, None, None)
            ),
            ParamAttrConstraint(AttrF, (AnyAttr(), BaseAttr(AttrF))),
        ),
        (
            ParamAttrConstraint.get(Base, AttrA()),
            ParamAttrConstraint(Base, (EqAttrConstraint(AttrA()),)),
        ),
        (VarConstraint.get("T"), VarConstraint("T", AnyAttr())),
        (VarConstraint.get("T", AttrA), VarConstraint("T", BaseAttr(AttrA))),
        (VarConstraint.get("T", BaseAttr(AttrA)), VarConstraint("T", BaseAttr(AttrA))),
        (AnyOf.get(), AnyOf(())),
        (AnyOf.get(AttrA), BaseAttr(AttrA)),
        (AnyOf.get(AttrA, AttrB), AnyOf((BaseAttr(AttrA), BaseAttr(AttrB)))),
        (
            AttrSetConstraint.get(AttrA(), AttrC()),
            AttrSetConstraint(frozenset((AttrA(), AttrC()))),
        ),
        (AttrSetConstraint.get(AttrA()), EqAttrConstraint(AttrA())),
    ],
)
def test_constraint_get(constr: AttrConstraint, expected: AttrConstraint):
    assert constr == expected


@pytest.mark.parametrize(
    "constr, var_dict, inferred",
    [
        (AnyAttr(), {}, None),
        (VarConstraint("A", AnyAttr()), {}, None),
        (VarConstraint("A", AnyAttr()), {"A": i32}, i32),
        (VarConstraint("A", EqAttrConstraint(i32)), {}, i32),
        (EqAttrConstraint(i32), {}, i32),
        (BaseAttr(type(i32)), {}, None),
        (AnyOf((EqAttrConstraint(i32), EqAttrConstraint(i64))), {}, None),
        (AnyOf((EqAttrConstraint(i32), EqAttrConstraint(i32))), {}, None),
        (
            AllOf(
                (
                    VarConstraint("A", AnyAttr()),
                    EqAttrConstraint(i32),
                )
            ),
            {},
            i32,
        ),
        (
            AllOf(
                (
                    VarConstraint("A", AnyAttr()),
                    EqAttrConstraint(i32),
                )
            ),
            {"A": i64},
            i64,
        ),
        (ParamAttrConstraint(IntegerType, (AnyAttr(), AnyAttr())), {}, None),
        (
            ParamAttrConstraint(
                IntegerType,
                (
                    EqAttrConstraint(IntAttr(32)),
                    EqAttrConstraint(SignednessAttr(Signedness.SIGNLESS)),
                ),
            ),
            {},
            i32,
        ),
        (MessageConstraint(EqAttrConstraint(i32), "msg"), {}, i32),
    ],
)
def test_constraint_inference(
    constr: AttrConstraint, var_dict: dict[str, Attribute], inferred: Attribute | None
) -> None:
    if inferred is None:
        assert not constr.can_infer(var_dict.keys())
    else:
        assert constr.can_infer(var_dict.keys())
        assert constr.infer(ConstraintContext(var_dict)) == inferred


@pytest.mark.parametrize(
    "constr1, constr2, result",
    [
        (AnyAttr(), AnyAttr(), AnyAttr()),
        (
            VarConstraint("A", AnyAttr()),
            VarConstraint("A", AnyAttr()),
            VarConstraint("A", AnyAttr()),
        ),
        (VarConstraint("A", AnyAttr()), VarConstraint("B", AnyAttr()), None),
        (BaseAttr(AttrB), BaseAttr(AttrB), BaseAttr(AttrB)),
        (BaseAttr(AttrB), BaseAttr(AttrA), None),
        (BaseAttr(AttrB), ParamAttrConstraint(AttrB, (AnyAttr(),)), BaseAttr(AttrB)),
        (ParamAttrConstraint(AttrB, (AnyAttr(),)), BaseAttr(AttrB), BaseAttr(AttrB)),
        (ParamAttrConstraint.get(AttrD, AttrA), BaseAttr(AttrB), None),
        (BaseAttr(AttrB), ParamAttrConstraint.get(AttrD, AttrA), None),
        (ParamAttrConstraint.get(AttrD, AttrA), BaseAttr(AttrD), BaseAttr(AttrD)),
        (BaseAttr(AttrD), ParamAttrConstraint.get(AttrD, AttrA), BaseAttr(AttrD)),
        (
            ParamAttrConstraint.get(AttrD, AttrA),
            ParamAttrConstraint.get(AttrD, AttrC),
            ParamAttrConstraint.get(AttrD, AttrA | AttrC),
        ),
        (
            ParamAttrConstraint.get(AttrF, AttrA, AttrA),
            ParamAttrConstraint.get(AttrF, AttrA, AttrC),
            ParamAttrConstraint.get(AttrF, AttrA, AttrA | AttrC),
        ),
        (
            ParamAttrConstraint.get(AttrF, AttrA, AttrA),
            ParamAttrConstraint.get(AttrF, AttrC, AttrA),
            ParamAttrConstraint.get(AttrF, AttrA | AttrC, AttrA),
        ),
        (
            ParamAttrConstraint.get(AttrF, AttrA, AttrA),
            ParamAttrConstraint.get(AttrF, AttrC, AttrC),
            None,
        ),
        (
            ParamAttrConstraint.get(AttrD, AttrA),
            ParamAttrConstraint.get(AttrF, AttrC, AttrC),
            None,
        ),
        (
            ParamAttrConstraint.get(AttrD, AttrA),
            VarConstraint("A", AnyAttr()),
            None,
        ),
    ],
)
def test_relax_constaint(
    constr1: AttrConstraint, constr2: AttrConstraint, result: AttrConstraint | None
):
    assert constr1.relax_constraint(constr2) == result
