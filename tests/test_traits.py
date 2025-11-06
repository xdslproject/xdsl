"""
Test the definition and usage of traits and interfaces.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from xdsl.dialects import test
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AnyTensorTypeConstr,
    AnyUnrankedMemRefTypeConstr,
    AnyUnrankedTensorTypeConstr,
    IntegerAttr,
    IntegerType,
    MemRefType,
    NoneAttr,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    TensorType,
    UnrankedTensorType,
    i1,
    i32,
    i64,
)
from xdsl.ir import Attribute, Operation, OpTrait, SSAValue
from xdsl.irdl import (
    Block,
    IRDLOperation,
    Region,
    attr_def,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    opt_attr_def,
    opt_region_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
    var_successor_def,
)
from xdsl.traits import (
    AlwaysSpeculatable,
    ConditionallySpeculatable,
    HasAncestor,
    HasParent,
    IsTerminator,
    MemoryEffectKind,
    OptionalSymbolOpInterface,
    RecursivelySpeculatable,
    ReturnLike,
    SameOperandsAndResultType,
    SymbolOpInterface,
    SymbolTable,
    has_effects,
    is_speculatable,
)
from xdsl.utils.exceptions import PyRDLOpDefinitionError, VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.test_value import create_ssa_value


@dataclass(frozen=True)
class LargerResultTrait(OpTrait):
    """Check that the only result has a larger bitwidth than the operand."""

    def verify(self, op: Operation) -> None:
        # This function is never called in this test
        raise NotImplementedError()


@dataclass(frozen=True)
class LargerOperandTrait(OpTrait):
    """Check that the only operand has a larger bitwidth than the result."""

    def verify(self, op: Operation):
        # These asserts should be exceptions in a non-testing environment.
        assert len(op.results) == 1
        assert len(op.operands) == 1
        assert isa(op.results[0].type, IntegerType)
        assert isa(op.operands[0].type, IntegerType)
        if op.results[0].type.width.data >= op.operands[0].type.width.data:
            raise VerifyException(
                "Operation has a result bitwidth greater "
                "or equal to the operand bitwidth."
            )


@dataclass(frozen=True)
class BitwidthSumLessThanTrait(OpTrait):
    """
    Check that the sum of the bitwidths of the
    operands and results is less than a given value.
    """

    parameters: int

    @property
    def max_sum(self):
        return self.parameters

    def verify(self, op: Operation):
        sum_bitwidth = 0
        for operand in op.operands:
            # This assert should be an exception in a non-testing environment.
            assert isa(operand.type, IntegerType)
            sum_bitwidth += operand.type.width.data
        for result in op.results:
            # This assert should be an exception in a non-testing environment.
            assert isa(result.type, IntegerType)
            sum_bitwidth += result.type.width.data

        if sum_bitwidth >= self.max_sum:
            raise VerifyException(
                f"Operation has a bitwidth sum greater or equal to {self.max_sum}."
            )


@irdl_op_definition
class TestOp(IRDLOperation):
    name = "test.test"
    traits = traits_def(LargerOperandTrait(), BitwidthSumLessThanTrait(64))

    ops = operand_def(IntegerType)
    res = result_def(IntegerType)


def test_has_trait_object():
    """
    Test the `has_trait` `Operation` method on a simple operation definition.
    """
    assert TestOp.has_trait(LargerOperandTrait)
    assert not TestOp.has_trait(LargerResultTrait)
    assert not TestOp.has_trait(BitwidthSumLessThanTrait(0))
    assert TestOp.has_trait(BitwidthSumLessThanTrait(64))


def test_get_traits_of_type():
    """
    Test the `get_traits_of_type` `Operation` method
    on a simple operation definition.
    """
    assert TestOp.get_traits_of_type(LargerOperandTrait) == [LargerOperandTrait()]
    assert TestOp.get_traits_of_type(LargerResultTrait) == []
    assert TestOp.get_traits_of_type(BitwidthSumLessThanTrait) == [
        BitwidthSumLessThanTrait(64)
    ]


def test_verifier():
    """
    Check that the traits verifier are correctly called.
    """
    operand64 = create_ssa_value(i64)
    operand32 = create_ssa_value(i32)
    operand1 = create_ssa_value(i1)
    op = TestOp.create(operands=[operand1], result_types=[i32])

    message = (
        "Operation has a result bitwidth greater or equal to the operand bitwidth."
    )
    with pytest.raises(VerifyException, match=message):
        op.verify()

    op = TestOp.create(operands=[operand64], result_types=[i32])
    with pytest.raises(
        VerifyException, match="Operation has a bitwidth sum greater or equal to 64."
    ):
        op.verify()

    op = TestOp.create(operands=[operand32], result_types=[i1])
    op.verify()


def test_verifier_order():
    """
    Check that trait verifiers are called after IRDL verifiers.
    """
    op = TestOp.create(operands=[], result_types=[i1])
    with pytest.raises(VerifyException, match="Expected 1 operand, but got 0"):
        op.verify()


class LargerOperandOp(IRDLOperation, ABC):
    traits = traits_def(LargerOperandTrait())


@irdl_op_definition
class TestCopyOp(LargerOperandOp):
    name = "test.test_copy"

    traits = traits_def(BitwidthSumLessThanTrait(64))


def test_trait_inheritance():
    """
    Check that traits are correctly inherited from parent classes.
    """
    assert TestCopyOp.traits.traits == frozenset(
        (
            LargerOperandTrait(),
            BitwidthSumLessThanTrait(64),
        )
    )


@irdl_op_definition
class NoTraitsOp(IRDLOperation):
    name = "test.no_traits_op"


def test_traits_undefined():
    """Check that traits are defaulted to the empty set."""
    assert NoTraitsOp.traits == traits_def()


class WrongTraitsType(IRDLOperation):
    name = "test.no_traits"

    traits = 1  # pyright: ignore[reportAssignmentType]


def test_traits_wrong_type():
    with pytest.raises(
        PyRDLOpDefinitionError,
        match=(
            "pyrdl operation definition 'WrongTraitsType' traits field should be an "
            "instance of'OpTraits'."
        ),
    ):
        irdl_op_definition(WrongTraitsType)


class GetNumResultsTrait(OpTrait):
    """
    An example of an MLIR interface, using traits.
    It provides a method to get the number of results of an operation.
    """

    # This is the default implementation of a trait (interface) method.
    @staticmethod
    def get_num_results(op: Operation):
        """
        Get the number of results of the operation
        """
        return len(op.results)


class GetNumResultsTraitForOpWithOneResult(GetNumResultsTrait):
    """
    Instance of the trait for the case where an operation has only a single result.
    """

    @staticmethod
    def get_num_results(op: Operation):
        return 1


@irdl_op_definition
class HasInterfaceOp(IRDLOperation):
    name = "test.op_with_interface"
    traits = traits_def(GetNumResultsTraitForOpWithOneResult())

    res = result_def(IntegerType)


def test_interface():
    """
    Test the features of a trait with methods (An MLIR interface).
    """
    op = HasInterfaceOp.create(result_types=(i32,))
    trait = HasInterfaceOp.get_trait(GetNumResultsTrait)
    assert trait is not None
    assert 1 == trait.get_num_results(op)


def test_get_trait_specialized():
    """
    Test get_trait and has_trait in the case where the trait is a child class of the
    trait we want.
    """
    assert HasInterfaceOp.has_trait(GetNumResultsTrait)
    assert HasInterfaceOp.has_trait(GetNumResultsTraitForOpWithOneResult)
    assert (
        HasInterfaceOp.get_trait(GetNumResultsTrait)
        == GetNumResultsTraitForOpWithOneResult()
    )
    assert HasInterfaceOp.get_traits_of_type(GetNumResultsTrait) == [
        GetNumResultsTraitForOpWithOneResult()
    ]


def test_symbol_op_interface():
    """
    Test that operations that conform to SymbolOpInterface have necessary attributes.
    """

    @irdl_op_definition
    class NoSymNameOp(IRDLOperation):
        name = "no_sym_name"
        traits = traits_def(SymbolOpInterface())

    op0 = NoSymNameOp()

    with pytest.raises(
        VerifyException, match='Operation no_sym_name must have a "sym_name" attribute'
    ):
        op0.verify()

    @irdl_op_definition
    class SymNameWrongTypeOp(IRDLOperation):
        name = "wrong_sym_name_type"

        sym_name = attr_def(IntegerAttr)
        traits = traits_def(SymbolOpInterface())

    op1 = SymNameWrongTypeOp(
        attributes={"sym_name": IntegerAttr.from_int_and_width(1, 32)}
    )

    with pytest.raises(
        VerifyException,
        match='Operation wrong_sym_name_type must have a "sym_name" attribute',
    ):
        op1.verify()

    @irdl_op_definition
    class SymNameOp(IRDLOperation):
        name = "sym_name"

        sym_name = attr_def(SymbolNameConstraint())
        traits = traits_def(SymbolOpInterface())

    op2 = SymNameOp(attributes={"sym_name": StringAttr("symbol_name")})
    op2.verify()


def test_optional_symbol_op_interface():
    """
    Test that operations that conform to OptionalSymbolOpInterface have the necessary attributes.
    """

    @irdl_op_definition
    class OptionalSymNameOp(IRDLOperation):
        name = "no_sym_name"

        sym_name = opt_attr_def(StringAttr)

        traits = traits_def(OptionalSymbolOpInterface())

    no_symbol = OptionalSymNameOp()
    interface = no_symbol.get_trait(SymbolOpInterface)
    assert interface is not None
    assert interface.is_optional_symbol(no_symbol)
    no_symbol.verify()
    assert interface.get_sym_attr_name(no_symbol) is None

    symbol = OptionalSymNameOp(attributes={"sym_name": StringAttr("main")})
    interface = symbol.get_trait(SymbolOpInterface)
    assert interface is not None
    assert interface.is_optional_symbol(symbol)
    symbol.verify()
    assert interface.get_sym_attr_name(symbol) == StringAttr("main")


@irdl_op_definition
class SymbolOp(IRDLOperation):
    name = "test.symbol"

    sym_name = attr_def(SymbolNameConstraint())

    traits = traits_def(SymbolOpInterface())

    def __init__(self, name: str):
        return super().__init__(attributes={"sym_name": StringAttr(name)})


@irdl_op_definition
class PropSymbolOp(IRDLOperation):
    name = "test.symbol"

    sym_name = prop_def(SymbolNameConstraint())

    traits = traits_def(SymbolOpInterface())

    def __init__(self, name: str):
        return super().__init__(properties={"sym_name": StringAttr(name)})


@pytest.mark.parametrize("SymbolOp", [SymbolOp, PropSymbolOp])
def test_symbol_table(SymbolOp: type[PropSymbolOp | SymbolOp]):
    # Some helper classes
    @irdl_op_definition
    class SymbolTableOp(IRDLOperation):
        name = "test.symbol_table"

        sym_name = opt_attr_def(StringAttr)

        one = region_def()
        two = opt_region_def()

        traits = traits_def(SymbolTable(), OptionalSymbolOpInterface())

    # Check that having a single region is verified
    op = SymbolTableOp(regions=[Region(), Region()])
    with pytest.raises(
        VerifyException,
        match="Operations with a 'SymbolTable' must have exactly one region",
    ):
        op.verify()

    # Check that having a single block is verified
    blocks = [Block(), Block()]
    op = SymbolTableOp(regions=[Region(blocks), []])
    with pytest.raises(
        VerifyException,
        match="Operations with a 'SymbolTable' must have exactly one block",
    ):
        op.verify()

    terminator = test.TestTermOp()

    # Check that symbol uniqueness is verified
    symbol = SymbolOp("name")
    dup_symbol = SymbolOp("name")
    op = SymbolTableOp(
        regions=[Region(Block([symbol, dup_symbol, terminator.clone()])), []]
    )
    with pytest.raises(
        VerifyException,
        match='Redefinition of symbol "name"',
    ):
        op.verify()

    # Check a flat happy case, with symbol lookup
    symbol = SymbolOp("name")

    op = SymbolTableOp(regions=[Region(Block([symbol, terminator.clone()])), []])
    op.verify()

    assert SymbolTable.lookup_symbol(op, "name") is symbol
    assert SymbolTable.lookup_symbol(op, "that_other_name") is None

    # Check a nested happy case, with symbol lookup
    symbol = SymbolOp("name")

    nested = SymbolTableOp(
        regions=[Region(Block([symbol, terminator.clone()])), []],
        attributes={"sym_name": StringAttr("nested")},
    )
    op = SymbolTableOp(regions=[Region(Block([nested, terminator.clone()])), []])
    op.verify()

    assert SymbolTable.lookup_symbol(op, "name") is None
    assert SymbolTable.lookup_symbol(op, SymbolRefAttr("nested", ["name"])) is symbol


@irdl_op_definition
class HasLazyParentOp(IRDLOperation):
    """An operation with traits that are defined "lazily"."""

    name = "test.has_lazy_parent"

    traits = lazy_traits_def(lambda: (HasParent(TestOp),))


def test_lazy_parent():
    """Test the trait infrastructure for an operation that defines a trait "lazily"."""
    op = HasLazyParentOp.create()
    assert len(op.get_traits_of_type(HasParent)) != 0
    assert op.get_traits_of_type(HasParent)[0].op_types == (TestOp,)
    assert op.has_trait(HasParent(TestOp))
    assert op.traits == traits_def(HasParent(TestOp))


@irdl_op_definition
class AncestorOp(IRDLOperation):
    name = "test.ancestor"

    traits = traits_def(HasAncestor(TestOp))


def test_has_ancestor():
    op = AncestorOp()

    assert op.get_traits_of_type(HasAncestor) == [HasAncestor(TestOp)]
    assert op.has_trait(HasAncestor(TestOp))

    with pytest.raises(
        VerifyException, match="'test.ancestor' expects ancestor op 'test.test'"
    ):
        op.verify()


def test_insert_or_update():
    @irdl_op_definition
    class SymbolTableOp(IRDLOperation):
        name = "test.symbol_table"

        reg = region_def()

        traits = traits_def(SymbolTable())

    # Check a flat happy case, with symbol lookup
    symbol = SymbolOp("name")
    symbol2 = SymbolOp("name2")
    terminator = test.TestTermOp()

    op = SymbolTableOp(regions=[Region(Block([symbol, terminator]))])
    op.verify()

    trait = op.get_trait(SymbolTable)
    assert trait is not None

    assert trait.insert_or_update(op, symbol.clone()) is symbol
    assert len(op.reg.ops) == 2

    assert trait.insert_or_update(op, symbol2) is None
    assert len(op.reg.ops) == 3
    assert symbol2 in list(op.reg.ops)


def nonpure():
    return TestOp.create()


def pure():
    return test.TestPureOp.create()


@pytest.mark.parametrize(
    ("trait", "speculatability", "nested_ops"),
    [
        ([], False, []),
        ([], False, [nonpure()]),
        ([], False, [pure()]),
        ([AlwaysSpeculatable()], True, []),
        ([AlwaysSpeculatable()], True, [nonpure()]),
        ([AlwaysSpeculatable()], True, [pure()]),
        ([RecursivelySpeculatable()], True, []),
        ([RecursivelySpeculatable()], False, [nonpure()]),
        ([RecursivelySpeculatable()], True, [pure()]),
    ],
)
def test_speculability(
    trait: tuple[ConditionallySpeculatable] | tuple[()],
    speculatability: bool,
    nested_ops: Sequence[Operation],
):
    @irdl_op_definition
    class SupeculatabilityTestOp(IRDLOperation):
        name = "test.speculatability"
        region = region_def()

        traits = traits_def(*trait)

    op = SupeculatabilityTestOp(regions=[Region(Block(nested_ops))])
    optrait = op.get_trait(ConditionallySpeculatable)

    if trait:
        assert optrait is not None
        assert optrait.is_speculatable(op) is speculatability
    else:
        assert optrait is None

    assert is_speculatable(op) is speculatability


@pytest.mark.parametrize(
    ("operands", "result_types"),
    [
        ([()], [()]),
        ([()], (test.TestType("foo"),)),
        ((create_ssa_value(test.TestType("foo")),), [()]),
    ],
)
def test_same_operands_and_result_type_trait_for_scalar_types(
    operands: tuple[SSAValue] | tuple[()],
    result_types: tuple[test.TestType] | tuple[()],
):
    @irdl_op_definition
    class SameOperandsAndResultTypeOp(IRDLOperation):
        name = "test.same_operand_and_result_type"

        ops = var_operand_def(test.TestType("foo"))
        res = var_result_def(test.TestType("foo"))

        traits = traits_def(SameOperandsAndResultType())

    op = SameOperandsAndResultTypeOp(operands=operands, result_types=result_types)

    with pytest.raises(
        VerifyException, match="requires at least one result or operand"
    ):
        op.verify()


@irdl_op_definition
class SameOperandsAndResultTypeOp(IRDLOperation):
    name = "test.same_operand_and_result_type"

    ops = var_operand_def(
        MemRefType.constr()
        | AnyUnrankedMemRefTypeConstr
        | AnyUnrankedTensorTypeConstr
        | AnyTensorTypeConstr
    )

    res = var_result_def(
        MemRefType.constr()
        | AnyUnrankedMemRefTypeConstr
        | AnyUnrankedTensorTypeConstr
        | AnyTensorTypeConstr
    )

    traits = traits_def(SameOperandsAndResultType())


@pytest.mark.parametrize(
    (
        "operand1_and_result_element_type",
        "operand_and_result_shape1",
        "result_element_type2",
        "result_shape2",
    ),
    [
        (
            test.TestType("foo"),
            [2, 3],
            test.TestType("qux"),
            [2, 3],
        ),
        (
            test.TestType("foo"),
            [2, 3],
            test.TestType("foo"),
            [2, 4],
        ),
        (
            test.TestType("qux"),
            [2, 3],
            test.TestType("foo"),
            [2, 3],
        ),
        (
            test.TestType("foo"),
            [2, 4],
            test.TestType("foo"),
            [2, 3],
        ),
    ],
)
def test_same_operands_and_result_type_trait_for_result_element_type_of_shaped_types(
    operand1_and_result_element_type: Attribute,
    operand_and_result_shape1: tuple[int],
    result_element_type2: Attribute,
    result_shape2: tuple[int],
):
    op = SameOperandsAndResultTypeOp(
        operands=[
            create_ssa_value(
                TensorType(operand1_and_result_element_type, operand_and_result_shape1)
            )
        ],
        result_types=[
            [
                TensorType(operand1_and_result_element_type, operand_and_result_shape1),
                TensorType(result_element_type2, result_shape2),
            ],
        ],
    )

    with pytest.raises(
        VerifyException,
        match="requires the same type for all operands and results",
    ):
        op.verify()


@pytest.mark.parametrize(
    "operands_num",
    [1, 2, 3],
)
@pytest.mark.parametrize(
    "results_num",
    [1, 2, 3],
)
@pytest.mark.parametrize(
    (
        "operand_type",
        "result_type",
    ),
    [
        (
            TensorType(test.TestType("foo"), [2, 3]),
            TensorType(test.TestType("qux"), [2, 3]),
        ),
        (
            TensorType(test.TestType("foo"), [2, 3]),
            TensorType(test.TestType("foo"), [2, 4]),
        ),
        (
            TensorType(test.TestType("qux"), [2, 3]),
            TensorType(test.TestType("foo"), [2, 3]),
        ),
        (
            TensorType(test.TestType("foo"), [2, 4]),
            TensorType(test.TestType("foo"), [2, 3]),
        ),
        (
            MemRefType(test.TestType("foo"), [2, 3]),
            MemRefType(test.TestType("qux"), [2, 3]),
        ),
        (
            MemRefType(test.TestType("foo"), [2, 3]),
            MemRefType(test.TestType("foo"), [2, 4]),
        ),
        (
            MemRefType(test.TestType("qux"), [2, 3]),
            MemRefType(test.TestType("foo"), [2, 3]),
        ),
        (
            MemRefType(test.TestType("foo"), [2, 4]),
            MemRefType(test.TestType("foo"), [2, 3]),
        ),
    ],
)
def test_same_operands_and_result_type_trait_for_element_type_of_shaped_types(
    operand_type: TensorType[Any],
    result_type: TensorType[Any],
    operands_num: int,
    results_num: int,
):
    op = SameOperandsAndResultTypeOp(
        operands=[
            [
                create_ssa_value(operand_type),
            ]
            * operands_num,
        ],
        result_types=[[result_type] * results_num],
    )

    with pytest.raises(
        VerifyException,
        match="requires the same type for all operands and results",
    ):
        op.verify()


@pytest.mark.parametrize(
    (
        "element_type",
        "shape",
        "operand1_and_result_encoding",
        "result_encoding2",
    ),
    [
        (
            test.TestType("foo"),
            [2, 3],
            StringAttr("bar"),
            StringAttr("baz"),
        ),
        (
            test.TestType("foo"),
            [2, 3],
            StringAttr("baz"),
            StringAttr("bar"),
        ),
        (
            test.TestType("foo"),
            [2, 3],
            StringAttr("bar"),
            NoneAttr(),
        ),
        (
            test.TestType("foo"),
            [2, 3],
            NoneAttr(),
            StringAttr("bar"),
        ),
    ],
)
def test_same_operands_and_result_type_trait_for_result_encoding_of_shaped_types(
    element_type: Attribute,
    shape: tuple[int],
    operand1_and_result_encoding: Attribute,
    result_encoding2: Attribute,
):
    op = SameOperandsAndResultTypeOp(
        operands=[
            [
                create_ssa_value(
                    TensorType(
                        element_type,
                        shape,
                        operand1_and_result_encoding,
                    )
                ),
            ],
        ],
        result_types=[
            [
                TensorType(element_type, shape, operand1_and_result_encoding),
                TensorType(element_type, shape, result_encoding2),
            ]
        ],
    )

    with pytest.raises(
        VerifyException,
        match="requires the same encoding for all operands and results",
    ):
        op.verify()


@pytest.mark.parametrize(
    "operands_num",
    [1, 2, 3],
)
@pytest.mark.parametrize(
    "results_num",
    [1, 2, 3],
)
@pytest.mark.parametrize(
    (
        "element_type",
        "shape",
        "operand_encoding",
        "result_encoding",
    ),
    [
        (
            test.TestType("foo"),
            [2, 3],
            StringAttr("bar"),
            StringAttr("baz"),
        ),
        (
            test.TestType("foo"),
            [2, 3],
            StringAttr("baz"),
            StringAttr("bar"),
        ),
        (
            test.TestType("foo"),
            [2, 3],
            StringAttr("bar"),
            NoneAttr(),
        ),
        (
            test.TestType("foo"),
            [2, 3],
            NoneAttr(),
            StringAttr("bar"),
        ),
    ],
)
def test_same_operands_and_result_type_trait_for_encoding_of_shaped_types(
    element_type: Attribute,
    shape: tuple[int],
    operand_encoding: Attribute,
    result_encoding: Attribute,
    operands_num: int,
    results_num: int,
):
    op = SameOperandsAndResultTypeOp(
        operands=[
            [
                create_ssa_value(
                    TensorType(
                        element_type,
                        shape,
                        operand_encoding,
                    )
                ),
            ]
            * operands_num,
        ],
        result_types=[[TensorType(element_type, shape, result_encoding)] * results_num],
    )

    with pytest.raises(
        VerifyException,
        match="requires the same encoding for all operands and results",
    ):
        op.verify()


@pytest.mark.parametrize(
    "operands_num",
    [1, 2, 3],
)
@pytest.mark.parametrize(
    "results_num",
    [1, 2, 3],
)
@pytest.mark.parametrize(
    (
        "operand1_and_result_shape",
        "operand2_shape",
    ),
    [
        (
            [1],
            [1],
        ),
        (
            [2, 3],
            [2, 3],
        ),
        (
            [2, 3],
            [2, DYNAMIC_INDEX],
        ),
        (
            [2, 4],
            [2, DYNAMIC_INDEX],
        ),
        (
            [2, DYNAMIC_INDEX],
            [2, DYNAMIC_INDEX],
        ),
    ],
)
def test_same_operands_and_result_type_trait_for_ranked_mixed_shapes(
    operand1_and_result_shape: tuple[int],
    operand2_shape: tuple[int],
    operands_num: int,
    results_num: int,
):
    op = SameOperandsAndResultTypeOp(
        operands=[
            [
                create_ssa_value(
                    TensorType(test.TestType("foo"), operand1_and_result_shape)
                ),
                create_ssa_value(TensorType(test.TestType("foo"), operand2_shape)),
            ]
            * operands_num,
        ],
        result_types=[
            [TensorType(test.TestType("foo"), operand1_and_result_shape)] * results_num
        ],
    )

    op.verify()


@pytest.mark.parametrize(
    "operands_num",
    [1, 2, 3],
)
@pytest.mark.parametrize(
    "results_num",
    [1, 2, 3],
)
@pytest.mark.parametrize(
    ("operand1_and_result_shape",),
    [
        ([1],),
        ([2, 3],),
        ([2, DYNAMIC_INDEX],),
        ([DYNAMIC_INDEX, DYNAMIC_INDEX],),
    ],
)
def test_same_operands_and_result_type_trait_for_mixed_rank_and_mixed_shapes(
    operand1_and_result_shape: tuple[int],
    operands_num: int,
    results_num: int,
):
    op = SameOperandsAndResultTypeOp(
        operands=[
            [
                create_ssa_value(
                    TensorType(test.TestType("foo"), operand1_and_result_shape)
                ),
                create_ssa_value(UnrankedTensorType(test.TestType("foo"))),
            ]
            * operands_num,
        ],
        result_types=[
            [TensorType(test.TestType("foo"), operand1_and_result_shape)] * results_num
        ],
    )

    op.verify()


def test_memory_effects():
    from xdsl.dialects.test import TestPureOp, TestReadOp, TestWriteOp

    assert not has_effects(TestPureOp(), MemoryEffectKind.ALLOC)
    assert not has_effects(TestReadOp(), MemoryEffectKind.ALLOC)
    assert not has_effects(TestWriteOp(), MemoryEffectKind.ALLOC)
    assert not has_effects(TestPureOp(), MemoryEffectKind.READ)
    assert has_effects(TestReadOp(), MemoryEffectKind.READ)
    assert not has_effects(TestWriteOp(), MemoryEffectKind.READ)
    assert not has_effects(TestPureOp(), MemoryEffectKind.WRITE)
    assert not has_effects(TestReadOp(), MemoryEffectKind.WRITE)
    assert has_effects(TestWriteOp(), MemoryEffectKind.WRITE)


@irdl_op_definition
class TestModifyTraitsOp(IRDLOperation):
    name = "test.test_modify_traits"


class AlwaysFailsTrait(OpTrait):
    def verify(self, op: Operation) -> None:
        raise VerifyException("Nope")


def test_modify_traits():
    op = TestModifyTraitsOp()

    op.verify()

    TestModifyTraitsOp.traits.add_trait(AlwaysFailsTrait())

    with pytest.raises(VerifyException, match="Nope"):
        op.verify()


def test_return_like():
    @irdl_op_definition
    class TestReturnLikeOp(IRDLOperation):
        name = "test.return_like"

        traits = traits_def(ReturnLike())
        my_results = var_result_def()
        my_successors = var_successor_def()

    terminator = TestReturnLikeOp(result_types=((),), successors=((),))

    with pytest.raises(VerifyException, match="test.return_like is not a terminator"):
        terminator.verify()

    TestReturnLikeOp.traits.add_trait(IsTerminator())
    terminator.verify()

    results = TestReturnLikeOp(result_types=((i32,),), successors=((),))

    with pytest.raises(
        VerifyException, match="test.return_like does not have zero results"
    ):
        results.verify()

    _ = Region((a := Block(), b := Block()))

    successors = TestReturnLikeOp(result_types=((),), successors=(b,))
    a.add_op(successors)

    with pytest.raises(
        VerifyException, match="test.return_like does not have zero successors"
    ):
        successors.verify()
