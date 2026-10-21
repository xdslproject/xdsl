"""
Test the definition and usage of traits and interfaces.
"""

from __future__ import annotations

import re
from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from typing_extensions import TypeVar

from xdsl.context import Context
from xdsl.dialects import test
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AnyTensorTypeConstr,
    AnyUnrankedMemRefTypeConstr,
    AnyUnrankedTensorTypeConstr,
    Builtin,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ModuleOp,
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
from xdsl.dialects.test import Test
from xdsl.interfaces import (
    ConditionallySpeculatableInterface,
    RegionBranchOpInterface,
    RegionBranchTerminatorOpInterface,
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
    var_region_def,
    var_result_def,
    var_successor_def,
)
from xdsl.parser import Parser
from xdsl.traits import (
    AlwaysSpeculatable,
    ConditionallySpeculatable,
    HasAncestor,
    HasParent,
    InvocationBounds,
    IsTerminator,
    MemoryEffectKind,
    OptionalSymbolOpInterface,
    RecursivelySpeculatable,
    RegionBranchOpTrait,
    RegionBranchPoint,
    RegionBranchTerminatorOpTrait,
    RegionSuccessor,
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
        VerifyException,
        match=re.escape("Operation has a bitwidth sum greater or equal to 64."),
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
            re.escape(
                "pyrdl operation definition 'WrongTraitsType' traits field should be an "
                "instance of'OpTraits'."
            )
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

    op1 = SymNameWrongTypeOp(attributes={"sym_name": IntegerAttr(1, 32)})

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
        VerifyException,
        match=re.escape("'test.ancestor' expects ancestor op 'test.test'"),
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


@pytest.mark.parametrize("speculatability", [True, False])
def test_conditionally_speculatable_interface(speculatability: bool):
    @irdl_op_definition
    class InterfaceSpeculatabilityTestOp(
        IRDLOperation, ConditionallySpeculatableInterface
    ):
        name = "test.interface_speculatability"
        region = region_def()

        def is_speculatable(self) -> bool:
            return speculatability

    op = InterfaceSpeculatabilityTestOp(regions=[Region(Block())])
    trait = op.get_trait(ConditionallySpeculatable)

    assert trait is not None
    assert trait.is_speculatable(op) is speculatability
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

    with pytest.raises(
        VerifyException, match=re.escape("test.return_like is not a terminator")
    ):
        terminator.verify()

    TestReturnLikeOp.traits.add_trait(IsTerminator())
    terminator.verify()

    results = TestReturnLikeOp(result_types=((i32,),), successors=((),))

    with pytest.raises(
        VerifyException, match=re.escape("test.return_like does not have zero results")
    ):
        results.verify()

    _ = Region((a := Block(), b := Block()))

    successors = TestReturnLikeOp(result_types=((),), successors=(b,))
    a.add_op(successors)

    with pytest.raises(
        VerifyException,
        match=re.escape("test.return_like does not have zero successors"),
    ):
        successors.verify()


class _ForOpTrait(RegionBranchOpTrait):
    @classmethod
    def successor_regions(
        cls, op: Operation, point: RegionBranchPoint
    ) -> Sequence[RegionSuccessor]:
        if point.is_parent():
            return [RegionSuccessor(op.regions[0])]
        return [RegionSuccessor(op.regions[0]), RegionSuccessor(op)]

    @classmethod
    def entry_successor_operands(
        cls,
        op: Operation,
        successor: RegionSuccessor,
    ) -> Sequence[SSAValue]:
        return op.operands

    @classmethod
    def successor_inputs(
        cls, op: Operation, successor: RegionSuccessor
    ) -> Sequence[SSAValue]:
        if successor.is_operation():
            return op.results
        # The first block argument is an induction variable, not fed by any operand.
        return op.regions[0].block.args[1:]


@irdl_op_definition
class ForOp(IRDLOperation):
    """
    Enters its body, which branches back to itself or to the parent, like a do-while loop. Its
    first block argument is not a successor input.
    """

    name = "test.for"
    args = var_operand_def()
    res = var_result_def()
    body = region_def()
    traits = traits_def(_ForOpTrait())


class _YieldTailTrait(RegionBranchTerminatorOpTrait):
    @classmethod
    def successor_operands(
        cls,
        op: Operation,
        successor: RegionSuccessor,
    ) -> Sequence[SSAValue]:
        return op.operands[1:]


@irdl_op_definition
class YieldTailOp(IRDLOperation):
    """A terminator forwarding all of its operands except the first."""

    name = "test.yield_tail"
    args = var_operand_def()
    traits = traits_def(IsTerminator(), _YieldTailTrait())


class _ExitOnlyTrait(RegionBranchOpTrait):
    """
    A trait implemented directly, without `RegionBranchOpInterface`, to exercise the defaults of
    `RegionBranchOpTrait`."""

    @classmethod
    def successor_regions(
        cls, op: Operation, point: RegionBranchPoint
    ) -> Sequence[RegionSuccessor]:
        return [RegionSuccessor(op)]


@irdl_op_definition
class DummyOp(IRDLOperation, RegionBranchTerminatorOpInterface):
    """A dummy op that is also a terminator."""

    name = "test.dummy_op"
    traits = traits_def(IsTerminator())

    def successor_operands(self, successor: RegionSuccessor) -> Sequence[SSAValue]:
        return ()


@irdl_op_definition
class LoopRegionsOp(IRDLOperation, RegionBranchOpInterface):
    """
    The parent enters either region. Each region branches back to itself, region 1 can also
    branch to the parent. Unlike MLIR's test op, the parent enters the regions, so that they are
    reachable.
    """

    name = "test.loop_regions_op"
    res = var_result_def()
    bodies = var_region_def()

    def successor_regions(self, point: RegionBranchPoint) -> Sequence[RegionSuccessor]:
        if (terminator := point.terminator) is None:
            return [RegionSuccessor(region) for region in self.regions]
        region = terminator.parent_region()
        assert region is not None
        successors = [RegionSuccessor(region)]
        if region is self.regions[1]:
            # This region also branches back to this operation.
            successors.insert(0, RegionSuccessor(self))
        return successors

    def successor_inputs(self, successor: RegionSuccessor) -> Sequence[SSAValue]:
        return self.results if successor.is_operation() else ()


@irdl_op_definition
class SequentialRegionsOp(IRDLOperation, RegionBranchOpInterface):
    """
    Regions are executed sequentially: the parent enters region 0, which branches to region 1,
    which branches back to the parent. Unlike MLIR's test op, the parent edges are modelled, so
    that the regions are reachable.
    """

    name = "test.sequential_regions_op"
    bodies = var_region_def()

    def successor_regions(self, point: RegionBranchPoint) -> Sequence[RegionSuccessor]:
        if (terminator := point.terminator) is None:
            return [RegionSuccessor(self.regions[0])]
        if terminator.parent_region() is self.regions[0]:
            return [RegionSuccessor(self.regions[1])]
        return [RegionSuccessor(self)]


# Ops forwarding values, which MLIR's unit tests do not cover.


@irdl_op_definition
class YieldOp(IRDLOperation, RegionBranchTerminatorOpInterface):
    """A terminator forwarding all of its operands (the default)."""

    name = "test.yield"
    args = var_operand_def()
    traits = traits_def(IsTerminator())


@irdl_op_definition
class LoopOp(IRDLOperation, RegionBranchOpInterface):
    """Enters its body, which branches back to itself or to the parent, like a do-while loop."""

    name = "test.loop"
    args = var_operand_def()
    res = var_result_def()
    body = region_def()

    def successor_regions(self, point: RegionBranchPoint) -> Sequence[RegionSuccessor]:
        if point.is_parent():
            return [RegionSuccessor(self.body)]
        return [RegionSuccessor(self.body), RegionSuccessor(self)]

    def entry_successor_operands(
        self, successor: RegionSuccessor
    ) -> Sequence[SSAValue]:
        return self.operands

    def successor_inputs(self, successor: RegionSuccessor) -> Sequence[SSAValue]:
        if successor.is_operation():
            return self.results
        return self.body.block.args


class _YieldStridedTrait(RegionBranchTerminatorOpTrait):
    @classmethod
    def successor_operands(
        cls,
        op: Operation,
        successor: RegionSuccessor,
    ) -> Sequence[SSAValue]:
        return op.operands[::2]


@irdl_op_definition
class YieldStridedOp(IRDLOperation):
    """A terminator forwarding every other operand, which is not a contiguous slice."""

    name = "test.yield_strided"
    args = var_operand_def()
    traits = traits_def(IsTerminator(), _YieldStridedTrait())


@irdl_op_definition
class IntegerCastOp(IRDLOperation, RegionBranchOpInterface):
    """
    Skips its body and forwards its operands to its results, allowing any integer types
    along that edge.
    """

    name = "test.integer_cast"
    args = var_operand_def()
    res = var_result_def()
    body = region_def()

    def successor_regions(self, point: RegionBranchPoint) -> Sequence[RegionSuccessor]:
        return [RegionSuccessor(self)]

    def entry_successor_operands(
        self, successor: RegionSuccessor
    ) -> Sequence[SSAValue]:
        return self.operands

    def successor_inputs(self, successor: RegionSuccessor) -> Sequence[SSAValue]:
        return self.results

    def are_types_compatible(self, lhs: Attribute, rhs: Attribute) -> bool:
        return isinstance(lhs, IntegerType) and isinstance(rhs, IntegerType)


_TEST_OPS: tuple[type[Operation], ...] = (
    ForOp,
    YieldTailOp,
    DummyOp,
    LoopRegionsOp,
    SequentialRegionsOp,
    YieldOp,
    LoopOp,
    YieldStridedOp,
    IntegerCastOp,
)


def _parse(ir: str) -> ModuleOp:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Test)
    for op in _TEST_OPS:
        ctx.load_op(op)
    return Parser(ctx, ir).parse_module()


def _name_to_ssa(module: ModuleOp) -> dict[str, SSAValue]:
    """Map the name hint of every named result and block argument to its value."""
    results: dict[str, SSAValue] = {
        result.name_hint: result
        for op in module.walk()
        for result in op.results
        if result.name_hint is not None
    }
    block_args: dict[str, SSAValue] = {
        arg.name_hint: arg
        for block in module.walk_blocks()
        for arg in block.args
        if arg.name_hint is not None
    }
    return results | block_args


_OpT = TypeVar("_OpT", bound=Operation)


def _find(module: ModuleOp, op_type: type[_OpT]) -> _OpT:
    return next(op for op in module.walk() if isinstance(op, op_type))


_TraitT = TypeVar("_TraitT", bound=OpTrait)


def _trait(op: Operation, trait_type: type[_TraitT]) -> _TraitT:
    trait = op.get_trait(trait_type)
    assert trait is not None
    return trait


_NO_INTERFACE_IR = """
"test.op"() ({
  %value = "test.op"() : () -> i32
  "test.region_branch_terminator"(%value) : (i32) -> ()
}) : () -> ()
"""


_LOOP_IR = """
%init = "test.op"() : () -> i32
%res = "test.loop"(%init) ({
^bb0(%arg: i32):
  %next = "test.op"(%arg) : (i32) -> i32
  "test.yield"(%next) : (i32) -> ()
}) : (i32) -> i32
"""


@pytest.fixture
def loop() -> tuple[LoopOp, YieldOp, dict[str, SSAValue]]:
    module = _parse(_LOOP_IR)
    return _find(module, LoopOp), _find(module, YieldOp), _name_to_ssa(module)


_SEQUENTIAL_REGIONS_IR = (
    '"test.sequential_regions_op"() ({ "test.dummy_op"() : () -> () }, '
    '{ "test.dummy_op"() : () -> () }) : () -> ()'
)


def test_region_successor():
    op = test.TestOp()
    region = Region()
    assert RegionSuccessor(region).region is region
    assert RegionSuccessor(region).operation is None
    assert RegionSuccessor(region).is_region()
    assert RegionSuccessor(op).operation is op
    assert RegionSuccessor(op).region is None
    assert RegionSuccessor(op).is_operation()
    assert RegionSuccessor(region) == RegionSuccessor(region)
    assert (
        len({RegionSuccessor(region), RegionSuccessor(region), RegionSuccessor(op)})
        == 2
    )


def test_region_branch_point():
    terminator = test.TestRegionBranchTerminatorOp(
        operands=[[]], result_types=[[]], successors=[[]]
    )
    point = RegionBranchPoint(terminator)
    assert RegionBranchPoint.PARENT.is_parent()
    assert RegionBranchPoint.PARENT.terminator is None
    assert RegionBranchPoint.PARENT.terminator_predecessor_and_trait is None
    assert not point.is_parent()
    assert point.terminator is terminator
    assert point.terminator_predecessor_and_trait == (
        terminator,
        terminator.get_trait(RegionBranchTerminatorOpTrait),
    )
    assert point == RegionBranchPoint(terminator)
    assert point != RegionBranchPoint.PARENT
    assert len({point, RegionBranchPoint(terminator), RegionBranchPoint.PARENT}) == 2


def test_region_branch_point_without_terminator_interface():
    message = "'test.op' does not implement RegionBranchTerminatorOpInterface."
    with pytest.raises(ValueError, match=re.escape(message)):
        RegionBranchPoint(test.TestOp())


def test_invocation_bounds():
    assert InvocationBounds(0, None) == InvocationBounds.UNKNOWN
    assert InvocationBounds(2, 2).upper == 2
    with pytest.raises(ValueError, match="lower bound must be at least 0"):
        InvocationBounds(-1, None)
    with pytest.raises(ValueError, match="cannot be less than lower bound"):
        InvocationBounds(2, 1)


def test_region_branch_op_trait_without_regions():
    @irdl_op_definition
    class NoRegionsOp(IRDLOperation):
        name = "test.no_regions"

        traits = traits_def(RegionBranchOpTrait())

    with pytest.raises(
        VerifyException,
        match=re.escape(
            "'test.no_regions' implements RegionBranchOpInterface but defines no "
            "regions."
        ),
    ):
        NoRegionsOp().verify()


def test_region_branch_terminator_op_trait_without_is_terminator():
    @irdl_op_definition
    class NotTerminatorOp(IRDLOperation):
        name = "test.not_terminator"

        traits = traits_def(RegionBranchTerminatorOpTrait())

    with pytest.raises(
        VerifyException,
        match=re.escape("Expected operation 'test.not_terminator' to be a terminator."),
    ):
        NotTerminatorOp().verify()


def test_trait_defaults():
    module = _parse('"test.op"() ({ "test.termop"() : () -> () }) : () -> ()')
    op = _find(module, test.TestOp)
    successor = RegionSuccessor(op)
    assert list(_ExitOnlyTrait.entry_successor_operands(op, successor)) == []
    assert list(_ExitOnlyTrait.entry_successor_regions(op, [])) == [successor]
    assert list(_ExitOnlyTrait.successor_inputs(op, successor)) == []
    assert list(_ExitOnlyTrait.region_invocation_bounds(op, [])) == [
        InvocationBounds.UNKNOWN
    ]
    assert list(_ExitOnlyTrait.successor_regions_from_region(op, op.regions[0])) == []
    assert _ExitOnlyTrait.are_types_compatible(op, i32, i32)
    assert not _ExitOnlyTrait.are_types_compatible(op, i32, i64)
    # The parent only branches out of the op, so no region is entered.
    assert not _ExitOnlyTrait.has_loop(op)


def test_defaults():
    module = _parse(_SEQUENTIAL_REGIONS_IR)
    op = _find(module, SequentialRegionsOp)
    trait = _trait(op, RegionBranchOpTrait)
    successor = RegionSuccessor(op.regions[1])
    assert list(trait.entry_successor_operands(op, successor)) == []
    assert list(trait.successor_inputs(op, successor)) == []
    assert (
        list(trait.region_invocation_bounds(op, [])) == [InvocationBounds.UNKNOWN] * 2
    )
    assert trait.are_types_compatible(op, i32, i32)
    assert not trait.are_types_compatible(op, i32, i64)


def test_terminator_trait_defaults():
    module = _parse(_NO_INTERFACE_IR)
    op = _find(module, test.TestOp)
    terminator = _find(module, test.TestRegionBranchTerminatorOp)
    operands = RegionBranchTerminatorOpTrait.successor_operands(
        terminator, RegionSuccessor(op)
    )
    assert list(operands) == [_name_to_ssa(module)["value"]]


def test_sequential_regions():
    module = _parse(_SEQUENTIAL_REGIONS_IR)
    op = _find(module, SequentialRegionsOp)
    trait = _trait(op, RegionBranchOpTrait)
    region0, region1 = op.regions
    terminator0, terminator1 = region0.block.last_op, region1.block.last_op
    assert terminator0 is not None
    assert terminator1 is not None
    exit_ = RegionSuccessor(op)
    parent = RegionBranchPoint.PARENT
    assert list(trait.successor_regions(op, parent)) == [RegionSuccessor(region0)]
    assert list(trait.successor_regions_from_region(op, region0)) == [
        RegionSuccessor(region1)
    ]
    assert list(trait.successor_regions_from_region(op, region1)) == [exit_]
    from_terminator0 = RegionBranchPoint(terminator0)
    from_terminator1 = RegionBranchPoint(terminator1)
    assert list(trait.successor_operands(op, parent, RegionSuccessor(region0))) == []
    assert (
        list(trait.successor_operands(op, from_terminator0, RegionSuccessor(region1)))
        == []
    )
    assert list(trait.successor_operands(op, from_terminator1, exit_)) == []
    assert list(trait.successor_inputs(op, exit_)) == []


def test_loop_regions():
    module = _parse(
        '%res = "test.loop_regions_op"() ({ "test.dummy_op"() : () -> () }, '
        '{ "test.dummy_op"() : () -> () }) : () -> i32'
    )
    op = _find(module, LoopRegionsOp)
    trait = _trait(op, RegionBranchOpTrait)
    region0, region1 = op.regions
    exit_ = RegionSuccessor(op)
    assert list(trait.successor_regions(op, RegionBranchPoint.PARENT)) == [
        RegionSuccessor(region0),
        RegionSuccessor(region1),
    ]
    assert list(trait.successor_regions_from_region(op, region0)) == [
        RegionSuccessor(region0)
    ]
    assert list(trait.successor_regions_from_region(op, region1)) == [
        exit_,
        RegionSuccessor(region1),
    ]
    assert list(trait.successor_inputs(op, exit_)) == list(op.results)
    assert list(trait.successor_inputs(op, RegionSuccessor(region0))) == []


def test_successor_regions(loop: tuple[LoopOp, YieldOp, dict[str, SSAValue]]):
    op, yield_op, _ = loop
    body, exit_ = RegionSuccessor(op.body), RegionSuccessor(op)
    trait = _trait(op, RegionBranchOpTrait)
    assert list(trait.entry_successor_regions(op, [None])) == [body]
    assert list(trait.successor_regions_from_region(op, op.body)) == [body, exit_]
    assert list(trait.all_region_branch_points(op)) == [
        RegionBranchPoint.PARENT,
        RegionBranchPoint(yield_op),
    ]
    terminator_trait = _trait(yield_op, RegionBranchTerminatorOpTrait)
    assert list(terminator_trait.successor_regions(yield_op, [None])) == [
        body,
        exit_,
    ]


def test_successor_operands(loop: tuple[LoopOp, YieldOp, dict[str, SSAValue]]):
    op, yield_op, ssa = loop
    body, exit_ = RegionSuccessor(op.body), RegionSuccessor(op)
    trait = _trait(op, RegionBranchOpTrait)
    parent, from_yield = RegionBranchPoint.PARENT, RegionBranchPoint(yield_op)
    assert list(trait.successor_operands(op, parent, body)) == [ssa["init"]]
    assert list(trait.successor_operands(op, from_yield, exit_)) == [ssa["next"]]
    assert list(trait.successor_inputs(op, body)) == [ssa["arg"]]
    assert list(trait.successor_inputs(op, exit_)) == [ssa["res"]]


def test_branch_points_skip_terminators_without_interface():
    # Region 0 ends in a terminator without the interface, region 1 is an empty block.
    module = _parse("""
    "test.loop_regions_op"() ({ "test.termop"() : () -> () }, {
    ^bb0:
    }) : () -> ()
    """)
    op = _find(module, LoopRegionsOp)
    trait = _trait(op, RegionBranchOpTrait)
    assert list(trait.all_region_branch_points(op)) == [RegionBranchPoint.PARENT]
    assert list(trait.successor_regions_from_region(op, op.regions[0])) == []
    assert not trait.is_repetitive_region(op, 0)
    assert not trait.is_repetitive_region(op, 1)


def test_terminator_successor_regions_without_parent_interface():
    terminator = test.TestRegionBranchTerminatorOp(
        operands=[[]], result_types=[[]], successors=[[]]
    )
    with pytest.raises(
        ValueError,
        match=re.escape("'test.region_branch_terminator' has no parent op."),
    ):
        _trait(terminator, RegionBranchTerminatorOpTrait).successor_regions(
            terminator, []
        )
    terminator = _find(_parse(_NO_INTERFACE_IR), test.TestRegionBranchTerminatorOp)
    message = "'test.op', parent of 'test.region_branch_terminator', does not implement"
    with pytest.raises(ValueError, match=re.escape(message)):
        _trait(terminator, RegionBranchTerminatorOpTrait).successor_regions(
            terminator, []
        )


def test_value_flow(loop: tuple[LoopOp, YieldOp, dict[str, SSAValue]]):
    op, yield_op, ssa = loop
    body, exit_ = RegionSuccessor(op.body), RegionSuccessor(op)
    parent, from_yield = RegionBranchPoint.PARENT, RegionBranchPoint(yield_op)
    trait = _trait(op, RegionBranchOpTrait)
    assert list(trait.predecessors(op, body)) == [parent, from_yield]
    assert list(trait.predecessors(op, exit_)) == [from_yield]
    # A loop-carried value has two predecessors: the entry operand and the yielded value.
    assert list(trait.predecessor_values(op, body, 0)) == [ssa["init"], ssa["next"]]
    assert list(trait.predecessor_values(op, exit_, 0)) == [ssa["next"]]


def test_value_flow_from_trait():
    # Two terminators branch to each successor. `yield_tail` forwards only its second operand, so
    # `index` refers to the forwarded operands, not to the terminator's own operands.
    module = _parse("""
    %init = "test.op"() : () -> i32
    %res = "test.for"(%init) ({
    ^bb0(%iv: index, %arg: i32):
      "test.region_branch_terminator"(%arg) : (i32) -> ()
    ^bb1(%x: index, %y: i32):
      "test.yield_tail"(%x, %y) : (index, i32) -> ()
    }) : (i32) -> i32
    """)
    op, ssa = _find(module, ForOp), _name_to_ssa(module)
    terminator = _find(module, test.TestRegionBranchTerminatorOp)
    yield_tail = _find(module, YieldTailOp)
    body, exit_ = RegionSuccessor(op.body), RegionSuccessor(op)
    points = [
        RegionBranchPoint.PARENT,
        RegionBranchPoint(terminator),
        RegionBranchPoint(yield_tail),
    ]
    trait = _trait(op, RegionBranchOpTrait)
    assert list(trait.predecessors(op, body)) == points
    assert list(trait.predecessors(op, exit_)) == points[1:]
    assert list(trait.predecessor_values(op, body, 0)) == [
        ssa["init"],
        ssa["arg"],
        ssa["y"],
    ]
    assert list(trait.predecessor_values(op, exit_, 0)) == [ssa["arg"], ssa["y"]]
    # A successor that no branch point reaches has no predecessors.
    unreached = RegionSuccessor(Region())
    assert list(trait.predecessors(op, unreached)) == []
    assert list(trait.predecessor_values(op, unreached, 0)) == []
    with pytest.raises(IndexError):
        trait.predecessor_values(op, exit_, 1)


def test_non_successor_inputs():
    module = _parse("""
    %init = "test.op"() : () -> i32
    %res = "test.for"(%init) ({
    ^bb0(%iv: index, %arg: i32):
      "test.region_branch_terminator"(%arg) : (i32) -> ()
    }) : (i32) -> i32
    """)
    op, ssa = _find(module, ForOp), _name_to_ssa(module)
    trait = _trait(op, RegionBranchOpTrait)
    assert list(trait.non_successor_inputs(op, RegionSuccessor(op.body))) == [ssa["iv"]]
    assert list(trait.non_successor_inputs(op, RegionSuccessor(op))) == []


def test_successor_mappings(loop: tuple[LoopOp, YieldOp, dict[str, SSAValue]]):
    op, yield_op, ssa = loop
    trait = _trait(op, RegionBranchOpTrait)
    assert trait.successor_operand_input_mapping(op) == {
        (op, 0): [ssa["arg"]],
        (yield_op, 0): [ssa["arg"], ssa["res"]],
    }
    assert trait.successor_operand_input_mapping(op, RegionBranchPoint(yield_op)) == {
        (yield_op, 0): [ssa["arg"], ssa["res"]],
    }
    assert trait.successor_input_operand_mapping(op) == {
        ssa["arg"]: [(op, 0), (yield_op, 0)],
        ssa["res"]: [(yield_op, 0)],
    }


def test_successor_mapping_with_repeated_operand():
    module = _parse("""
    %init = "test.op"() : () -> i32
    %res:2 = "test.loop"(%init, %init) ({
    ^bb0(%a: i32, %b: i32):
      %next = "test.op"() : () -> i32
      "test.yield"(%next, %next) : (i32, i32) -> ()
    }) : (i32, i32) -> (i32, i32)
    """)
    op, yield_op = _find(module, LoopOp), _find(module, YieldOp)
    block_args, results = op.body.block.args, op.res
    trait = _trait(op, RegionBranchOpTrait)
    mapping = trait.successor_operand_input_mapping(op)
    assert mapping[(op, 0)] == [block_args[0]]
    assert mapping[(op, 1)] == [block_args[1]]
    assert mapping[(yield_op, 0)] == [block_args[0], results[0]]
    assert mapping[(yield_op, 1)] == [block_args[1], results[1]]


@pytest.mark.parametrize(
    ("ir", "message"),
    [
        (
            """
            %a, %b, %c = "test.op"() : () -> (i32, i32, i32)
            "test.loop"() ({
            ^bb0(%x: i32, %y: i32):
              "test.yield_strided"(%a, %b, %c) : (i32, i32, i32) -> ()
            }) : () -> (i32, i32)
            """,
            "are not a contiguous slice",
        ),
        (
            """
            %a = "test.op"() : () -> i32
            "test.loop"() ({
            ^bb0(%x: i32):
              "test.yield_tail"(%a, %a) : (i32, i32) -> ()
            }) : () -> i32
            """,
            "operand slots are ambiguous",
        ),
    ],
)
def test_successor_mapping_unrecoverable_slots(ir: str, message: str):
    op = _find(_parse(ir), LoopOp)
    terminator = op.body.block.last_op
    assert terminator is not None
    trait = _trait(op, RegionBranchOpTrait)
    with pytest.raises(ValueError, match=message):
        trait.successor_operand_input_mapping(op, RegionBranchPoint(terminator))


# Ports of the `RecursiveRegions` and `NotRecursiveRegions` tests.


def test_recursive_regions():
    module = _parse("""
    "test.loop_regions_op"() (
      { "test.dummy_op"() : () -> () },
      { "test.dummy_op"() : () -> () },
      { "test.dummy_op"() : () -> () }
    ) : () -> ()
    """)
    op = _find(module, LoopRegionsOp)
    trait = _trait(op, RegionBranchOpTrait)
    assert all(trait.is_repetitive_region(op, index) for index in range(3))
    # The parent enters every region, and each region branches back to itself.
    assert trait.has_loop(op)


def test_not_recursive_regions():
    module = _parse("""
    "test.sequential_regions_op"() (
      { "test.dummy_op"() : () -> () },
      { "test.dummy_op"() : () -> () }
    ) : () -> ()
    """)
    op = _find(module, SequentialRegionsOp)
    trait = _trait(op, RegionBranchOpTrait)
    assert not any(trait.is_repetitive_region(op, index) for index in range(2))
    assert not trait.has_loop(op)


def test_converging_successors():
    # Both blocks of region 0 branch to region 1, so the traversal reaches region 1 twice.
    module = _parse("""
    "test.sequential_regions_op"() ({
      "test.dummy_op"() : () -> ()
    ^bb1:
      "test.dummy_op"() : () -> ()
    }, {
      "test.dummy_op"() : () -> ()
    }) : () -> ()
    """)
    op = _find(module, SequentialRegionsOp)
    trait = _trait(op, RegionBranchOpTrait)
    assert not trait.is_repetitive_region(op, 0)
    # The terminators forward no values, so no operand slot is mapped.
    assert trait.successor_operand_input_mapping(op) == {}


def test_has_loop(loop: tuple[LoopOp, YieldOp, dict[str, SSAValue]]):
    op, _, _ = loop
    trait = _trait(op, RegionBranchOpTrait)
    assert trait.has_loop(op)
    assert trait.is_repetitive_region(op, 0)


def test_verify_valid(loop: tuple[LoopOp, YieldOp, dict[str, SSAValue]]):
    op, _, _ = loop
    op.verify()


@pytest.mark.parametrize(
    ("ir", "message"),
    [
        (
            """
            %init = "test.op"() : () -> i32
            "test.loop"(%init) ({
            ^bb0(%a: i32, %b: i32):
              "test.yield"(%a, %b) : (i32, i32) -> ()
            }) : (i32) -> (i32, i32)
            """,
            "along control flow edge from parent to Region #0: region branch point has 1 operands, "
            "but region successor needs 2 inputs",
        ),
        (
            """
            %init = "test.op"() : () -> i32
            "test.loop"(%init) ({
            ^bb0(%a: i64):
              "test.yield"(%a) : (i64) -> ()
            }) : (i32) -> i64
            """,
            "along control flow edge from parent to Region #0: successor operand type #0 i32 "
            "should match successor input type #0 i64",
        ),
        (
            """
            %init = "test.op"() : () -> i32
            "test.loop"(%init) ({
            ^bb0(%a: i32):
              %next = "test.op"() : () -> i64
              "test.yield"(%next) : (i64) -> ()
            }) : (i32) -> i64
            """,
            "along control flow edge from Operation 'test.yield' to Region #0: successor "
            "operand type #0 i64 should match successor input type #0 i32",
        ),
        (
            """
            %init = "test.op"() : () -> i32
            "test.loop"(%init) ({
            ^bb0(%a: i32):
              "test.yield"(%a) : (i32) -> ()
            }) : (i32) -> i64
            """,
            "along control flow edge from Operation 'test.yield' to Operation 'test.loop': "
            "successor operand type #0 i32 should match successor input type #0 i64",
        ),
    ],
)
def test_verify_invalid_edges(ir: str, message: str):
    op = _find(_parse(ir), LoopOp)
    with pytest.raises(VerifyException, match=re.escape(message)):
        op.verify()


def test_verify_custom_type_compatibility():
    module = _parse("""
    %x = "test.op"() : () -> i32
    %y = "test.integer_cast"(%x) ({ "test.termop"() : () -> () }) : (i32) -> i64
    %z = "test.integer_cast"(%x) ({ "test.termop"() : () -> () }) : (i32) -> index
    """)
    valid, invalid = (op for op in module.walk() if isinstance(op, IntegerCastOp))
    trait = _trait(valid, RegionBranchOpTrait)
    assert trait.are_types_compatible(valid, i32, i64)
    valid.verify()
    message = (
        "along control flow edge from parent to Operation 'test.integer_cast': "
        "successor operand type #0 i32 should match successor input type #0 index"
    )
    with pytest.raises(VerifyException, match=re.escape(message)):
        invalid.verify()


def test_verify_no_regions():
    # Variadic regions may be empty at runtime.
    _find(_parse('"test.loop_regions_op"() : () -> ()'), LoopRegionsOp).verify()
