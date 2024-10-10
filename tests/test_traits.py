"""
Test the definition and usage of traits and interfaces.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from xdsl.dialects import test
from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
    i1,
    i32,
    i64,
)
from xdsl.ir import Operation, OpTrait
from xdsl.irdl import (
    Block,
    IRDLOperation,
    Region,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_region_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
)
from xdsl.traits import (
    AlwaysSpeculatable,
    ConditionallySpeculatable,
    HasAncestor,
    HasParent,
    OptionalSymbolOpInterface,
    RecursivelySpeculatable,
    SymbolOpInterface,
    SymbolTable,
    is_speculatable,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue


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
        assert isinstance(op.results[0].type, IntegerType)
        assert isinstance(op.operands[0].type, IntegerType)
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
            assert isinstance(operand.type, IntegerType)
            sum_bitwidth += operand.type.width.data
        for result in op.results:
            # This assert should be an exception in a non-testing environment.
            assert isinstance(result.type, IntegerType)
            sum_bitwidth += result.type.width.data

        if sum_bitwidth >= self.max_sum:
            raise VerifyException(
                "Operation has a bitwidth sum " f"greater or equal to {self.max_sum}."
            )


@irdl_op_definition
class TestOp(IRDLOperation):
    name = "test.test"
    traits = frozenset([LargerOperandTrait(), BitwidthSumLessThanTrait(64)])

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
    operand64 = TestSSAValue(i64)
    operand32 = TestSSAValue(i32)
    operand1 = TestSSAValue(i1)
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
    traits = frozenset([LargerOperandTrait()])


@irdl_op_definition
class TestCopyOp(LargerOperandOp):
    name = "test.test_copy"

    traits = LargerOperandOp.traits.union([BitwidthSumLessThanTrait(64)])


def test_trait_inheritance():
    """
    Check that traits are correctly inherited from parent classes.
    """
    assert TestCopyOp.traits == frozenset(
        [LargerOperandTrait(), BitwidthSumLessThanTrait(64)]
    )


@irdl_op_definition
class NoTraitsOp(IRDLOperation):
    name = "test.no_traits_op"


def test_traits_undefined():
    """Check that traits are defaulted to the empty set."""
    assert NoTraitsOp.traits == frozenset()


class WrongTraitsType(IRDLOperation):
    name = "test.no_traits"

    traits = 1  # pyright: ignore[reportAssignmentType]


def test_traits_wrong_type():
    with pytest.raises(Exception):
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


class OpWithInterface(IRDLOperation):
    name = "test.op_with_interface"
    traits = frozenset([GetNumResultsTraitForOpWithOneResult()])

    res = result_def(IntegerType)


def test_interface():
    """
    Test the features of a trait with methods (An MLIR interface).
    """
    op = OpWithInterface.create(result_types=(i32,))
    trait = OpWithInterface.get_trait(GetNumResultsTrait)
    assert trait is not None
    assert 1 == trait.get_num_results(op)


def test_get_trait_specialized():
    """
    Test get_trait and has_trait in the case where the trait is a child class of the
    trait we want.
    """
    assert OpWithInterface.has_trait(GetNumResultsTrait)
    assert OpWithInterface.has_trait(GetNumResultsTraitForOpWithOneResult)
    assert (
        OpWithInterface.get_trait(GetNumResultsTrait)
        == GetNumResultsTraitForOpWithOneResult()
    )
    assert OpWithInterface.get_traits_of_type(GetNumResultsTrait) == [
        GetNumResultsTraitForOpWithOneResult()
    ]


def test_symbol_op_interface():
    """
    Test that operations that conform to SymbolOpInterface have necessary attributes.
    """

    @irdl_op_definition
    class NoSymNameOp(IRDLOperation):
        name = "no_sym_name"
        traits = frozenset((SymbolOpInterface(),))

    op0 = NoSymNameOp()

    with pytest.raises(
        VerifyException, match='Operation no_sym_name must have a "sym_name" attribute'
    ):
        op0.verify()

    @irdl_op_definition
    class SymNameWrongTypeOp(IRDLOperation):
        name = "wrong_sym_name_type"

        sym_name = attr_def(AnyIntegerAttr)
        traits = frozenset((SymbolOpInterface(),))

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

        sym_name = attr_def(StringAttr)
        traits = frozenset((SymbolOpInterface(),))

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

        traits = frozenset((OptionalSymbolOpInterface(),))

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

    sym_name = attr_def(StringAttr)

    traits = frozenset([SymbolOpInterface()])

    def __init__(self, name: str):
        return super().__init__(attributes={"sym_name": StringAttr(name)})


@irdl_op_definition
class PropSymbolOp(IRDLOperation):
    name = "test.symbol"

    sym_name = prop_def(StringAttr)

    traits = frozenset([SymbolOpInterface()])

    def __init__(self, name: str):
        return super().__init__(properties={"sym_name": StringAttr(name)})


@pytest.mark.parametrize("SymbolOp", (SymbolOp, PropSymbolOp))
def test_symbol_table(SymbolOp: type[PropSymbolOp | SymbolOp]):
    # Some helper classes
    @irdl_op_definition
    class SymbolTableOp(IRDLOperation):
        name = "test.symbol_table"

        sym_name = opt_attr_def(StringAttr)

        one = region_def()
        two = opt_region_def()

        traits = frozenset([SymbolTable(), OptionalSymbolOpInterface()])

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

    traits = traits_def(lambda: frozenset([HasParent(TestOp)]))


def test_lazy_parent():
    """Test the trait infrastructure for an operation that defines a trait "lazily"."""
    op = HasLazyParentOp.create()
    assert len(op.get_traits_of_type(HasParent)) != 0
    assert op.get_traits_of_type(HasParent)[0].op_types == (TestOp,)
    assert op.has_trait(HasParent(TestOp))
    assert op.traits == frozenset([HasParent(TestOp)])


@irdl_op_definition
class AncestorOp(IRDLOperation):
    name = "test.ancestor"

    traits = frozenset((HasAncestor(TestOp),))


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

        traits = frozenset([SymbolTable()])

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

        traits = frozenset(trait)

    op = SupeculatabilityTestOp(regions=[Region(Block(nested_ops))])
    optrait = op.get_trait(ConditionallySpeculatable)

    if trait:
        assert optrait is not None
        assert optrait.is_speculatable(op) is speculatability
    else:
        assert optrait is None

    assert is_speculatable(op) is speculatability
