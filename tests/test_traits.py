"""
Test the definition and usage of traits and interfaces.
"""

from __future__ import annotations

import pytest

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from xdsl.ir import OpResult, OpTrait, Operation
from xdsl.irdl import Operand, irdl_op_definition, IRDLOperation
from xdsl.utils.exceptions import VerifyException
from xdsl.dialects.builtin import IntegerType, i1, i32, i64
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
        assert isinstance(op.results[0].typ, IntegerType)
        assert isinstance(op.operands[0].typ, IntegerType)
        if op.results[0].typ.width.data >= op.operands[0].typ.width.data:
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
            assert isinstance(operand.typ, IntegerType)
            sum_bitwidth += operand.typ.width.data
        for result in op.results:
            # This assert should be an exception in a non-testing environment.
            assert isinstance(result.typ, IntegerType)
            sum_bitwidth += result.typ.width.data

        if sum_bitwidth >= self.max_sum:
            raise VerifyException(
                "Operation has a bitwidth sum " f"greater or equal to {self.max_sum}."
            )


@irdl_op_definition
class TestOp(IRDLOperation):
    name = "test.test"
    traits = frozenset([LargerOperandTrait(), BitwidthSumLessThanTrait(64)])

    ops: Annotated[Operand, IntegerType]
    res: Annotated[OpResult, IntegerType]


def test_has_trait_object():
    """
    Test the `has_trait` `Operation` method on a simple operation definition.
    """
    assert TestOp.has_trait(LargerOperandTrait)
    assert not TestOp.has_trait(LargerResultTrait)
    assert not TestOp.has_trait(BitwidthSumLessThanTrait, 0)
    assert TestOp.has_trait(BitwidthSumLessThanTrait, 64)


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

    traits = 1  # type: ignore


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

    res: Annotated[OpResult, IntegerType]


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
