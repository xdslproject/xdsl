from __future__ import annotations
from dataclasses import dataclass

import pytest

from xdsl.ir import OpResult, OpTrait, Operation
from xdsl.irdl import VarOpResult, VarOperand, irdl_op_definition
from xdsl.utils.exceptions import VerifyException
from xdsl.dialects.builtin import i32


@dataclass(frozen=True)
class NoResultsTrait(OpTrait):

    def verify(self, op: Operation):
        if len(op.results) != 0:
            raise VerifyException(
                f"Op has {len(op.results)} results, should have none")


@dataclass(frozen=True)
class NoOperandsTrait(OpTrait):

    def verify(self, op: Operation):
        if len(op.operands) != 0:
            raise VerifyException(
                f"Op has {len(op.operands)} operands, should have none")


@dataclass(frozen=True)
class NOperandsTrait(OpTrait):
    parameter: int

    def verify(self, op: Operation):
        if len(op.operands) != self.parameter:
            raise VerifyException(
                f"Op has {len(op.operands)} operands, should have {self.parameter}"
            )


@irdl_op_definition
class MyOp(Operation):
    name = "test.my_op"
    traits = frozenset([NoResultsTrait(), NOperandsTrait(1)])

    ops: VarOperand
    res: VarOpResult


def test_has_trait_object():
    """
    Test the `has_trait` `Operation` method on a simple operation definition.
    """
    assert MyOp.has_trait(NoResultsTrait())
    assert not MyOp.has_trait(NoOperandsTrait())
    assert not MyOp.has_trait(NOperandsTrait(0))
    assert MyOp.has_trait(NOperandsTrait(1))


def test_get_traits_of_type():
    """
    Test the `get_traits_of_type` `Operation` method
    on a simple operation definition.
    """
    assert MyOp.get_traits_of_type(NoResultsTrait) == [NoResultsTrait()]
    assert MyOp.get_traits_of_type(NoOperandsTrait) == []
    assert MyOp.get_traits_of_type(NOperandsTrait) == [NOperandsTrait(1)]


def test_verifier():
    """
    Check that the traits verifier are correctly called.
    """
    operand = OpResult(i32, [], [])  # type: ignore
    op = MyOp.create(operands=[operand], result_types=[i32])
    with pytest.raises(VerifyException):
        op.verify()

    op = MyOp.create(operands=[], result_types=[])
    with pytest.raises(VerifyException):
        op.verify()

    op = MyOp.create(operands=[operand], result_types=[])
    op.verify()


class NoOperandsOp(Operation):
    traits = frozenset([NoOperandsTrait()])


@irdl_op_definition
class NoResultNorOperandOp(NoOperandsOp):
    name = "test.no_result_nor_operand_op"

    traits = frozenset([NoResultsTrait()])


def test_trait_inheritance():
    """
    Check that traits are correctly inherited from parent classes.
    """
    assert NoResultNorOperandOp.traits == frozenset(
        [NoResultsTrait(), NoOperandsTrait()])


@irdl_op_definition
class NoTraitsOp(Operation):
    name = "test.no_traits_op"


def test_traits_undefined():
    """Check that traits are defaulted to the empty set."""
    assert NoTraitsOp.traits == frozenset()


class WrongTraitsType(Operation):
    name = "test.no_traits"

    traits = 1  # type: ignore


def test_traits_wrong_type():
    with pytest.raises(Exception):
        irdl_op_definition(WrongTraitsType)
