from typing import TYPE_CHECKING, reveal_type

from xdsl.dialects.builtin import AnyTensorType, IntegerType, TensorType
from xdsl.dialects.stablehlo import (
    AddOp,
    _abstract_operation_class_factory,  # pyright: ignore[reportPrivateUsage]
)
from xdsl.irdl import VarConstraint, base
from xdsl.utils.test_value import TestSSAValue


def test_abstract_factory():
    unop, binop = _abstract_operation_class_factory(
        "", VarConstraint("T", base(AnyTensorType))
    )
    unop.operand
    if TYPE_CHECKING:
        reveal_type(unop.T, expected_text="VarConstraint[Attribute]")
        reveal_type(unop.operand, expected_text="SSAValue[Attribute]")
        reveal_type(binop.T, expected_text="VarConstraint[Attribute]")
        reveal_type(binop.lhs, expected_text="SSAValue[Attribute]")
        reveal_type(binop.rhs, expected_text="SSAValue[Attribute]")


def test_impact_of_abstract_factory_in_ops_that_uses_base_classes():
    a = TestSSAValue(TensorType(IntegerType(32), []))
    addop = AddOp(a, a)
    if TYPE_CHECKING:
        reveal_type(addop.T, expected_text="VarConstraint[Attribute]")
        reveal_type(addop.lhs, expected_text="SSAValue[Attribute]")
        reveal_type(addop.rhs, expected_text="SSAValue[Attribute]")
