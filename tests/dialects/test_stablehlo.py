import platform
from typing import TYPE_CHECKING

from xdsl.dialects.builtin import AnyTensorType, IntegerType, TensorType
from xdsl.dialects.stablehlo import (
    AddOp,
    _abstract_operation_class_factory,  # pyright: ignore[reportPrivateUsage]
)
from xdsl.irdl import VarConstraint, base
from xdsl.utils.test_value import TestSSAValue

has_reveal_type = TYPE_CHECKING and int(platform.python_version_tuple()[1]) >= 11
if has_reveal_type:
    # reveal_type is only supported on python 3.11 and above
    # https://docs.python.org/3/library/typing.html#typing.reveal_type
    from typing import reveal_type


def test_abstract_factory():
    unop, binop = _abstract_operation_class_factory(
        "", VarConstraint("T", base(AnyTensorType))
    )
    unop.operand
    if has_reveal_type:
        reveal_type(unop.T, expected_text="VarConstraint[Attribute]")
        reveal_type(unop.operand, expected_text="SSAValue[Attribute]")
        reveal_type(binop.T, expected_text="VarConstraint[Attribute]")
        reveal_type(binop.lhs, expected_text="SSAValue[Attribute]")
        reveal_type(binop.rhs, expected_text="SSAValue[Attribute]")


def test_impact_of_abstract_factory_in_ops_that_uses_base_classes():
    a = TestSSAValue(TensorType(IntegerType(32), []))
    addop = AddOp(a, a)
    if has_reveal_type:
        reveal_type(addop.T, expected_text="VarConstraint[Attribute]")
        reveal_type(addop.lhs, expected_text="SSAValue[Attribute]")
        reveal_type(addop.rhs, expected_text="SSAValue[Attribute]")
