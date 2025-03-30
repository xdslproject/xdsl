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
    binop.lhs
    binop.rhs


def test_impact_of_abstract_factory_in_ops_that_uses_base_classes():
    a = TestSSAValue(TensorType(IntegerType(32), []))
    addop = AddOp(a, a)
    addop.lhs
    addop.rhs
