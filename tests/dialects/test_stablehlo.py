from xdsl.dialects.builtin import AnyTensorType
from xdsl.dialects.stablehlo import (
    _abstract_operation_class_factory,  # pyright: ignore[reportPrivateUsage]
)
from xdsl.irdl import VarConstraint, base


def test_abstract_factory():
    unop, binop = _abstract_operation_class_factory(
        "", VarConstraint("T", base(AnyTensorType))
    )
    unop.operand
    binop.lhs
    binop.rhs
