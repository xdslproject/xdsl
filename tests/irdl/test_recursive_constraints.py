from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import ClassVar

import pytest
from typing_extensions import TypeVar

from xdsl.dialects.builtin import IntAttr
from xdsl.irdl import (
    AnyInt,
    AttrConstraint,
    ConstraintContext,
    IntConstraint,
    IntVarConstraint,
    IRDLOperation,
    irdl_op_definition,
    result_def,
)
from xdsl.utils.exceptions import VerifyException

# Test a constraint that relies on inferring the value of other constaints


@dataclass(frozen=True)
class AddConstraint(IntConstraint):
    lhs: IntConstraint
    rhs: IntConstraint

    def verify(self, i: int, constraint_context: ConstraintContext) -> None:
        if i != self.infer(constraint_context):
            raise VerifyException()

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        return self.lhs.can_infer(var_constraint_names) and self.rhs.can_infer(
            var_constraint_names
        )

    def infer(self, context: ConstraintContext) -> int:
        return self.lhs.infer(context) + self.rhs.infer(context)

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> IntConstraint:
        return AddConstraint(
            self.lhs.mapping_type_vars(type_var_mapping),
            self.rhs.mapping_type_vars(type_var_mapping),
        )


# Should function no matter which order the constraints appear in
@irdl_op_definition
class RecursiveOp(IRDLOperation):
    name = "test.recursive"

    A: ClassVar = IntVarConstraint("A", AnyInt())
    B: ClassVar = IntVarConstraint("B", AnyInt())

    in1 = result_def(IntAttr.constr(AddConstraint(A, B)))
    in2 = result_def(IntAttr.constr(A))
    in3 = result_def(IntAttr.constr(B))
    in4 = result_def(IntAttr.constr(AddConstraint(A, B)))


def test_recursive_constraint():
    op = RecursiveOp(result_types=(IntAttr(3), IntAttr(1), IntAttr(2), IntAttr(3)))

    op.verify()

    op2 = RecursiveOp(result_types=(IntAttr(2), IntAttr(1), IntAttr(2), IntAttr(3)))

    with pytest.raises(
        VerifyException, match="result 'in1' at position 0 does not verify"
    ):
        op2.verify()
