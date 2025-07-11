from dataclasses import dataclass
from typing import TypeAlias

from typing_extensions import TypeVar

from xdsl.dialects.builtin import IntAttr
from xdsl.ir import Attribute
from xdsl.irdl import AttrConstraint, BaseAttr, ConstraintContext, GenericAttrConstraint
from xdsl.utils.exceptions import VerifyException

IntConstraint: TypeAlias = GenericAttrConstraint[IntAttr]
AnyIntConstr: IntConstraint = BaseAttr(IntAttr)


@dataclass(frozen=True)
class AtLeast(IntConstraint):
    """Constrain an integer to be at least a given value."""

    bound: int
    """The minimum value the integer can take."""

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if not isinstance(attr, IntAttr):
            raise VerifyException(f"{attr} should be an IntAttr")
        if attr.data < self.bound:
            raise VerifyException(f"expected integer >= {self.bound}, got {attr.data}")

    def mapping_type_vars(
        self, type_var_mapping: dict[TypeVar, AttrConstraint]
    ) -> IntConstraint:
        return self
