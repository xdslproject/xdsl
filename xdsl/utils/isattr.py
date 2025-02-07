from typing import TYPE_CHECKING, Any, TypeGuard

from xdsl.ir import AttributeInvT
from xdsl.irdl import GenericAttrConstraint
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

if TYPE_CHECKING:
    from typing_extensions import TypeForm


def isattr(
    arg: Any, hint: "TypeForm[AttributeInvT]" | GenericAttrConstraint[AttributeInvT]
) -> TypeGuard[AttributeInvT]:
    """
    A helper method to check whether a given attribute has a given type or conforms to a
    given constraint.
    """
    from xdsl.irdl import ConstraintContext

    if isinstance(hint, GenericAttrConstraint):
        try:
            hint.verify(arg, ConstraintContext())
            return True
        except VerifyException:
            return False

    return isa(arg, hint)
