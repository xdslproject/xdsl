from typing import Any, TypeGuard

from xdsl.ir import AttributeInvT
from xdsl.irdl import GenericAttrConstraint
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


def isattr(
    arg: Any, hint: type[AttributeInvT] | GenericAttrConstraint[AttributeInvT]
) -> TypeGuard[AttributeInvT]:
    from xdsl.irdl import ConstraintContext

    if isinstance(hint, GenericAttrConstraint):
        try:
            hint.verify(arg, ConstraintContext())
            return True
        except VerifyException:
            return False

    return isa(arg, hint)
