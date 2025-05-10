from typing import Any, TypeGuard

from typing_extensions import deprecated

from xdsl.ir import AttributeInvT
from xdsl.irdl import GenericAttrConstraint


@deprecated("Please use hint.matches(arg) or isa(arg, hint) instead")
def isattr(
    arg: Any, hint: GenericAttrConstraint[AttributeInvT]
) -> TypeGuard[AttributeInvT]:
    return hint.matches(arg)
