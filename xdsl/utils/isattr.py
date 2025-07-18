from typing import Any, TypeGuard

from typing_extensions import deprecated

from xdsl.ir import AttributeInvT
from xdsl.irdl import AttrConstraint


@deprecated("Please use hint.verifies(arg) or isa(arg, hint) instead")
def isattr(arg: Any, hint: AttrConstraint[AttributeInvT]) -> TypeGuard[AttributeInvT]:
    return hint.verifies(arg)
