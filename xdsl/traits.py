from dataclasses import dataclass
from xdsl.ir import OpTrait, Operation
from xdsl.utils.exceptions import VerifyException


class Pure(OpTrait):
    """A trait that signals that an operation has no side effects."""


@dataclass(frozen=True)
class HasParent(OpTrait):
    """Constraint the operation to have a specific parent operation."""

    parameters: type[Operation]

    def verify(self, op: Operation) -> None:
        parent = op.parent_op()
        if parent is None:
            raise VerifyException(
                f"Operation expects a parent of type {self.parameters.name}, "
                "but has no parent."
            )
        if not isinstance(parent, self.parameters):
            raise VerifyException(
                f"Operation expects a parent of type '{self.parameters.name}', "
                f"but has a parent of type '{parent.name}'."
            )
        return super().verify(op)
