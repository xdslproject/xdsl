from dataclasses import dataclass
from xdsl.ir import OpTrait, Operation
from xdsl.utils.exceptions import VerifyException


class Pure(OpTrait):
    """A trait that signals that an operation has no side effects."""


@dataclass(frozen=True)
class HasParent(OpTrait):
    """Constraint the operation to have a specific parent operation."""

    parameters: tuple[type[Operation], ...]

    def __init__(self, parameters: type[Operation] | tuple[type[Operation], ...]):
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        if len(parameters) == 0:
            raise ValueError("parameters must not be empty")
        super().__init__(parameters)

    def verify(self, op: Operation) -> None:
        parent = op.parent_op()
        if isinstance(parent, tuple(self.parameters)):
            return
        if len(self.parameters) == 1:
            raise VerifyException(
                f"'{op.name}' expects parent op '{self.parameters[0].name}'"
            )
        names = ", ".join([f"'{p.name}'" for p in self.parameters])
        raise VerifyException(f"'{op.name}' expects parent op to be one of {names}")
