from __future__ import annotations
from typing import Type
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.utils import *


@dataclass
class InterpResult:
    success: bool
    error_msg: str = ""


class InterpreterException(Exception):
    ...


# TODO: make this an enum + find out how to extend enums
# Looks like extending Enums is not possible. Hmm
@dataclass(frozen=True)
class InterpModifier(ABC):
    """
    An InterpModifier enables registering more than one interpretation function for an operation.
    This is useful when the operation is to be interpreted differently depending on the context it is in.
    An example for this is the interpretation for pdl.OperationOp, wich has to be different depending
    on whether it is used during matching of IR or during generation of new IR.
    """
    pass


@dataclass(frozen=True)
class DefaultInterpModifier(InterpModifier):
    pass


@dataclass
class Interpreter():

    _registered_ops: dict[tuple[Type[Operation], InterpModifier],
                          Callable[[Operation, Interpreter, *Any],
                                   InterpResult]] = field(default_factory=dict)
    _visited_ops: list[Operation] = field(default_factory=list)

    def register_interpretable_op(
        self,
        op_type: Type[Operation],
        interp_fun: Callable[[Operation, Interpreter, *Any], InterpResult],
        modifier: InterpModifier = DefaultInterpModifier()):
        self._registered_ops[(op_type, modifier)] = interp_fun

    def interpret_op(
        self,
        op: Operation,
        *args: Any,
        modifier: InterpModifier = DefaultInterpModifier()
    ) -> InterpResult:
        if ((op_type := type(op)), modifier) in self._registered_ops:
            return self._registered_ops[(op_type, modifier)](op, self, *args)
        else:
            raise Exception(
                f"Operation with type: {op_type} not registered for interpretation!"
            )
