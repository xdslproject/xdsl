from abc import ABC, abstractmethod
from xdsl.dialects import builtin
from xdsl.ir import MLContext
from typing import ClassVar


class ModulePass(ABC):
    """
    A Pass is a named rewrite pass over an IR module.

    All passes are expected to leave the IR in a valid state after application.
    That is, the IR verifies. In turn, all passes can expect the IR they are
    applied to to be in a valid state.
    """

    name: ClassVar[str]

    @abstractmethod
    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        ...
