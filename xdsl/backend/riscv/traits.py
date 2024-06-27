from dataclasses import dataclass, field

from xdsl.ir import Operation
from xdsl.traits import HasInsnRepresentation


@dataclass(frozen=True)
class StaticInsnRepresentation(HasInsnRepresentation):
    """
    Returns the first parameter as an insn template string.

    See https://sourceware.org/binutils/docs/as/RISC_002dV_002dDirectives.html for more information
    """

    insn: str = field(kw_only=True)

    def get_insn(self, op: Operation) -> str:
        """
        Return the insn representation of the operation for printing.
        """
        return self.insn
