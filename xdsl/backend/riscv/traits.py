from xdsl.ir import Operation
from xdsl.traits import HasInsnsRepresentation


class StaticInsnsRepresentation(HasInsnsRepresentation):
    """
    Returns the first parameter as an insn template string.

    See https://sourceware.org/binutils/docs/as/RISC_002dV_002dDirectives.html for more information
    """

    def get_insn(self, op: Operation) -> str:
        """
        Return the insns representation of the operation for printing.
        """
        insn_str = self.parameters
        if not isinstance(insn_str, str):
            raise ValueError(
                "Parameter of StaticInsnsRepresentation must be the insn string"
            )
        return insn_str
