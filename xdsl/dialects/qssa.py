from collections.abc import Sequence

from xdsl.dialects.builtin import IntAttr
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer


@irdl_attr_definition
class QubitsAttr(ParametrizedAttribute, TypeAttribute):
    """
    Type for a collection of `n` qubits
    """

    name = "qssa.qubits"

    # number of qubits
    n: ParameterDef[IntAttr]

    def __init__(self, n: int | IntAttr):
        if isinstance(n, int):
            n = IntAttr(n)
        super().__init__((n,))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            n = parser.parse_integer(allow_boolean=False, allow_negative=False)
            return (IntAttr(n),)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(f"{self.n.data}")


qubits1 = QubitsAttr(1)
qubits2 = QubitsAttr(2)


@irdl_op_definition
class HGateOp(IRDLOperation):
    name = "qssa.h"

    input = operand_def(qubits1)

    output = result_def(qubits1)

    assembly_format = "$input attr-dict"

    def __init__(self, input: SSAValue):
        super().__init__(operands=(input,), result_types=(qubits1,))


QSSA = Dialect(
    "qssa",
    [
        HGateOp,
    ],
    [
        QubitsAttr,
    ],
)
