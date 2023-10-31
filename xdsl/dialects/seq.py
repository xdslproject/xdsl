"""
CIRCT’s seq dialect

[1] https://circt.llvm.org/docs/Dialects/Seq/
"""

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IntegerAttr,
    IntegerType,
    TypeAttribute,
)
from xdsl.ir import Dialect, Operation, OpResult, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    ParametrizedAttribute,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class ClockType(ParametrizedAttribute, TypeAttribute):
    """
    A type for clock-carrying wires. Signals which can be used to drive the clock input of sequential operations.
    """

    name = "seq.clock"


clock = ClockType()


@irdl_op_definition
class ClockDivider(IRDLOperation):
    """Produces a clock divided by a power of two"""

    name = "seq.clock_div"

    pow2 = attr_def(AnyIntegerAttr)
    clockIn: Operand = operand_def(ClockType)
    clockOut: OpResult = result_def(ClockType)

    def __init__(self, clockIn: SSAValue | Operation, pow2: int | AnyIntegerAttr):
        if isinstance(pow2, int):
            pow2 = IntegerAttr(pow2, IntegerType(8))
        super().__init__(
            operands=[clockIn], attributes={"pow2": pow2}, result_types=[clock]
        )

    def verify_(self) -> None:
        if self.pow2.type != IntegerType(8):
            raise VerifyException("pow2 has to be an 8-bit signless integer")
        if self.pow2.value.data.bit_count() != 1:
            raise VerifyException(
                f"divider value {self.pow2.value.data} is not a power of 2"
            )

    @classmethod
    def parse(cls, parser: Parser):
        input_ = parser.parse_operand()
        parser.parse_keyword("by")
        divider = parser.parse_integer()
        return cls(input_, divider)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_operand(self.clockIn)
        printer.print(" by ")
        printer.print(self.pow2.value.data)


Seq = Dialect(
    "seq",
    [
        ClockDivider,
    ],
    [
        ClockType,
    ],
)
