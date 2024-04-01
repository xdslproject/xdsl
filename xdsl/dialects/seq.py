"""
CIRCT’s seq dialect

[1] https://circt.llvm.org/docs/Dialects/Seq/
"""

from typing import Annotated

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IntegerAttr,
    IntegerType,
    TypeAttribute,
    i1,
)
from xdsl.dialects.hw import InnerSymAttr
from xdsl.ir import Attribute, Dialect, Operation, OpResult, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    ConstraintVar,
    IRDLOperation,
    Operand,
    ParametrizedAttribute,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
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


@irdl_op_definition
class CompRegOp(IRDLOperation):
    """
    Register a value, storing it for one cycle.
    """

    name = "seq.compreg"

    DataType = Annotated[Attribute, ConstraintVar("DataType")]

    inner_sym = opt_attr_def(InnerSymAttr)
    input = operand_def(DataType)
    clk = operand_def(clock)
    reset = opt_operand_def(i1)
    reset_value = opt_operand_def(DataType)
    data = result_def(DataType)

    irdl_options = [AttrSizedOperandSegments()]

    assembly_format = (
        "(`sym` $inner_sym^)? $input `,` $clk (`reset` $reset^ `,` $reset_value)? attr-dict "
        "`:` type($input)"
    )

    def __init__(
        self,
        input: SSAValue,
        clk: SSAValue,
        reset: tuple[SSAValue, SSAValue] | None = None,
    ):
        super().__init__(
            operands=[
                input,
                clk,
                reset[0] if reset is not None else None,
                reset[1] if reset is not None else None,
            ],
            result_types=[input.type],
        )

    def verify_(self):
        if (self.reset is not None and self.reset_value is None) or (
            self.reset_value is not None and self.reset is None
        ):
            raise VerifyException("Both reset and reset_value must be set when one is")


Seq = Dialect(
    "seq",
    [
        ClockDivider,
        CompRegOp,
    ],
    [
        ClockType,
    ],
)
