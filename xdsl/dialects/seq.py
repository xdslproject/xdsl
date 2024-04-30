"""
CIRCT’s seq dialect

[1] https://circt.llvm.org/docs/Dialects/Seq/
"""

from enum import Enum
from typing import Annotated

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IntegerAttr,
    IntegerType,
    TypeAttribute,
    i1,
)
from xdsl.dialects.hw import InnerSymAttr
from xdsl.ir import Attribute, Data, Dialect, Operation, OpResult, SSAValue
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
from xdsl.parser import AttrParser, Parser
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
    power_on_value = opt_operand_def(DataType)
    data = result_def(DataType)

    irdl_options = [AttrSizedOperandSegments()]

    assembly_format = (
        "(`sym` $inner_sym^)? $input `,` $clk "
        "(`reset` $reset^ `,` $reset_value)? "
        "(`powerOn` $power_on_value^)? "
        "attr-dict `:` type($input)"
    )

    def __init__(
        self,
        input: SSAValue,
        clk: SSAValue,
        reset: tuple[SSAValue, SSAValue] | None = None,
        power_on_value: SSAValue | None = None,
    ):
        super().__init__(
            operands=[
                input,
                clk,
                reset[0] if reset is not None else None,
                reset[1] if reset is not None else None,
                power_on_value,
            ],
            result_types=[input.type],
        )

    def verify_(self):
        if (self.reset is None) != (self.reset_value is None):
            raise VerifyException("Both reset and reset_value must be set when one is")


class ClockConstAttrData(Enum):
    LOW = 0
    HIGH = 1


@irdl_attr_definition
class ClockConstAttr(Data[ClockConstAttrData]):
    """
    Clock constant.

    This attribute diverges slightly from the upstream implementation
    as xDSL does not allow completely unstructured parsing and printing
    of attributes (for good reasons).
    """

    name = "seq.clock_constant"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> ClockConstAttrData:
        with parser.in_angle_brackets():
            return ClockConstAttr.parse_parameter_free_standing(parser)

    @classmethod
    def parse_parameter_free_standing(cls, parser: AttrParser) -> ClockConstAttrData:
        if parser.parse_optional_keyword("low") is not None:
            return ClockConstAttrData.LOW
        if parser.parse_optional_keyword("high") is not None:
            return ClockConstAttrData.HIGH
        parser.raise_error("Expected either low or high clock value")

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.print_parameter_free_standing(printer)

    def print_parameter_free_standing(self, printer: Printer) -> None:
        match self.data:
            case ClockConstAttrData.LOW:
                printer.print("low")
            case ClockConstAttrData.HIGH:
                printer.print("high")


@irdl_op_definition
class ConstClockOp(IRDLOperation):
    """
    The constant operation produces a constant clock value.
    """

    name = "seq.const_clock"

    value: ClockConstAttr = attr_def(ClockConstAttr)
    result: OpResult = result_def(clock)

    @classmethod
    def parse(cls, parser: Parser) -> "ConstClockOp":
        value = ClockConstAttr(ClockConstAttr.parse_parameter_free_standing(parser))
        attrs = parser.parse_optional_attr_dict_with_reserved_attr_names(("value",))
        attrs_data = attrs.data if attrs is not None else {}
        attrs_data["value"] = value
        return ConstClockOp.create(attributes=attrs_data, result_types=[clock])

    def print(self, printer: Printer):
        printer.print(" ")
        self.value.print_parameter_free_standing(printer)
        printer.print_op_attributes(self.attributes, reserved_attr_names=("value",))


Seq = Dialect(
    "seq",
    [
        ClockDivider,
        CompRegOp,
        ConstClockOp,
    ],
    [
        ClockType,
        ClockConstAttr,
    ],
)
