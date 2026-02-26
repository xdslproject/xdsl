"""
CIRCT’s seq dialect.

See external [documentation](https://circt.llvm.org/docs/Dialects/Seq/).
"""

from enum import Enum
from typing import ClassVar

from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    TypeAttribute,
    i1,
)
from xdsl.dialects.hw import InnerSymAttr
from xdsl.ir import Data, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AttrSizedOperandSegments,
    IRDLOperation,
    ParametrizedAttribute,
    VarConstraint,
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
class ClockDividerOp(IRDLOperation):
    """Produces a clock divided by a power of two"""

    name = "seq.clock_div"

    pow2 = attr_def(IntegerAttr)
    clockIn = operand_def(ClockType)
    clockOut = result_def(ClockType)

    def __init__(self, clockIn: SSAValue | Operation, pow2: int | IntegerAttr):
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
        printer.print_string(" ")
        printer.print_operand(self.clockIn)
        printer.print_string(" by ")
        self.pow2.print_without_type(printer)


@irdl_op_definition
class CompRegOp(IRDLOperation):
    """
    Register a value, storing it for one cycle.
    """

    name = "seq.compreg"

    DATA_TYPE: ClassVar = VarConstraint("DataType", AnyAttr())

    inner_sym = opt_attr_def(InnerSymAttr)
    input = operand_def(DATA_TYPE)
    clk = operand_def(clock)
    reset = opt_operand_def(i1)
    reset_value = opt_operand_def(DATA_TYPE)
    power_on_value = opt_operand_def(DATA_TYPE)
    data = result_def(DATA_TYPE)

    irdl_options = (AttrSizedOperandSegments(),)

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
                printer.print_string("low")
            case ClockConstAttrData.HIGH:
                printer.print_string("high")


@irdl_op_definition
class ConstClockOp(IRDLOperation):
    """
    The constant operation produces a constant clock value.
    """

    name = "seq.const_clock"

    value = attr_def(ClockConstAttr)
    result = result_def(clock)

    @classmethod
    def parse(cls, parser: Parser) -> "ConstClockOp":
        value = ClockConstAttr(ClockConstAttr.parse_parameter_free_standing(parser))
        attrs = parser.parse_optional_attr_dict_with_reserved_attr_names(("value",))
        attrs_data = dict(attrs.data) if attrs is not None else {}
        attrs_data["value"] = value
        return ConstClockOp.create(attributes=attrs_data, result_types=[clock])

    def print(self, printer: Printer):
        printer.print_string(" ")
        self.value.print_parameter_free_standing(printer)
        printer.print_op_attributes(self.attributes, reserved_attr_names=("value",))


Seq = Dialect(
    "seq",
    [
        ClockDividerOp,
        CompRegOp,
        ConstClockOp,
    ],
    [
        ClockType,
        ClockConstAttr,
    ],
)
