from __future__ import annotations

from xdsl.dialects import arith, builtin
from xdsl.ir import Dialect, Operation, SSAValue, VerifyException
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer

i8 = builtin.IntegerType(8)


@irdl_op_definition
class PrintFormatOp(IRDLOperation):
    """
    A string formatting and printing utility.

    Can be though of as a printf equivalent but with python style format strings.

    Example uses:
    %42 = arith.constant 42 : i32
    printf.print_format "The magic number is {}", %42
    """

    name = "printf.print_format"

    format_str = attr_def(builtin.StringAttr)
    format_vals = var_operand_def()

    def __init__(self, format_str: str, *vals: SSAValue | Operation):
        super().__init__(
            operands=[vals], attributes={"format_str": builtin.StringAttr(format_str)}
        )

    def verify_(self) -> None:
        num_of_templates = self.format_str.data.count("{}")
        if not num_of_templates == len(self.format_vals):
            raise VerifyException(
                "Number of templates in template string must match number of arguments!"
            )

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_attribute(self.format_str)

        def print_val_and_type(ssa_val: SSAValue):
            printer.print_ssa_value(ssa_val)
            printer.print_string(" : ")
            printer.print_attribute(ssa_val.type)

        if len(self.format_vals) > 0:
            printer.print_string(", ")
            printer.print_list(self.operands, print_val_and_type)

        if len(self.attributes) > 1:
            attrs = self.attributes.copy()
            attrs.pop("format_str")
            printer.print_op_attributes(attrs)

    @classmethod
    def parse(cls: type[PrintFormatOp], parser: Parser) -> PrintFormatOp:
        format_str = parser.parse_str_literal()
        args: list[SSAValue] = []
        while parser.parse_optional_characters(",") is not None:
            args.append(arg := parser.parse_operand())
            parser.parse_characters(":", " - all arguments must have a type")
            arg_type = parser.parse_type()
            if arg.type != arg_type:
                parser.raise_error(f"Parsed ssa vlue {arg} must be of type {arg_type}")

        attr_dict = parser.parse_optional_attr_dict()

        if "format_str" in attr_dict:
            parser.raise_error(
                "format_str keyword is a reserved attribute for printf.print_format!"
            )

        op = PrintFormatOp(format_str, *args)

        op.attributes.update(attr_dict)

        return op


@irdl_op_definition
class PrintCharOp(IRDLOperation):
    """
    Print a single character

    Equivalent to putchar in C, but uses signless bytes as input (instead of ui32).
    Unlike the C implementation, this op does not return anything.
    """

    name = "printf.print_char"
    char = operand_def(i8)

    def __init__(self, char: SSAValue | Operation):
        super().__init__(
            operands=[char],
        )

    @staticmethod
    def from_constant_char(char: str) -> PrintCharOp:
        """
        This constructor returns a PrintCharOp that prints the value supplied
        in "char" as a python char.
        """
        if len(char) != 1:
            raise ValueError(
                f'Unexpected char value "{char}", input must be a single ascii character'
            )
        ascii_value = ord(char)
        if ascii_value > 128:
            raise ValueError("Only ascii characters are supported")
        char_constant = arith.ConstantOp.from_int_and_width(ascii_value, i8)
        return PrintCharOp(char_constant)


@irdl_op_definition
class PrintIntOp(IRDLOperation):
    """
    Print a single Integer
    """

    name = "printf.print_int"
    int = operand_def(builtin.IntegerType)

    def __init__(self, integer: SSAValue | Operation):
        super().__init__(
            operands=[integer],
        )


Printf = Dialect(
    "printf",
    [
        PrintFormatOp,
        PrintCharOp,
        PrintIntOp,
    ],
    [],
)
