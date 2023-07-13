from __future__ import annotations

from xdsl.dialects import builtin
from xdsl.ir import Dialect, Operation, SSAValue, VerifyException
from xdsl.irdl import (
    IRDLOperation,
    VarOperand,
    attr_def,
    irdl_op_definition,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer


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

    format_str: builtin.StringAttr = attr_def(builtin.StringAttr)
    format_vals: VarOperand = var_operand_def()

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


Printf = Dialect(
    [
        PrintFormatOp,
    ],
    [],
)
