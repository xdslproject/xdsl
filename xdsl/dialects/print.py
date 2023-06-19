from __future__ import annotations

from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    VarOperand,
    attr_def,
    var_operand_def,
)
from xdsl.dialects import builtin
from xdsl.ir import Dialect, SSAValue, Operation, VerifyException


@irdl_op_definition
class PrintLnOp(IRDLOperation):
    """
    A string formatting and printing utility.

    Can be though of as a printf equivalent but with python style format strings.

    Example uses:
    %42 = arith.constant 42 : i32
    print.println "The magic number is {}", %42
    """

    name = "print.println"

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
            printer.print_attribute(ssa_val.typ)

        if len(self.format_vals) > 0:
            printer.print_string(", ")
            printer.print_list(self.operands, print_val_and_type)

        if len(self.attributes) > 1:
            attrs = self.attributes.copy()
            attrs.pop("format_str")
            printer.print_string(" {")
            printer.print_attribute_dictionary(
                attrs, printer.print_string, printer.print_attribute
            )
            printer.print_string("}")

    @classmethod
    def parse(cls: type[PrintLnOp], parser: Parser) -> PrintLnOp:
        format_str = parser.parse_str_literal()
        args: list[SSAValue] = []
        while parser.parse_optional_characters(",") is not None:
            args.append(arg := parser.parse_operand())
            parser.parse_characters(":", " - all arguments must have a type")
            typ = parser.parse_type()
            if arg.typ != typ:
                parser.raise_error(f"Parsed ssa vlue {arg} must be of type {typ}")

        attr_dict = parser.parse_optional_attr_dict()

        if "format_str" in attr_dict:
            parser.raise_error(
                "format_str keyword is a reserved attribute for print.println!"
            )

        op = PrintLnOp(format_str, *args)

        op.attributes.update(attr_dict)

        return op


Print = Dialect(
    [
        PrintLnOp,
    ],
    [],
)
