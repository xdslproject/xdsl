from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import Self

from xdsl.dialects.builtin import IndexType, TensorType
from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def, var_operand_def
from xdsl.parser import Parser
from xdsl.printer import Printer


@irdl_op_definition
class EmptyOp(IRDLOperation):
    name = "tensor.empty"

    dynamic_sizes = var_operand_def(IndexType)

    tensor = result_def(TensorType[Attribute])

    def __init__(self, dynamic_sizes: Sequence[SSAValue], tensor_type: Attribute):
        super().__init__(
            operands=(dynamic_sizes,),
            result_types=(tensor_type,),
        )

    def print(self, printer: Printer):
        printer.print_string("(")
        printer.print_string(")")

        if self.dynamic_sizes:
            printer.print_string("(")
            printer.print_list(self.dynamic_sizes, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(
                (i.type for i in self.dynamic_sizes), printer.print_attribute
            )
            printer.print_string(")")

        printer.print_string(" : ")
        printer.print_string("(")
        printer.print_string(")")
        printer.print_string(" -> ")
        printer.print_attribute(self.tensor.type)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        pos = parser.pos

        if parser.parse_punctuation("("):
            if parser.parse_punctuation(")"):
                unresolved_dynamic_sizes = ()
                unresolved_types = ()
            else:
                unresolved_dynamic_sizes = parser.parse_comma_separated_list(
                    Parser.Delimiter.NONE, parser.parse_unresolved_operand
                )
                parser.parse_punctuation(":")
                unresolved_types = parser.parse_comma_separated_list(
                    Parser.Delimiter.NONE, parser.parse_type
                )
                parser.parse_punctuation(")")
            dynamic_sizes = parser.resolve_operands(
                unresolved_dynamic_sizes, unresolved_types, pos
            )
        else:
            dynamic_sizes = ()

        parser.parse_punctuation(":")

        if parser.parse_punctuation("->"):
            tensor = parser.parse_comma_separated_list(
                parser.Delimiter.NONE, parser.parse_attribute
            )
        else:
            tensor = ()

        empty = cls(dynamic_sizes, tensor)

        return empty


Tensor = Dialect(
    "tensor",
    [
        EmptyOp,
    ],
    [],
)
