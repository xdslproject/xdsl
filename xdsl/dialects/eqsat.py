"""
An embedding of equivalence classes in IR, for use in equality saturation with
non-destructive rewrites.

Please see the Equality Saturation Project for details:
https://github.com/orgs/xdslproject/projects/23

TODO: add documentation once we have end-to-end flow working:
https://github.com/xdslproject/xdsl/issues/3174
"""

from __future__ import annotations

from typing import Annotated

from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    irdl_op_definition,
    result_def,
    var_operand_def,
)
from xdsl.utils.exceptions import DiagnosticException


@irdl_op_definition
class EClassOp(IRDLOperation):
    T = Annotated[Attribute, ConstraintVar("T")]

    name = "eqsat.eclass"
    arguments = var_operand_def(T)
    result = result_def(T)

    assembly_format = "$arguments attr-dict `:` type($result)"

    def __init__(self, *arguments: SSAValue, res_type: Attribute | None = None):
        if not arguments:
            raise DiagnosticException("eclass op must have at least one operand")
        if res_type is None:
            res_type = arguments[0].type

        super().__init__(operands=[arguments], result_types=[res_type])

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_list(self.arguments, printer.print_ssa_value)
        printer.print_op_attributes(self.attributes, print_keyword=True)
        printer.print(" : ")
        printer.print_attribute(self.arguments[0].type)

    @classmethod
    def parse(cls, parser: Parser) -> EClassOp:
        pos = parser.pos
        unresolved_operands = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_unresolved_operand
        )

        attrs = parser.parse_optional_attr_dict_with_keyword()

        parser.parse_punctuation(":")

        t = parser.parse_type()

        operands = parser.resolve_operands(
            unresolved_operands, (t,) * len(unresolved_operands), pos
        )

        op = EClassOp(*operands)
        if attrs is not None:
            op.attributes.update(attrs.data)

        return op


EqSat = Dialect(
    "eqsat",
    [
        EClassOp,
    ],
)
