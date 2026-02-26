from typing import cast

from xdsl.dialects import builtin
from xdsl.ir import (
    Dialect,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyOf,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    region_def,
    result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import HasParent, IsTerminator
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@irdl_attr_definition
class ListType(ParametrizedAttribute, TypeAttribute):
    name = "list.list"
    elem_type: builtin.IntegerType


@irdl_op_definition
class LengthOp(IRDLOperation):
    name = "list.length"

    li = operand_def(ListType)
    result = result_def(builtin.i32)

    def __init__(self, li: SSAValue):
        super().__init__(
            operands=[li],
            result_types=[builtin.i32],
        )

    assembly_format = "$li attr-dict `:` type($li) `->` type($result)"


@irdl_op_definition
class MapOp(IRDLOperation):
    name = "list.map"

    li = operand_def(ListType)
    body = region_def("single_block")
    result = result_def(ListType)

    def __init__(
        self,
        li: SSAValue,
        body: Region,
        result_element_type: builtin.IntegerType,
    ):
        super().__init__(
            operands=[li],
            regions=[body],
            result_types=[ListType(result_element_type)],
        )

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_operand(self.li)
        printer.print_string(" with (")
        printer.print_block_argument(self.body.block.args[0])
        printer.print_string(") -> ")
        printer.print_attribute(self.result.type.elem_type)
        printer.print_string(" ")
        printer.print_region(self.body, print_entry_block_args=False)
        printer.print_op_attributes(self.attributes)

    @classmethod
    def parse(cls, parser: Parser) -> "MapOp":
        unresolved_li = parser.parse_unresolved_operand("Expected list operand")
        parser.parse_keyword("with")
        parser.parse_punctuation("(")
        map_argument = parser.parse_argument()
        parser.parse_punctuation(")")
        parser.parse_punctuation("->")
        result_type = parser.parse_type()
        body = parser.parse_region((map_argument,))
        attr_dict = parser.parse_optional_attr_dict()

        if not isa(map_argument.type, builtin.IntegerType):
            parser.raise_error("input element must be an integer")
        if not isa(result_type, builtin.IntegerType):
            parser.raise_error("result element must be an integer")

        li = parser.resolve_operand(unresolved_li, ListType(map_argument.type))

        op = cls(li, body, result_type)
        op.attributes = attr_dict

        return op

    def verify_(self):
        if len(self.body.block.arg_types) != 1:
            raise VerifyException("body must have exactly one argument")

        input_list_elem_type = cast(ListType, self.li.type).elem_type
        if self.body.block.arg_types[0] != input_list_elem_type:
            raise VerifyException(
                f"argument type ({self.body.block.arg_types[0]}) does not "
                f"match element type of input list ({input_list_elem_type})"
            )

        last_op = self.body.block.last_op
        if last_op is None or not isa(last_op, YieldOp):
            raise VerifyException("missing yield terminator in body region")

        result_list_elem_type = self.result.type.elem_type
        if last_op.yielded.type != result_list_elem_type:
            raise VerifyException(
                f"yielded type ({last_op.yielded.type}) does not match "
                f"element type of result list ({result_list_elem_type})"
            )


@irdl_op_definition
class PrintOp(IRDLOperation):
    name = "list.print"

    li = operand_def(ListType)

    def __init__(self, li: SSAValue):
        super().__init__(
            operands=[li],
        )

    assembly_format = "$li attr-dict `:` type($li)"


@irdl_op_definition
class RangeOp(IRDLOperation):
    name = "list.range"

    lower = operand_def(builtin.i32)
    upper = operand_def(builtin.i32)
    result = result_def(ListType)

    def __init__(self, lower: SSAValue, upper: SSAValue, result_type: ListType):
        super().__init__(
            operands=[lower, upper],
            result_types=[result_type],
        )

    assembly_format = "$lower `to` $upper attr-dict `:` type($result)"


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "list.yield"

    yielded = operand_def(AnyOf([builtin.IntegerType, ListType]))

    traits = lazy_traits_def(
        lambda: (
            IsTerminator(),
            HasParent(MapOp),
        )
    )

    def __init__(self, yielded: SSAValue):
        super().__init__(
            operands=[yielded],
        )

    assembly_format = "$yielded attr-dict `:` type($yielded)"


LIST_DIALECT = Dialect("list", [LengthOp, MapOp, PrintOp, RangeOp, YieldOp], [ListType])
