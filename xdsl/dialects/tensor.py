from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

from typing_extensions import Self

from xdsl.dialects.builtin import AnySignlessIntegerOrIndexType, IndexType, TensorType
from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


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
        if self.dynamic_sizes:
            printer.print_string("(")
            printer.print_list(self.dynamic_sizes, printer.print_ssa_value)
            printer.print_string(")")
        else:
            printer.print_string("(")
            printer.print_string(")")

        printer.print_string(" : ")
        printer.print_attribute(self.tensor.type)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        pos = parser.pos
        parser.parse_punctuation("(")
        if parser.parse_optional_punctuation(")"):
            dynamic_sizes = ()
        else:
            unresolved_dynamic_sizes = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_unresolved_operand
            )
            unresolved_types = (IndexType(),) * len(unresolved_dynamic_sizes)
            parser.parse_punctuation(")")
            dynamic_sizes = parser.resolve_operands(
                unresolved_dynamic_sizes, unresolved_types, pos
            )
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()

        empty = cls(dynamic_sizes, result_type)

        return empty


@irdl_op_definition
class ReshapeOp(IRDLOperation):
    name = "tensor.reshape"

    source = operand_def(TensorType[Attribute])
    shape = operand_def(TensorType[AnySignlessIntegerOrIndexType])
    result = result_def(TensorType[Attribute])

    def __init__(self, source: SSAValue, shape: SSAValue, result_type: Attribute):
        super().__init__(
            operands=(
                source,
                shape,
            ),
            result_types=(result_type,),
        )

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.source)
        printer.print_string("(")
        printer.print_ssa_value(self.shape)
        printer.print_string(")")
        printer.print_string(" : ")
        printer.print_string("(")
        printer.print_attribute(self.source.type)
        printer.print_string(", ")
        printer.print_attribute(self.shape.type)
        printer.print_string(")")
        printer.print_string(" -> ")
        printer.print_attribute(self.result.type)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        attrs = parser.parse_optional_attr_dict()
        source = parser.parse_operand()
        parser.parse_punctuation("(")
        shape = parser.parse_operand()
        parser.parse_punctuation(")")
        parser.parse_punctuation(":")
        parser.parse_punctuation("(")
        parser.parse_comma_separated_list(Parser.Delimiter.NONE, parser.parse_type)
        parser.parse_punctuation(")")
        parser.parse_optional_punctuation("->")
        result_type = parser.parse_attribute()

        reshape = cls(
            source,
            shape,
            result_type,
        )
        reshape.attributes |= attrs
        return reshape

    def _verify(self) -> None:
        if (
            not isinstance(source_type := self.source.type, TensorType)
            or not isinstance(shape_type := self.shape.type, TensorType)
            or not isinstance(res_type := self.result.type, TensorType)
        ):
            assert (
                False
            ), "tensor elementwise operation operands and result must be of type TensorType"

        source_type = cast(TensorType[Attribute], source_type)
        shape_type = cast(TensorType[Attribute], shape_type)
        res_type = cast(TensorType[Attribute], res_type)

        if source_type.element_type != res_type.element_type:
            raise VerifyException(
                "element types of source and result tensor types should be the same"
            )

        source_type = source_type.get_shape()
        shape_type = shape_type.get_shape()
        res_type = res_type.get_shape()

        if len(shape_type) != 1:
            raise VerifyException("shape tensor must have a rank one")

        # concerns the case of static reshaping
        if math.prod(source_type) != math.prod(res_type):
            raise VerifyException(
                "source and result tensor should have the same number of elements"
            )

        shape_size = shape_type[0]
        if shape_size != len(res_type):
            raise VerifyException(
                "length of shape operand differs from the result's tensor rank"
            )


Tensor = Dialect(
    "tensor",
    [
        EmptyOp,
        ReshapeOp,
    ],
    [],
)
