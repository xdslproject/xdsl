"""
A target-independent representation of streams of buffers over time.

Currently a higher-level representation of the `snitch_stream` dialect, operating on
memrefs instead of registers storing pointers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from typing_extensions import Self

from xdsl.dialects import memref, stream
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IntAttr
from xdsl.ir import Dialect, Region, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    prop_def,
    region_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import NoTerminator


@irdl_op_definition
class ReadOp(stream.ReadOperation):
    name = "memref_stream.read"


@irdl_op_definition
class WriteOp(stream.WriteOperation):
    name = "memref_stream.write"


@irdl_op_definition
class StreamingRegionOp(IRDLOperation):
    """
    An operation that creates streams from access patterns, which are only available to
    read from and write to within the body of the operation.

    Within the loop body, memrefs that are streamed must not be otherwise accessed
    via memref.load, memref.store or any other access mean, including extraction (e.g.: memref.view).
    """

    name = "memref_stream.streaming_region"

    inputs = var_operand_def(memref.MemRefType)
    """
    Pointers to memory buffers that will be streamed. The corresponding stride pattern
    defines the order in which the elements of the input buffers will be read.
    """
    outputs = var_operand_def(memref.MemRefType)
    """
    Pointers to memory buffers that will be streamed. The corresponding stride pattern
    defines the order in which the elements of the input buffers will be written to.
    """
    indexing_maps = prop_def(ArrayAttr[AffineMapAttr])
    """
    Stride patterns that define the order of the input and output streams.
    Like in linalg.generic, the indexing maps corresponding to inputs are followed by the
    indexing maps for the outputs.
    """
    bounds = prop_def(ArrayAttr[IntAttr])
    """
    The bounds of the iteration space, from the outermost loop inwards. All indexing maps must have the same number of dimensions as the length of `bounds`.
    """

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = frozenset((NoTerminator(),))

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        indexing_maps: ArrayAttr[AffineMapAttr],
        bounds: ArrayAttr[IntAttr],
        body: Region,
    ) -> None:
        super().__init__(
            operands=[inputs, outputs],
            regions=[body],
            properties={
                "bounds": bounds,
                "indexing_maps": indexing_maps,
            },
        )

    def print(self, printer: Printer):
        printer.print_string(" {bounds = [")
        printer.print_list(self.bounds, lambda bound: printer.print(bound.data))
        printer.print_string("], indexing_maps = ")
        printer.print_attribute(self.indexing_maps)
        printer.print_string("}")

        if self.inputs:
            printer.print_string(" ins(")
            printer.print_list(self.inputs, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list((i.type for i in self.inputs), printer.print_attribute)
            printer.print_string(")")

        if self.outputs:
            printer.print_string(" outs(")
            printer.print_list(self.outputs, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list((o.type for o in self.outputs), printer.print_attribute)
            printer.print_string(")")

        if self.attributes:
            printer.print(" attrs = ")
            printer.print_op_attributes(self.attributes)

        printer.print_string(" ")
        printer.print_region(self.body)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        parser.parse_punctuation("{")
        parser.parse_identifier("bounds")
        parser.parse_punctuation("=")
        bound_vals = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_integer
        )
        bounds = ArrayAttr(IntAttr(bound) for bound in bound_vals)

        parser.parse_punctuation(",")
        parser.parse_identifier("indexing_maps")
        parser.parse_punctuation("=")
        indexing_maps = parser.parse_attribute()
        indexing_maps = cast(ArrayAttr[AffineMapAttr], indexing_maps)

        parser.parse_punctuation("}")

        pos = parser.pos
        if parser.parse_optional_characters("ins"):
            parser.parse_punctuation("(")
            unresolved_ins = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_unresolved_operand
            )
            parser.parse_punctuation(":")
            ins_types = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_type
            )
            parser.parse_punctuation(")")
            ins = parser.resolve_operands(unresolved_ins, ins_types, pos)
        else:
            ins = ()

        pos = parser.pos
        if parser.parse_optional_characters("outs"):
            parser.parse_punctuation("(")
            unresolved_outs = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_unresolved_operand
            )
            parser.parse_punctuation(":")
            outs_types = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_type
            )
            parser.parse_punctuation(")")
            outs = parser.resolve_operands(unresolved_outs, outs_types, pos)
        else:
            outs = ()

        if parser.parse_optional_keyword("attrs"):
            parser.parse_punctuation("=")
            extra_attrs = parser.expect(
                parser.parse_optional_attr_dict, "expect extra attributes"
            )
        else:
            extra_attrs = {}

        body = parser.parse_region()

        generic = cls(
            ins,
            outs,
            indexing_maps,
            bounds,
            body,
        )
        generic.attributes |= extra_attrs

        return generic


MemrefStream = Dialect(
    "memref_stream",
    [
        ReadOp,
        WriteOp,
        StreamingRegionOp,
    ],
    [],
)
