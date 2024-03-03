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
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IntAttr, StringAttr
from xdsl.dialects.linalg import IteratorType, IteratorTypeAttr
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import Attribute, Dialect, Region, SSAValue
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
from xdsl.traits import IsTerminator, NoTerminator


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
    via memref.load, memref.store or any other access means, including extraction (e.g.: memref.view).
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


@irdl_op_definition
class GenericOp(IRDLOperation):
    name = "memref_stream.generic"

    inputs = var_operand_def()
    """
    Pointers to memory buffers or streams to be operated on. The corresponding stride
    pattern defines the order in which the elements of the input buffers will be read.
    """
    outputs = var_operand_def(memref.MemRefType | stream.WritableStreamType)
    """
    Pointers to memory buffers or streams to be operated on. The corresponding stride
    pattern defines the order in which the elements of the input buffers will be written
    to.
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

    iterator_types = prop_def(ArrayAttr[IteratorTypeAttr])

    body: Region = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        body: Region,
        indexing_maps: ArrayAttr[AffineMapAttr],
        iterator_types: ArrayAttr[Attribute],
        bounds: ArrayAttr[IntAttr],
    ) -> None:
        super().__init__(
            operands=[inputs, outputs],
            properties={
                "bounds": bounds,
                "indexing_maps": ArrayAttr(indexing_maps),
                "iterator_types": ArrayAttr(iterator_types),
            },
            regions=[body],
        )

    def get_static_loop_ranges(self) -> tuple[int, ...]:
        return tuple(bound.data for bound in self.bounds)

    def print(self, printer: Printer):
        printer.print_string(" {bounds = ")
        printer.print_attribute(self.bounds)
        printer.print_string(", indexing_maps = ")
        printer.print_attribute(self.indexing_maps)
        printer.print_string(", iterator_types = [")
        printer.print_list(
            self.iterator_types,
            lambda iterator_type: printer.print_string_literal(iterator_type.data),
        )
        printer.print_string("]")
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

        extra_attrs = self.attributes.copy()
        if "indexing_maps" in extra_attrs:
            del extra_attrs["indexing_maps"]
        if "iterator_types" in extra_attrs:
            del extra_attrs["iterator_types"]
        if "doc" in extra_attrs:
            del extra_attrs["doc"]
        if "library_call" in extra_attrs:
            del extra_attrs["library_call"]

        if extra_attrs:
            printer.print(" attrs = ")
            printer.print_op_attributes(extra_attrs)

        printer.print_string(" ")
        printer.print_region(self.body)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        attrs_start_pos = parser.pos
        attrs = parser.parse_optional_attr_dict()
        attrs_end_pos = parser.pos

        if "bounds" in attrs:
            bounds = attrs["bounds"]
            assert isinstance(bounds, ArrayAttr)
            bounds = cast(ArrayAttr[IntAttr], bounds)
            del attrs["bounds"]
        else:
            parser.raise_error(
                "Expected bounds for memref_stream.generic",
                attrs_start_pos,
                attrs_end_pos,
            )

        if "indexing_maps" in attrs:
            indexing_maps = attrs["indexing_maps"]
            assert isinstance(indexing_maps, ArrayAttr)
            indexing_maps = cast(ArrayAttr[AffineMapAttr], indexing_maps)
            del attrs["indexing_maps"]
        else:
            parser.raise_error(
                "Expected indexing_maps for linalg.generic",
                attrs_start_pos,
                attrs_end_pos,
            )

        if "iterator_types" in attrs:
            # Get iterator types and make sure they're an ArrayAttr
            parsed_iterator_types = attrs["iterator_types"]
            assert isinstance(parsed_iterator_types, ArrayAttr)
            parsed_iterator_types = cast(ArrayAttr[Attribute], parsed_iterator_types)
            del attrs["iterator_types"]

            # Make sure they're iterator types
            iterator_types: list[IteratorTypeAttr] = []
            for iterator_type in parsed_iterator_types:
                match iterator_type:
                    case IteratorTypeAttr():
                        iterator_types.append(iterator_type)
                    case StringAttr():
                        iterator_type = IteratorTypeAttr(
                            IteratorType(iterator_type.data)
                        )
                        iterator_types.append(iterator_type)
                    case _:
                        parser.raise_error(
                            f"Unknown iterator type {iterator_type}",
                            attrs_start_pos,
                            attrs_end_pos,
                        )
        else:
            parser.raise_error(
                "Expected iterator_types for linalg.generic",
                attrs_start_pos,
                attrs_end_pos,
            )

        if "doc" in attrs:
            doc = attrs["doc"]
            assert isinstance(doc, StringAttr)
            del attrs["doc"]
        else:
            doc = None

        if "library_call" in attrs:
            library_call = attrs["library_call"]
            assert isinstance(library_call, StringAttr)
            del attrs["library_call"]
        else:
            library_call = None

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
            body,
            indexing_maps,
            ArrayAttr(iterator_types),
            bounds,
        )
        generic.attributes |= attrs
        generic.attributes |= extra_attrs

        return generic


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "memref_stream.yield"

    traits = frozenset([IsTerminator()])


MemrefStream = Dialect(
    "memref_stream",
    [
        ReadOp,
        WriteOp,
        StreamingRegionOp,
        GenericOp,
        YieldOp,
    ],
    [],
)
