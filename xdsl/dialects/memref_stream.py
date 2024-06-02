"""
A target-independent representation of streams of buffers over time.

Currently a higher-level representation of the `snitch_stream` dialect, operating on
memrefs instead of registers storing pointers.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import auto
from itertools import product
from typing import Any, cast

from typing_extensions import Self

from xdsl.dialects import memref, stream
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IntAttr, StringAttr
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    ParametrizedAttribute,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    region_def,
    var_operand_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator, NoTerminator
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


class IteratorType(StrEnum):
    "Iterator type for memref_stream Attribute"

    PARALLEL = auto()
    REDUCTION = auto()


@irdl_attr_definition
class IteratorTypeAttr(EnumAttribute[IteratorType]):
    name = "memref_stream.iterator_type"

    @classmethod
    def parallel(cls) -> IteratorTypeAttr:
        return IteratorTypeAttr(IteratorType.PARALLEL)

    @classmethod
    def reduction(cls) -> IteratorTypeAttr:
        return IteratorTypeAttr(IteratorType.REDUCTION)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> IteratorType:
        with parser.in_angle_brackets():
            return super().parse_parameter(parser)

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            super().print_parameter(printer)


@irdl_attr_definition
class StridePattern(ParametrizedAttribute):
    """
    Attribute representing the order and offsets in which elements will be read from or
    written to a stream.

    ```
    // 2D access pattern
    #pat = #memref_stream.stride_pattern<ub = [16, 8], strides = (d0, d1) -> (d0 + 1, d1 + 2)>
    // Corresponds to the following locations
    // for i in range(16):
    //   for j in range(8):
    //     yield (i + 1, j + 2)
    // Note that the upper bounds and strides go from the outermost loop inwards
    ```
    """

    name = "memref_stream.stride_pattern"

    ub: ParameterDef[ArrayAttr[IntAttr]]
    index_map: ParameterDef[AffineMapAttr]

    def __init__(self, ub: ArrayAttr[IntAttr], index_map: ParameterDef[AffineMapAttr]):
        super().__init__((ub, index_map))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_identifier("ub")
            parser.parse_punctuation("=")
            ub = ArrayAttr(
                IntAttr(i)
                for i in parser.parse_comma_separated_list(
                    parser.Delimiter.SQUARE, parser.parse_integer
                )
            )
            parser.parse_punctuation(",")
            parser.parse_identifier("index_map")
            parser.parse_punctuation("=")
            index_map = AffineMapAttr(parser.parse_affine_map())
            return (ub, index_map)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("ub = [")
            printer.print_list(self.ub, lambda attr: printer.print(attr.data))
            printer.print_string(f"], index_map = {self.index_map.data}")

    def rank(self):
        return len(self.ub)

    def verify(self) -> None:
        if len(self.ub) != self.index_map.data.num_dims:
            raise VerifyException(
                f"Expect stride pattern upper bounds {self.ub} to be equal in length to dimensions of {self.index_map}"
            )
        if self.index_map.data.num_symbols:
            raise VerifyException(
                f"Expect stride pattern map to not contain symbols: {self.index_map}"
            )

    def index_iter(self) -> Iterator[tuple[int, ...]]:
        for indices in product(*(range(bound.data) for bound in self.ub.data)):
            indices: tuple[int, ...] = indices
            yield self.index_map.data.eval(indices, ())

    def offsets(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self.index_iter())


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
    patterns = prop_def(ArrayAttr[StridePattern])
    """
    Stride patterns that define the order of the input and output streams.
    Like in linalg.generic, the indexing maps corresponding to inputs are followed by the
    indexing maps for the outputs.
    """

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = frozenset((NoTerminator(),))

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        patterns: ArrayAttr[StridePattern],
        body: Region,
    ) -> None:
        super().__init__(
            operands=[inputs, outputs],
            regions=[body],
            properties={
                "patterns": patterns,
            },
        )

    def print(self, printer: Printer):
        printer.print_string(" {patterns = ")
        printer.print_attribute(self.patterns)
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
        parser.parse_identifier("patterns")
        parser.parse_punctuation("=")

        patterns = parser.parse_attribute()
        if not isinstance(patterns, ArrayAttr):
            parser.raise_error(f"Expected ArrayAttr {patterns}")
        patterns = cast(ArrayAttr[Any], patterns)
        for pattern in patterns:
            if not isinstance(pattern, StridePattern):
                parser.raise_error(f"Expected StridePattern {pattern}")
        patterns = cast(ArrayAttr[StridePattern], patterns)

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
            patterns,
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

    def verify_(self) -> None:
        # Parallel iterator types must preceed reduction iterators
        iterator_types = self.iterator_types.data
        num_parallel = iterator_types.count(IteratorTypeAttr.parallel())
        if IteratorTypeAttr.parallel() in iterator_types[num_parallel:]:
            raise VerifyException(
                f"Unexpected order of iterator types: {[it.data.value for it in iterator_types]}"
            )


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
    [
        IteratorTypeAttr,
        StridePattern,
    ],
)
