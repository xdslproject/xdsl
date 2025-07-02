"""
A target-independent representation of streams of buffers over time.

Currently a higher-level representation of the `snitch_stream` dialect, operating on
memrefs instead of registers storing pointers.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import auto
from itertools import product
from typing import Any, ClassVar, Generic, cast

from typing_extensions import Self, TypeVar

from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    ContainerType,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    MemRefType,
    StringAttr,
)
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AttrSizedOperandSegments,
    GenericAttrConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    VarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import (
    HasCanonicalizationPatternsTrait,
    IsTerminator,
    NoTerminator,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.str_enum import StrEnum

_StreamTypeElement = TypeVar(
    "_StreamTypeElement", bound=Attribute, covariant=True, default=Attribute
)


@irdl_attr_definition
class ReadableStreamType(
    Generic[_StreamTypeElement],
    ParametrizedAttribute,
    TypeAttribute,
    ContainerType[_StreamTypeElement],
):
    name = "memref_stream.readable"

    element_type: _StreamTypeElement

    def get_element_type(self) -> _StreamTypeElement:
        return self.element_type

    @classmethod
    def constr(
        cls,
        element_type: GenericAttrConstraint[_StreamTypeElement] = AnyAttr(),
    ) -> ParamAttrConstraint[ReadableStreamType[_StreamTypeElement]]:
        return ParamAttrConstraint[ReadableStreamType[_StreamTypeElement]](
            ReadableStreamType, (element_type,)
        )


@irdl_attr_definition
class WritableStreamType(
    Generic[_StreamTypeElement],
    ParametrizedAttribute,
    TypeAttribute,
    ContainerType[_StreamTypeElement],
):
    name = "memref_stream.writable"

    element_type: _StreamTypeElement

    def get_element_type(self) -> _StreamTypeElement:
        return self.element_type

    @classmethod
    def constr(
        cls,
        element_type: GenericAttrConstraint[_StreamTypeElement] = AnyAttr(),
    ) -> ParamAttrConstraint[WritableStreamType[_StreamTypeElement]]:
        return ParamAttrConstraint[WritableStreamType[_StreamTypeElement]](
            WritableStreamType, (element_type,)
        )


class IteratorType(StrEnum):
    "Iterator type for memref_stream Attribute"

    PARALLEL = auto()
    """
    The corresponding iterators appear in the output.
    """
    REDUCTION = auto()
    """
    The corresponding iterators do not appear in the output.
    """
    INTERLEAVED = auto()
    """
    All inputs and outputs of the operation will be operated this many times in parallel.
    This is helpful to circumvent the latency in the loop.
    For example, if the ALU of the target has a pipeline of length 4, and the operation
    accumulates its innermost dimension, there will be stalls waiting fof the pipeline to
    clear in each iteration.
    By interleaving the loop with a factor of 4, four dimensions can be processed in
    parallel, removing the stalls.
    The corresponding iterators may appear in the output.
    """


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
    def interleaved(cls) -> IteratorTypeAttr:
        return IteratorTypeAttr(IteratorType.INTERLEAVED)

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

    ub: ArrayAttr[IntegerAttr[IndexType]]
    index_map: AffineMapAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_identifier("ub")
            parser.parse_punctuation("=")
            index = IndexType()
            ub = ArrayAttr(
                IntegerAttr(i, index)
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
            printer.print_string("ub = ")
            with printer.in_square_brackets():
                printer.print_list(
                    self.ub, lambda attr: attr.print_without_type(printer)
                )
            printer.print_string(f", index_map = {self.index_map.data}")

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
        for indices in product(*(range(bound.value.data) for bound in self.ub.data)):
            indices: tuple[int, ...] = indices
            yield self.index_map.data.eval(indices, ())

    def offsets(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self.index_iter())


@irdl_op_definition
class ReadOp(IRDLOperation):
    name = "memref_stream.read"

    T: ClassVar = VarConstraint("T", AnyAttr())

    stream = operand_def(ReadableStreamType.constr(T))
    res = result_def(T)

    assembly_format = "`from` $stream attr-dict `:` type($res)"

    def __init__(self, stream_val: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            assert isinstance(stream_type := stream_val.type, ReadableStreamType)
            stream_type = cast(ReadableStreamType[Attribute], stream_type)
            result_type = stream_type.element_type
        super().__init__(operands=[stream_val], result_types=[result_type])

    def assembly_line(self) -> str | None:
        return None


@irdl_op_definition
class WriteOp(IRDLOperation):
    name = "memref_stream.write"

    T: ClassVar = VarConstraint("T", AnyAttr())

    value = operand_def(T)
    stream = operand_def(WritableStreamType.constr(T))

    assembly_format = "$value `to` $stream attr-dict `:` type($value)"

    def __init__(self, value: SSAValue, stream: SSAValue):
        super().__init__(operands=[value, stream])

    def assembly_line(self) -> str | None:
        return None


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

    traits = traits_def(NoTerminator())

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
        with printer.indented():
            printer.print_string(" {")
            if self.patterns.data:
                printer.print_string("\npatterns = [")
                with printer.indented():
                    if self.patterns.data:
                        printer.print_string("\n")
                        printer.print_list(
                            self.patterns.data,
                            printer.print_attribute,
                            delimiter=",\n",
                        )
                printer.print_string("\n]")
            else:
                printer.print_string("\npatterns = []")
        printer.print_string("\n}")

        if self.inputs:
            printer.print_string(" ins(")
            printer.print_list(self.inputs, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(self.inputs.types, printer.print_attribute)
            printer.print_string(")")

        if self.outputs:
            printer.print_string(" outs(")
            printer.print_list(self.outputs, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(self.outputs.types, printer.print_attribute)
            printer.print_string(")")

        if self.attributes:
            printer.print_string(" attrs = ")
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


class GenericOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls):
        from xdsl.transforms.canonicalization_patterns.memref_stream import (
            RemoveUnusedInitOperandPattern,
        )

        return (RemoveUnusedInitOperandPattern(),)


@irdl_op_definition
class GenericOp(IRDLOperation):
    name = "memref_stream.generic"

    inputs = var_operand_def()
    """
    Pointers to memory buffers or streams to be operated on. The corresponding stride
    pattern defines the order in which the elements of the input buffers will be read.
    """
    outputs = var_operand_def(MemRefType.constr() | WritableStreamType.constr())
    """
    Pointers to memory buffers or streams to be operated on. The corresponding stride
    pattern defines the order in which the elements of the input buffers will be written
    to.
    """
    inits = var_operand_def()
    """
    Initial values for outputs. The outputs are at corresponding `init_indices`. The
    inits may be set only for the imperfectly nested form.
    """
    indexing_maps = prop_def(ArrayAttr[AffineMapAttr])
    """
    Stride patterns that define the order of the input and output streams.
    Like in linalg.generic, the indexing maps corresponding to inputs are followed by
    the indexing maps for the outputs.
    """
    bounds = prop_def(ArrayAttr[IntegerAttr[IndexType]])
    """
    The bounds of the iteration space, from the outermost loop inwards.
    All indexing maps must have the same number of dimensions as the length of `bounds`.
    """

    iterator_types = prop_def(ArrayAttr[IteratorTypeAttr])
    init_indices = prop_def(ArrayAttr[IntAttr])
    """
    Indices into the `outputs` that correspond to the initial values in `inits`.
    """

    doc = opt_prop_def(StringAttr)
    library_call = opt_prop_def(StringAttr)

    body = region_def("single_block")

    traits = traits_def(GenericOpHasCanonicalizationPatternsTrait())

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        inits: Sequence[SSAValue],
        body: Region,
        indexing_maps: ArrayAttr[AffineMapAttr],
        iterator_types: ArrayAttr[Attribute],
        bounds: ArrayAttr[IntegerAttr[IndexType]],
        init_indices: ArrayAttr[IntAttr],
        doc: StringAttr | None = None,
        library_call: StringAttr | None = None,
    ) -> None:
        for m in indexing_maps:
            if m.data.num_symbols:
                raise NotImplementedError(
                    f"Symbols currently not implemented in {self.name} indexing maps"
                )
        super().__init__(
            operands=[inputs, outputs, inits],
            properties={
                "bounds": bounds,
                "init_indices": init_indices,
                "indexing_maps": indexing_maps,
                "iterator_types": iterator_types,
                "doc": doc,
                "library_call": library_call,
            },
            regions=[body],
        )

    def get_static_loop_ranges(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """
        This operation can represent two sets of perfectly nested loops, or one.
        If it is one, then the first element of the returned tuple has all the loop
        bounds, and the second is empty.
        If there are two, then the first element of the returned tuple has the outer
        bounds, and the second the inner.
        Interleaved iterators are not returned in either tuple.
        """
        output_maps = self.indexing_maps.data[len(self.inputs) :]
        # min_dims will equal len(self.iterator_types) in the perfect nest case
        min_dims = min(m.data.num_dims for m in output_maps)
        num_interleaved = sum(
            it.data == IteratorType.INTERLEAVED for it in self.iterator_types
        )
        if num_interleaved:
            res = (
                tuple(
                    bound.value.data
                    for bound in self.bounds.data[: min_dims - num_interleaved]
                ),
                tuple(
                    bound.value.data
                    for bound in self.bounds.data[
                        min_dims - num_interleaved : -num_interleaved
                    ]
                ),
            )
        else:
            res = (
                tuple(bound.value.data for bound in self.bounds.data[:min_dims]),
                tuple(
                    bound.value.data
                    for bound in self.bounds.data[min_dims - num_interleaved :]
                ),
            )
        return res

    @property
    def is_imperfectly_nested(self) -> bool:
        return bool(self.get_static_loop_ranges()[1])

    def _print_init(self, printer: Printer, init: SSAValue | None):
        if init is None:
            printer.print_string("None")
        else:
            printer.print_ssa_value(init)
            printer.print_string(" : ")
            printer.print_attribute(init.type)

    def print(self, printer: Printer):
        printer.print_string(" {")
        with printer.indented():
            if self.bounds:
                printer.print_string("\nbounds = [")
                with printer.indented():
                    printer.print_list(
                        self.bounds.data,
                        lambda bound: printer.print_string(f"{bound.value.data}"),
                    )
                printer.print_string("],")
            else:
                printer.print_string("\nbounds = [],")

            if self.indexing_maps:
                printer.print_string("\nindexing_maps = [")
                with printer.indented():
                    printer.print_list(
                        self.indexing_maps.data,
                        lambda m: printer.print_string(f"\n{m}"),
                        delimiter=",",
                    )
                printer.print_string("\n],")
            else:
                printer.print_string("\nindexing_maps = [].")
            printer.print_string("\niterator_types = [")
            printer.print_list(
                self.iterator_types,
                lambda iterator_type: printer.print_string_literal(iterator_type.data),
            )
            printer.print_string("]")
            if self.doc:
                printer.print_string(",\ndoc = ")
                printer.print_attribute(self.doc)
            if self.library_call:
                printer.print_string(",\nlibrary_call = ")
                printer.print_attribute(self.library_call)
        printer.print_string("\n}")

        if self.inputs:
            printer.print_string(" ins(")
            printer.print_list(self.inputs, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(self.inputs.types, printer.print_attribute)
            printer.print_string(")")

        if self.outputs:
            printer.print_string(" outs(")
            printer.print_list(self.outputs, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(self.outputs.types, printer.print_attribute)
            printer.print_string(")")

        if self.inits:
            printer.print_string(" inits(")
            inits: list[SSAValue | None] = [None] * len(self.outputs)
            for i, val in zip(self.init_indices, self.inits):
                inits[i.data] = val
            printer.print_list(
                inits,
                lambda val: self._print_init(printer, val),
            )
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
            printer.print_string(" attrs = ")
            printer.print_op_attributes(extra_attrs)

        printer.print_string(" ")
        printer.print_region(self.body)

    @classmethod
    def _parse_init(cls, parser: Parser) -> SSAValue | None:
        if parser.parse_optional_characters("None"):
            return None
        unresolved = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        type = parser.parse_type()
        return parser.resolve_operand(unresolved, type)

    @classmethod
    def _parse_inits(
        cls, parser: Parser
    ) -> tuple[tuple[SSAValue, ...], tuple[int, ...]]:
        if not parser.parse_optional_characters("inits"):
            return ((), ())

        parser.parse_punctuation("(")
        optional_inits = parser.parse_comma_separated_list(
            Parser.Delimiter.NONE, lambda: cls._parse_init(parser)
        )
        parser.parse_punctuation(")")
        enumerated_inits = tuple(
            (i, val) for i, val in enumerate(optional_inits) if val is not None
        )
        inits = tuple(init for _, init in enumerated_inits)
        init_indices = tuple(i for i, _ in enumerated_inits)

        return (tuple(inits), init_indices)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        attrs_start_pos = parser.pos
        attrs = parser.parse_optional_attr_dict()
        attrs_end_pos = parser.pos

        if "bounds" in attrs:
            bounds = attrs["bounds"]
            assert isa(bounds, ArrayAttr[IntegerAttr[IntegerType | IndexType]]), bounds
            index = IndexType()
            bounds = ArrayAttr(
                tuple(IntegerAttr(attr.value, index) for attr in bounds.data)
            )
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
                "Expected indexing_maps for memref_stream.generic",
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
                "Expected iterator_types for memref_stream.generic",
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
            outs_types = ()
            outs = ()

        inits, init_indices = cls._parse_inits(parser)

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
            inits,
            body,
            indexing_maps,
            ArrayAttr(iterator_types),
            bounds,
            ArrayAttr(IntAttr(index) for index in init_indices),
            doc,
            library_call,
        )
        generic.attributes |= attrs
        generic.attributes |= extra_attrs

        return generic

    def verify_(self) -> None:
        if len(self.inits) != len(self.init_indices):
            raise VerifyException(
                f"Mismatching number of inits and init indices: {len(self.inits)} != {self.init_indices}"
            )

        # Parallel iterator types must preceed reduction iterators
        iterator_types = self.iterator_types.data
        num_parallel = iterator_types.count(IteratorTypeAttr.parallel())
        num_reduction = iterator_types.count(IteratorTypeAttr.reduction())
        num_interleaved = iterator_types.count(IteratorTypeAttr.interleaved())

        if IteratorTypeAttr.parallel() in iterator_types[num_parallel:]:
            raise VerifyException(
                f"Unexpected order of iterator types: {[it.data.value for it in iterator_types]}"
            )
        if (
            IteratorTypeAttr.reduction()
            in iterator_types[num_parallel + num_reduction :]
        ):
            raise VerifyException(
                f"Unexpected order of iterator types: {[it.data.value for it in iterator_types]}"
            )
        if num_interleaved > 1:
            raise VerifyException(f"Too many interleaved bounds: {num_interleaved}")
        assert num_parallel + num_reduction + num_interleaved == len(iterator_types)

        if len(self.inputs) + len(self.outputs) != len(self.indexing_maps):
            raise VerifyException(
                "The number of affine maps must match the number of inputs and outputs"
            )

        # Whether or not the operation represents an imperfect loop nest, verify that the
        # bounds of the outer + inner nests match the domain of the input affine maps
        input_count = len(self.inputs)
        input_maps = self.indexing_maps.data[:input_count]

        for i, m in enumerate(input_maps):
            if len(iterator_types) != m.data.num_dims:
                raise VerifyException(f"Invalid number of dims in indexing map {i}")

        # If the operation represents an imperfect loop nest, the bounds must match the
        # number of parallel iterators; otherwise they must match the total number of
        # iterators. In either case, they must all be the same.
        output_count = len(self.outputs)
        output_maps = self.indexing_maps.data[input_count:]

        min_dims = min(m.data.num_dims for m in output_maps)
        max_dims = max(m.data.num_dims for m in output_maps)

        if min_dims != max_dims:
            raise VerifyException(
                "The number of dims in output indexing maps must all be the same"
            )

        if min_dims not in (len(iterator_types), num_parallel + num_interleaved):
            # To signify that the output is imperfectly nested, the output affine map has
            # as many dims as parallel iterators. Otherwise, it has as many dims as
            # the total number of iterators.
            raise VerifyException(
                "The number of dims in output indexing maps must be "
                f"{len(iterator_types)} or {num_parallel + num_interleaved}"
            )

        if len(self.init_indices) != len(self.inits):
            raise VerifyException(
                "The number of inits and init_indices must be the same"
            )

        # The values of the inits must correspond to outputs where the domain of the
        # affine map has the same number of dimensions as the number of parallel
        # iterators.
        num_outputs = len(self.outputs)
        output_maps = self.indexing_maps.data[-num_outputs:]
        for index in self.init_indices:
            if not (0 <= index.data <= num_outputs):
                raise VerifyException(f"Init index out of bounds: {index.data}")
            m = output_maps[index.data]
            if m.data.num_dims != (num_parallel + num_interleaved):
                raise VerifyException(
                    "Incompatible affine map and initial value for output at index "
                    f"{index}"
                )

        interleave_factor = self.bounds.data[-1].value.data if num_interleaved else 1

        # If the operation is interleaved, use the interleaving factor to check
        # the number of arguments
        init_count = len(self.inits)
        # Outputs with initial values correspond to accumulators in the presence of
        # reduction
        acc_count = output_count if num_reduction else (output_count - init_count)
        expected_block_arg_count = (input_count + acc_count) * interleave_factor

        if expected_block_arg_count != len(self.body.block.args):
            raise VerifyException(
                f"Invalid number of arguments in block ({len(self.body.block.args)}), expected {expected_block_arg_count}"
            )


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "memref_stream.yield"

    traits = traits_def(IsTerminator())


@irdl_op_definition
class FillOp(IRDLOperation):
    name = "memref_stream.fill"

    T: ClassVar = VarConstraint("T", AnyAttr())

    memref = operand_def(memref.MemRefType.constr(element_type=T))
    value = operand_def(T)

    assembly_format = "$memref `with` $value attr-dict `:` type($memref)"

    def __init__(self, memref: SSAValue, value: SSAValue):
        super().__init__(operands=(memref, value))


MemRefStream = Dialect(
    "memref_stream",
    [
        ReadOp,
        WriteOp,
        StreamingRegionOp,
        GenericOp,
        YieldOp,
        FillOp,
    ],
    [
        ReadableStreamType,
        WritableStreamType,
        IteratorTypeAttr,
        StridePattern,
    ],
)
