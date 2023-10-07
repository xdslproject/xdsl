from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Generic, TypeVar, cast

from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    ContainerType,
    IntAttr,
    ShapedType,
)
from xdsl.dialects.memref import MemRefType
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    OpResult,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.ir.affine import AffineMap
from xdsl.irdl import (
    AttrSizedOperandSegments,
    ConstraintVar,
    IRDLOperation,
    Operand,
    ParameterDef,
    VarOperand,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    prop_def,
    region_def,
    result_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator

_StreamTypeElement = TypeVar("_StreamTypeElement", bound=Attribute)


class StreamType(
    Generic[_StreamTypeElement],
    ParametrizedAttribute,
    TypeAttribute,
    ContainerType[_StreamTypeElement],
):
    element_type: ParameterDef[_StreamTypeElement]

    def __init__(self, element_type: _StreamTypeElement):
        super().__init__([element_type])

    def get_element_type(self) -> _StreamTypeElement:
        return self.element_type


@irdl_attr_definition
class InputStreamType(Generic[_StreamTypeElement], StreamType[_StreamTypeElement]):
    name = "input_stream"


@irdl_attr_definition
class OutputStreamType(Generic[_StreamTypeElement], StreamType[_StreamTypeElement]):
    name = "output_stream"


@irdl_op_definition
class GenericOp(IRDLOperation):
    name = "stream.generic"

    stream_inputs: VarOperand = var_operand_def(InputStreamType)
    memref_inputs: VarOperand = var_operand_def(MemRefType)

    stream_outputs: VarOperand = var_operand_def(InputStreamType)
    memref_outputs: VarOperand = var_operand_def(MemRefType)

    body: Region = region_def("single_block")

    # Trait attributes
    indexing_maps: ArrayAttr[AffineMapAttr] = prop_def(ArrayAttr[AffineMapAttr])
    """
    len(indexing_maps) == len(memref_inputs) + len(memref_outputs)
    """
    iter_count = opt_attr_def(IntAttr)
    """
    If `indexing_maps` is empty, contains the number of iterations to run the loop,
    otherwise empty.
    """

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        stream_inputs: Sequence[SSAValue],
        memref_inputs: Sequence[SSAValue],
        stream_outputs: Sequence[SSAValue],
        memref_outputs: Sequence[SSAValue],
        body: Region,
        indexing_maps: ArrayAttr[AffineMapAttr],
        iter_count: IntAttr | None,
    ) -> None:
        super().__init__(
            operands=[stream_inputs, memref_inputs, stream_outputs, memref_outputs],
            result_types=[],
            properties={
                "indexing_maps": indexing_maps,
                "iter_count": iter_count,
            },
            regions=[body],
        )

    def get_indexing_maps(self) -> list[AffineMap]:
        return [attr.data for attr in self.indexing_maps]

    def get_num_loops(self) -> int:
        if self.iter_count is not None:
            return self.iter_count.data

        return self.indexing_maps.data[0].data.num_dims

    def get_loops_to_shapes_map(self) -> AffineMap:
        """
        Returns a map to answer the question: "given an iteration space over
        the codomain, what are the subshapes of the operands involved in the
        computation".
        The default behavior is to just concatenate all the indexing maps.
        """
        result_exprs = tuple(
            res for map in self.get_indexing_maps() for res in map.results
        )

        dims = self.get_num_loops()

        # FIXME: Support symbols.
        for map in self.get_indexing_maps():
            if map.num_symbols != 0:
                raise NotImplementedError(
                    "Indexing maps with symbols not supported for now."
                )

        syms = 0
        return AffineMap(dims, syms, result_exprs)

    def get_shapes_to_loops_map(self) -> AffineMap:
        """
        Returns a map to answer the question: "Given a list of operand ranges,
        what is the subportion of the iteration space involved in the
        computation". This is the inverse problem of `get_loops_to_shapes_map`.
        Return the empty AffineMap when such an AffineMap cannot be
        constructed. The default behavior is based on a very simple inference
        procedure that only works with permutation affine maps. A more advanced
        Tensor-Comprehension like inference is possible but has proven to be
        ambiguous in unfavorable case. A safer and more robust alternative is
        to allow each op to define its own AffineMap.
        """
        loops_to_shapes = self.get_loops_to_shapes_map()
        inverse = loops_to_shapes.inverse_permutation()
        if not inverse:
            raise NotImplementedError(
                "Non-invertible maps need dynamic shapes, which are not implemented."
            )
        return inverse

    def get_static_shapes(self) -> list[int]:
        sizes: list[int] = []
        for input in self.memref_inputs:
            if isinstance(input.type, ShapedType):
                for dim in input.type.get_shape():
                    sizes.append(dim)
        for output in self.memref_outputs:
            if isinstance(output.type, ShapedType):
                for dim in output.type.get_shape():
                    sizes.append(dim)
        return sizes

    def get_static_loop_ranges(self) -> list[int]:
        if self.iter_count:
            return [self.iter_count.data]

        shapes_to_loops = self.get_shapes_to_loops_map()
        return shapes_to_loops.eval(self.get_static_shapes(), [])


@irdl_op_definition
class ReadOp(IRDLOperation):
    name = "stream.read"

    T = Annotated[Attribute, ConstraintVar("T")]

    stream: Operand = operand_def(InputStreamType[T])
    res: OpResult = result_def(T)

    def __init__(self, stream: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            assert isinstance(stream.type, InputStreamType)
            stream_type = cast(InputStreamType[Attribute], stream.type)
            result_type = stream_type.element_type
        super().__init__(operands=[stream], result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser) -> ReadOp:
        unresolved = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()
        resolved = parser.resolve_operand(unresolved, InputStreamType(result_type))
        return ReadOp(resolved, result_type)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print(self.stream)
        printer.print_string(" : ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class WriteOp(IRDLOperation):
    name = "stream.write"

    T = Annotated[Attribute, ConstraintVar("T")]

    value: Operand = operand_def(T)
    stream: Operand = operand_def(OutputStreamType[T])

    def __init__(self, value: SSAValue, stream: SSAValue):
        super().__init__(operands=[value, stream])

    @classmethod
    def parse(cls, parser: Parser) -> WriteOp:
        unresolved_stream = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        unresolved_value = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()
        resolved_value = parser.resolve_operand(unresolved_value, result_type)
        resolved_stream = parser.resolve_operand(
            unresolved_stream, InputStreamType(result_type)
        )
        return WriteOp(resolved_value, resolved_stream)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.stream)
        printer.print_string(", ")
        printer.print_ssa_value(self.value)
        printer.print_string(" : ")
        printer.print_attribute(self.value.type)


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "stream.yield"

    values: VarOperand = var_operand_def()

    traits = frozenset([IsTerminator()])

    def __init__(self, *operands: SSAValue | Operation) -> None:
        super().__init__(operands=[SSAValue.get(operand) for operand in operands])


@irdl_op_definition
class StridedReadOp(IRDLOperation):
    """
    Generates a stream reading from a memref sequentially. Currently only supports reading
    each element in order.
    """

    name = "stream.strided_read"

    T = Annotated[Attribute, ConstraintVar("T")]

    memref = operand_def(MemRefType[T])
    stream = operand_def(InputStreamType[T])

    def __init__(self, memref: SSAValue):
        assert isinstance(memref.type, MemRefType)
        memref_type = cast(MemRefType[Attribute], memref.type)
        super().__init__(
            operands=[memref], result_types=[InputStreamType(memref_type.element_type)]
        )


@irdl_op_definition
class StridedWriteOp(IRDLOperation):
    """
    Generates a stream writing from a memref sequentially. Currently only supports writing
    each element in order.
    """

    name = "stream.strided_write"

    T = Annotated[Attribute, ConstraintVar("T")]

    memref = operand_def(MemRefType[T])
    stream = operand_def(OutputStreamType[T])

    def __init__(self, memref: SSAValue):
        assert isinstance(memref.type, MemRefType)
        memref_type = cast(MemRefType[Attribute], memref.type)
        super().__init__(
            operands=[memref], result_types=[OutputStreamType(memref_type.element_type)]
        )


Linalg = Dialect(
    [
        GenericOp,
        YieldOp,
        ReadOp,
        WriteOp,
        StridedReadOp,
        StridedWriteOp,
    ],
    [
        InputStreamType,
        OutputStreamType,
    ],
)
