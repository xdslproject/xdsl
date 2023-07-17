from __future__ import annotations

from enum import Enum
from typing import Sequence

from xdsl.dialects.builtin import (
    AffineMapAttr,
    AnyShapedType,
    AnyTensorType,
    ArrayAttr,
    ShapedType,
    StringAttr,
)
from xdsl.ir import Attribute, Data, Dialect, Operation, Region, SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    VarOperand,
    VarOpResult,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    opt_attr_def,
    region_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser.attribute_parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator


class IteratorType(Enum):
    "Iterator type for linalg trait"

    PARALLEL = "parallel"
    REDUCTION = "reduction"
    WINDOW = "window"


@irdl_attr_definition
class IteratorTypeAttr(Data[IteratorType]):
    name = "linalg.iterator_type"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> IteratorType:
        if parser.parse_optional_keyword("parallel") is not None:
            return IteratorType.PARALLEL
        if parser.parse_optional_keyword("reduction") is not None:
            return IteratorType.REDUCTION
        if parser.parse_optional_keyword("window") is not None:
            return IteratorType.WINDOW
        parser.raise_error("`parallel`, `reduction` or `window` expected")

    def print_parameter(self, printer: Printer) -> None:
        data = self.data
        match data:
            case IteratorType.PARALLEL:
                printer.print_string("parallel")
            case IteratorType.REDUCTION:
                printer.print_string("reduction")
            case IteratorType.WINDOW:
                printer.print_string("window")


@irdl_op_definition
class Generic(IRDLOperation):
    name = "linalg.generic"

    inputs: VarOperand = var_operand_def()
    outputs: VarOperand = var_operand_def(AnyShapedType())

    res: VarOpResult = var_result_def(AnyTensorType)

    body: Region = region_def("single_block")

    # Trait attributes
    indexing_maps: ArrayAttr[AffineMapAttr] = attr_def(ArrayAttr[AffineMapAttr])
    iterator_types: ArrayAttr[IteratorTypeAttr] = attr_def(ArrayAttr[IteratorTypeAttr])
    doc: StringAttr | None = opt_attr_def(StringAttr)
    library_call: StringAttr | None = opt_attr_def(StringAttr)

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        body: Region,
        indexing_maps: Sequence[AffineMapAttr],
        iterator_types: Sequence[Attribute],
        doc: StringAttr | None = None,
        library_call: StringAttr | None = None,
    ) -> None:
        super().__init__(
            operands=[inputs, outputs],
            result_types=[[]],
            attributes={
                "indexing_maps": ArrayAttr(indexing_maps),
                "iterator_types": ArrayAttr(iterator_types),
                "doc": doc,
                "library_call": library_call,
            },
            regions=[body],
        )

    def get_indexing_maps(self) -> list[AffineMap]:
        return [attr.data for attr in self.indexing_maps]

    def get_num_loops(self) -> int:
        return self.indexing_maps.data[0].data.num_dims

    def get_loops_to_shapes_map(self) -> AffineMap:
        """
        Returns a map to answer the question: "given an iteration space over
        the codomain, what are the subshapes of the operands involved in the
        computation".
        The default behavior is to just concatenate all the indexing maps.
        """
        result_exprs = [res for map in self.get_indexing_maps() for res in map.results]

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
        for input in self.inputs:
            if isinstance(input.type, ShapedType):
                for dim in input.type.get_shape():
                    sizes.append(dim)
        for output in self.outputs:
            if isinstance(output.type, ShapedType):
                for dim in output.type.get_shape():
                    sizes.append(dim)
        return sizes

    def get_static_loop_ranges(self) -> list[int]:
        shapes_to_loops = self.get_shapes_to_loops_map()
        return shapes_to_loops.eval(self.get_static_shapes(), [])


@irdl_op_definition
class Yield(IRDLOperation):
    name = "linalg.yield"

    values: VarOperand = var_operand_def()

    traits = frozenset([IsTerminator()])

    def __init__(self, *operands: SSAValue | Operation) -> None:
        super().__init__(operands=[SSAValue.get(operand) for operand in operands])


Linalg = Dialect([Generic, Yield], [IteratorTypeAttr])
