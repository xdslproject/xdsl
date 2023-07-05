from __future__ import annotations

from xdsl.dialects.builtin import (
    ArrayAttr,
    AffineMapAttr,
    StringAttr,
    AnyShapedType,
    AnyTensorType,
)
from xdsl.ir import Dialect, Attribute, Region, SSAValue, Data, Operation
from xdsl.parser.attribute_parser import AttrParser
from xdsl.printer import Printer
from xdsl.parser import Parser
from xdsl.traits import IsTerminator
from xdsl.irdl import (
    irdl_op_definition,
    irdl_attr_definition,
    IRDLOperation,
    VarOperand,
    var_operand_def,
    var_result_def,
    VarOpResult,
    region_def,
    attr_def,
    AttrSizedOperandSegments,
    opt_attr_def,
)
from typing import Sequence
from enum import Enum


class IteratorType(Enum):
    "Iterator type for linalg trait"

    PARALLEL = "parallel"
    REDUCTION = "reduction"
    WINDOW = "window"


@irdl_attr_definition
class IteratorTypeAttr(Data[IteratorType]):
    name = "linalg.iterator_type"

    @staticmethod
    def parse_parameter(parser: AttrParser) -> IteratorType:
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


@irdl_op_definition
class Yield(IRDLOperation):
    name = "linalg.yield"

    values: VarOperand = var_operand_def()

    traits = frozenset([IsTerminator()])

    def __init__(self, *operands: SSAValue | Operation) -> None:
        super().__init__(operands=[SSAValue.get(operand) for operand in operands])


Linalg = Dialect([Generic, Yield], [IteratorTypeAttr])
