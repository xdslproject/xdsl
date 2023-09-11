""" """

from collections.abc import Sequence

from xdsl.dialects.builtin import AnyIntegerAttr
from xdsl.ir import Dialect, Operation, Region, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    VarOperand,
    attr_def,
    irdl_op_definition,
    region_def,
    var_operand_def,
)
from xdsl.traits import IsTerminator


@irdl_op_definition
class Stream(IRDLOperation):
    """ """

    name = "snitchstream.stream"

    inputs: VarOperand = var_operand_def()
    outputs: VarOperand = var_operand_def()

    dimension: AnyIntegerAttr = attr_def(AnyIntegerAttr)
    bound: AnyIntegerAttr = attr_def(AnyIntegerAttr)
    stride: AnyIntegerAttr = attr_def(AnyIntegerAttr)

    body: Region = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        body: Region,
        dimension: AnyIntegerAttr,
        bound: AnyIntegerAttr,
        stride: AnyIntegerAttr,
    ) -> None:
        super().__init__(
            operands=[inputs, outputs],
            result_types=[],
            attributes={
                "dimension": dimension,
                "bound": bound,
                "stride": stride,
            },
            regions=[body],
        )


@irdl_op_definition
class Yield(IRDLOperation):
    name = "snitchstream.yield"

    values: VarOperand = var_operand_def()

    traits = frozenset([IsTerminator()])

    def __init__(self, *operands: SSAValue | Operation) -> None:
        super().__init__(operands=[SSAValue.get(operand) for operand in operands])


SnitchStream = Dialect(
    [
        Stream,
        Yield,
    ],
    [],
)
