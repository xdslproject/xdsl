"""
This dialect provides operations to target features of the Snitch[1]
streaming architecture based on custom extensions to the RISC-V ISA.
This dialect works on 'riscv' types directly as all arguments are of
'riscv.reg<>' type and it is meant to be as close as possible to the asm
that aims at generating.

[1] https://pulp-platform.github.io/snitch/publications
"""

from abc import ABC

from typing import Annotated

from xdsl.dialects.riscv import RegisterType

from xdsl.dialects.builtin import AnyIntegerAttr

from xdsl.ir import Dialect, Operation, SSAValue

from xdsl.irdl import IRDLOperation, irdl_op_definition, Operand, OpAttr


class SsrSetDimensionConfigOperation(IRDLOperation, ABC):
    """
    A base class for Snitch operations that set a
    configuration value for a specific dimension handled by a streamer.
    """

    stream: Annotated[Operand, RegisterType]
    value: Annotated[Operand, RegisterType]
    dimension: OpAttr[AnyIntegerAttr]

    def __init__(
        self,
        stream: Operation | SSAValue,
        value: Operation | SSAValue,
        dimension: AnyIntegerAttr,
    ):
        super().__init__(
            operands=[stream, value],
            attributes={
                "dimension": dimension,
            },
        )


class SsrSetStreamConfigOperation(IRDLOperation, ABC):
    """
    A base class for Snitch operations that set a
    configuration value for a streamer.
    """

    stream: Annotated[Operand, RegisterType]
    value: Annotated[Operand, RegisterType]

    def __init__(self, stream: Operation | SSAValue, value: Operation | SSAValue):
        super().__init__(operands=[stream, value])


@irdl_op_definition
class SsrSetDimensionBoundOp(SsrSetDimensionConfigOperation):
    """
    Set the bound for one of the dimensions handled by a
    specific streamer.
    """

    name = "snitch.ssr_set_dimension_bound"


@irdl_op_definition
class SsrSetDimensionStrideOp(SsrSetDimensionConfigOperation):
    """
    Set the stride for one of the dimensions handled by a
    specific streamer.
    """

    name = "snitch.ssr_set_dimension_stride"


@irdl_op_definition
class SsrSetDimensionSourceOp(SsrSetDimensionConfigOperation):
    """
    Set the data source for one of the dimensions handled by a
    specific streamer.
    """

    name = "snitch.ssr_set_dimension_source"


@irdl_op_definition
class SsrSetDimensionDestinationOp(SsrSetDimensionConfigOperation):
    """
    Set the data destination for one of the dimensions handled by a
    specific streamer.
    """

    name = "snitch.ssr_set_dimension_destination"


@irdl_op_definition
class SsrSetStreamRepetitionOp(SsrSetStreamConfigOperation):
    """
    Setup repetition count for a specific data mover.
    """

    name = "snitch.ssr_set_stream_repetition"


@irdl_op_definition
class SsrEnable(IRDLOperation):
    """
    Enable stream semantics.
    """

    name = "snitch.ssr_enable"

    def __init__(self):
        super().__init__()


@irdl_op_definition
class SsrDisable(IRDLOperation):
    """
    Disable stream semantics.
    """

    name = "snitch.ssr_disable"

    def __init__(self):
        super().__init__()


Snitch = Dialect(
    [
        SsrSetDimensionBoundOp,
        SsrSetDimensionStrideOp,
        SsrSetDimensionSourceOp,
        SsrSetDimensionDestinationOp,
        SsrSetStreamRepetitionOp,
        SsrEnable,
        SsrDisable,
    ],
    [],
)
