"""
This dialect provides operations to target features of the Snitch[1]
streaming architecture based on custom extensions to the RISC-V ISA.
This dialect works on 'riscv' types directly as all arguments are of
'riscv.reg<>' type and it is meant to be as close as possible to the asm
that aims at generating.

[1] https://pulp-platform.github.io/snitch/publications
"""

from abc import ABC
from dataclasses import dataclass

from xdsl.dialects.builtin import AnyIntegerAttr
from xdsl.dialects.riscv import IntRegisterType
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import IRDLOperation, Operand, attr_def, irdl_op_definition, operand_def
from xdsl.utils.exceptions import VerifyException


@dataclass(frozen=True)
class SnitchResources:
    """
    Bounds for resources provided by the Snitch architecture.
    """

    # Number of dimensions supported by each data mover.
    dimensions: int = 4


class SsrSetDimensionConfigOperation(IRDLOperation, ABC):
    """
    A base class for Snitch operations that set a
    configuration value for a specific dimension handled by a streamer.
    """

    stream: Operand = operand_def(IntRegisterType)
    value: Operand = operand_def(IntRegisterType)
    dimension: AnyIntegerAttr = attr_def(AnyIntegerAttr)

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

    def verify_(self) -> None:
        if self.dimension.value.data >= SnitchResources.dimensions:
            raise VerifyException(
                f"dimension attribute out of range [0..{SnitchResources.dimensions-1}], "
                f"Snitch supports up to {SnitchResources.dimensions} dimensions per streamer"
            )


class SsrSetStreamConfigOperation(IRDLOperation, ABC):
    """
    A base class for Snitch operations that set a
    configuration value for a streamer.
    """

    stream: Operand = operand_def(IntRegisterType)
    value: Operand = operand_def(IntRegisterType)

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
