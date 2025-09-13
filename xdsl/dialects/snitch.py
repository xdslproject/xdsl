"""
This dialect provides operations to target features of the Snitch[1]
streaming architecture based on custom extensions to the RISC-V ISA.
This dialect works on 'riscv' types directly as all arguments are of
'riscv.reg<>' type and it is meant to be as close as possible to the asm
that aims at generating.

[1] https://pulp-platform.github.io/snitch/publications
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

from typing_extensions import TypeVar

from xdsl.dialects.builtin import ContainerType, IntAttr
from xdsl.dialects.riscv import IntRegisterType
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AttrConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    var_result_def,
)
from xdsl.utils.exceptions import VerifyException

_StreamTypeElement = TypeVar(
    "_StreamTypeElement", bound=Attribute, covariant=True, default=Attribute
)


@irdl_attr_definition
class ReadableStreamType(
    ParametrizedAttribute,
    TypeAttribute,
    ContainerType[_StreamTypeElement],
    Generic[_StreamTypeElement],
):
    name = "snitch.readable"

    element_type: _StreamTypeElement

    def get_element_type(self) -> _StreamTypeElement:
        return self.element_type

    @staticmethod
    def constr(
        element_type: AttrConstraint[_StreamTypeElement] = AnyAttr(),
    ) -> ParamAttrConstraint[ReadableStreamType[_StreamTypeElement]]:
        return ParamAttrConstraint[ReadableStreamType[_StreamTypeElement]](
            ReadableStreamType, (element_type,)
        )


@irdl_attr_definition
class WritableStreamType(
    ParametrizedAttribute,
    TypeAttribute,
    ContainerType[_StreamTypeElement],
    Generic[_StreamTypeElement],
):
    name = "snitch.writable"

    element_type: _StreamTypeElement

    def get_element_type(self) -> _StreamTypeElement:
        return self.element_type

    @staticmethod
    def constr(
        element_type: AttrConstraint[_StreamTypeElement] = AnyAttr(),
    ) -> ParamAttrConstraint[WritableStreamType[_StreamTypeElement]]:
        return ParamAttrConstraint[WritableStreamType[_StreamTypeElement]](
            WritableStreamType, (element_type,)
        )


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

    value = operand_def(IntRegisterType)
    dm = attr_def(IntAttr)
    dimension = attr_def(IntAttr)

    def __init__(
        self,
        value: Operation | SSAValue,
        dm: IntAttr,
        dimension: IntAttr,
    ):
        super().__init__(
            operands=[value],
            attributes={
                "dm": dm,
                "dimension": dimension,
            },
        )

    def verify_(self) -> None:
        if self.dimension.data >= SnitchResources.dimensions:
            raise VerifyException(
                f"dimension attribute out of range [0..{SnitchResources.dimensions - 1}], "
                f"Snitch supports up to {SnitchResources.dimensions} dimensions per streamer"
            )


class SsrSetStreamConfigOperation(IRDLOperation, ABC):
    """
    A base class for Snitch operations that set a
    configuration value for a streamer.
    """

    value = operand_def(IntRegisterType)
    dm = attr_def(IntAttr)

    def __init__(self, value: Operation | SSAValue, dm: IntAttr):
        super().__init__(
            operands=[value],
            attributes={
                "dm": dm,
            },
        )


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
class SsrEnableOp(IRDLOperation):
    """
    Enable stream semantics.
    """

    name = "snitch.ssr_enable"

    streams = var_result_def(ReadableStreamType.constr() | WritableStreamType.constr())

    def __init__(self, stream_types: Sequence[Attribute]):
        super().__init__(result_types=[stream_types])


@irdl_op_definition
class SsrDisableOp(IRDLOperation):
    """
    Disable stream semantics.
    """

    name = "snitch.ssr_disable"

    def __init__(self):
        super().__init__()


Snitch = Dialect(
    "snitch",
    [
        SsrSetDimensionBoundOp,
        SsrSetDimensionStrideOp,
        SsrSetDimensionSourceOp,
        SsrSetDimensionDestinationOp,
        SsrSetStreamRepetitionOp,
        SsrEnableOp,
        SsrDisableOp,
    ],
    [
        ReadableStreamType,
        WritableStreamType,
    ],
)
