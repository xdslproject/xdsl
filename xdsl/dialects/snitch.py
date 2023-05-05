"""This dialect provides operations to address features of the Snitch[1]
architecture; it works on riscv types directly as all arguments are of
riscv.reg<> type and it is meant to be as close as possible to the asm
that aims at generating.

[1] https://pulp-platform.github.io/snitch/publications
"""

from typing import Annotated

from xdsl.dialects.riscv import RegisterType

from xdsl.dialects.builtin import AnyIntegerAttr

from xdsl.ir import Dialect, Operation, SSAValue

from xdsl.irdl import IRDLOperation, irdl_op_definition, Operand, OpAttr


@irdl_op_definition
class SsrSetupShape(IRDLOperation):
    """
    Setup the shape (bound and stride) for an arbitrary dimension handled by a
    specific data mover.
    """

    name: str = "snitch.ssr_setup_shape"

    datamover: Annotated[Operand, RegisterType]
    bound: Annotated[Operand, RegisterType]
    stride: Annotated[Operand, RegisterType]
    dimension: OpAttr[AnyIntegerAttr]

    def __init__(
        self,
        datamover: Operation | SSAValue,
        bound: Operation | SSAValue,
        stride: Operation | SSAValue,
        dimension: AnyIntegerAttr,
    ):
        super().__init__(
            operands=[datamover, bound, stride],
            attributes={
                "dimension": dimension,
            },
        )


@irdl_op_definition
class SsrSetupRepetition(IRDLOperation):
    """
    Setup repetition count for a specific data mover.
    """

    name: str = "snitch.ssr_setup_repetition"

    datamover: Annotated[Operand, RegisterType]
    repetition: Annotated[Operand, RegisterType]

    def __init__(
        self,
        datamover: Operation | SSAValue,
        repetition: Operation | SSAValue,
    ):
        super().__init__(operands=[datamover, repetition])


@irdl_op_definition
class SsrRead(IRDLOperation):
    """
    Setup a dimension for a data mover to be a read stream on a
    specific base address.
    """

    name: str = "snitch.ssr_read"

    datamover: Annotated[Operand, RegisterType]
    address: Annotated[Operand, RegisterType]
    dimension: OpAttr[AnyIntegerAttr]

    def __init__(
        self,
        datamover: Operation | SSAValue,
        address: Operation | SSAValue,
        dimension: AnyIntegerAttr,
    ):
        super().__init__(
            operands=[datamover, address],
            attributes={
                "dimension": dimension,
            },
        )


@irdl_op_definition
class SsrWrite(IRDLOperation):
    """
    Setup a dimension for a data mover to be a write stream on a
    specific base address.
    """

    name: str = "snitch.ssr_write"

    datamover: Annotated[Operand, RegisterType]
    address: Annotated[Operand, RegisterType]
    dimension: OpAttr[AnyIntegerAttr]

    def __init__(
        self,
        datamover: Operation | SSAValue,
        address: Operation | SSAValue,
        dimension: AnyIntegerAttr,
    ):
        super().__init__(
            operands=[datamover, address],
            attributes={
                "dimension": dimension,
            },
        )


@irdl_op_definition
class SsrEnable(IRDLOperation):
    """
    Enable stream semantics.
    """

    name: str = "snitch.ssr_enable"

    def __init__(self):
        super().__init__()


@irdl_op_definition
class SsrDisable(IRDLOperation):
    """
    Disable stream semantics.
    """

    name: str = "snitch.ssr_disable"

    def __init__(self):
        super().__init__()


Snitch = Dialect(
    [
        SsrSetupShape,
        SsrSetupRepetition,
        SsrRead,
        SsrWrite,
        SsrEnable,
        SsrDisable,
    ],
    [],
)
