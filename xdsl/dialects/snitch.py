from typing import Annotated

from xdsl.dialects.riscv import RegisterType

from xdsl.dialects.builtin import AnyIntegerAttr

from xdsl.ir import Dialect

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


@irdl_op_definition
class SsrSetupRepetition(IRDLOperation):
    """
    Setup repetition count for a specific data mover.
    """

    name: str = "snitch.ssr_setup_repetition"

    datamover: Annotated[Operand, RegisterType]
    repetition: Annotated[Operand, RegisterType]


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


@irdl_op_definition
class SsrEnable(IRDLOperation):
    """
    Enable stream semantics.
    """

    name: str = "snitch.ssr_enable"


@irdl_op_definition
class SsrDisable(IRDLOperation):
    """
    Disable stream semantics.
    """

    name: str = "snitch.ssr_disable"


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
