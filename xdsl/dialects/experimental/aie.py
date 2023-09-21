from __future__ import annotations

from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    AnyAttr,
    ArrayAttr,
    IndexType,
    IntegerAttr,
    StringAttr,
    i32,
)
from xdsl.ir import Dialect, OpResult
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    attr_def,
    irdl_op_definition,
    operand_def,
    result_def,
)


@irdl_op_definition
class AMSelOp(IRDLOperation):
    name = "AIE.amsel"
    arbiterID: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    msel: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    result: OpResult = result_def(IndexType())

    def __init__(self, arbiterID: IntegerAttr[i32], msel: IntegerAttr[i32]):
        super().__init__(
            attributes={"arbiterID": arbiterID, "msel": msel},
            result_types=[IndexType()],
        )


@irdl_op_definition
class BufferOp(IRDLOperation):
    name = "buffer"
    tile: Operand = operand_def(IndexType())
    shape: ArrayAttr[i32] = attr_def(ArrayAttr[i32])
    element_type: AnyAttr = attr_def(AnyAttr())
    buffer: OpResult = result_def(memref.MemRefType)

    def __init__(self, tile: IndexType(), element_type: AnyAttr, shape: ArrayAttr[i32]):
        buffer_type = memref.MemRefType.from_element_type_and_shape(element_type, shape)
        super().__init__(operands=[tile], result_types=[buffer_type])


@irdl_op_definition
class TileOp(IRDLOperation):
    name = "tile"
    col: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    row: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    result: OpResult = result_def(IndexType())

    def __init__(self, col: IntegerAttr[i32], row: IntegerAttr[i32]):
        super().__init__(
            attributes={"col": col, "row": row}, result_types=[IndexType()]
        )


@irdl_op_definition
class ConnectOp(IRDLOperation):
    name = "connect"
    sourceBundle: StringAttr = attr_def(StringAttr)
    sourceChannel: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    destBundle: StringAttr = attr_def(StringAttr)
    destChannel: IntegerAttr[i32] = attr_def(IntegerAttr[i32])

    def __init__(
        self,
        sourceBundle: StringAttr,
        sourceChannel: IntegerAttr[i32],
        destBundle: StringAttr,
        destChannel: IntegerAttr[i32],
    ):
        super().__init__(
            attributes={
                "sourceBundle": sourceBundle,
                "sourceChannel": sourceChannel,
                "destBundle": destBundle,
                "destChannel": destChannel,
            }
        )


AIE = Dialect([AMSelOp, BufferOp, TileOp], [])
