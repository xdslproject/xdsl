from collections.abc import Sequence

from xdsl.dialects import memref
from xdsl.ir import Attribute, Block, Dialect, Operation, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import IsolatedFromAbove, NoTerminator


@irdl_op_definition
class NodeOp(IRDLOperation):
    name = "hida_struct.node"

    args = var_operand_def()
    region: Region = region_def()

    traits = traits_def(IsolatedFromAbove(), NoTerminator())

    def __init__(self, block: Block, args: Sequence[Operation | SSAValue]):
        super().__init__(regions=[Region(block)], operands=[args])


@irdl_op_definition
class ScheduleOp(IRDLOperation):
    name = "hida_struct.schedule"

    args = var_operand_def()
    region = region_def()

    traits = traits_def(IsolatedFromAbove(), NoTerminator())

    def __init__(self, block: Block, args: Sequence[Operation | SSAValue]):
        super().__init__(regions=[Region(block)], operands=[args])


@irdl_op_definition
class BufferOp(IRDLOperation):
    name = "hida_struct.buffer"

    res = result_def()

    def _init_(self, buf_type: memref.MemRefType[Attribute]):
        super().__init__(result_types=[buf_type])


@irdl_op_definition
class Stream(IRDLOperation):
    name = "hida_struct.stream"

    def _init_(self):
        super().__init__()


HIDA_struct = Dialect(
    "hida_struct",
    [NodeOp, ScheduleOp, BufferOp, Stream],
    [],
)
