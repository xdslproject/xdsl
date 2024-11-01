from xdsl.dialects import memref
from xdsl.ir import Attribute, Dialect, Region
from xdsl.irdl import IRDLOperation, irdl_op_definition, region_def, result_def
from xdsl.traits import IsolatedFromAbove


@irdl_op_definition
class NodeOp(IRDLOperation):
    name = "hida_struct.node"

    region: Region = region_def()

    traits = frozenset([IsolatedFromAbove()])

    def __init__(self, region: Region):
        super().__init__(regions=[region])


@irdl_op_definition
class Schedule(IRDLOperation):
    name = "hida_struct.schedule"

    region: Region = region_def()

    traits = frozenset([IsolatedFromAbove()])

    def __init__(self, region: Region):
        super().__init__(regions=[region])


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
    [NodeOp, Schedule, BufferOp, Stream],
    [],
)
