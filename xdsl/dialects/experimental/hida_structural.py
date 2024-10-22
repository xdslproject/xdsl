from xdsl.irdl import irdl_op_definition, IRDLOperation, region_def
from xdsl.ir import Region, Dialect
from xdsl.traits import IsolatedFromAbove

@irdl_op_definition
class Node(IRDLOperation):
    name = "hida_struct.node"

    region : Region = region_def()

    traits = frozenset([IsolatedFromAbove()])

    def __init__(self, region : Region):
        super().__init__(regions=[region])

@irdl_op_definition
class Schedule(IRDLOperation):
    name = "hida_struct.schedule"

    region : Region = region_def()

    traits = frozenset([IsolatedFromAbove()])

    def __init__(self, region : Region):
        super().__init__(regions=[region])

@irdl_op_definition
class Buffer(IRDLOperation):
    name = "hida_struct.buffer"

    def _init_(self):
        super().__init__()

@irdl_op_definition
class Stream(IRDLOperation):
    name = "hida_struct.stream"

    def _init_(self):
        super().__init__()

HIDA_struct = Dialect(
    "hida_struct",
    [
        Node,
        Schedule,
        Buffer,
        Stream
    ],
    [],
)