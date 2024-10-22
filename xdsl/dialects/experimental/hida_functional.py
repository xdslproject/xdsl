from xdsl.irdl import irdl_op_definition, IRDLOperation, region_def
from xdsl.ir import Region, Dialect

@irdl_op_definition
class Task(IRDLOperation):
    name = "hida_func.task"

    region: Region = region_def()

    def __init__(self, region : Region):
        super().__init__(regions=[region])

@irdl_op_definition
class Dispatch(IRDLOperation):
    name = "hida_func.dispatch"

    region: Region = region_def()

    def _init_(self, region : Region):
        super().__init__(regions=[region])

HIDA_func = Dialect(
    "hida_func",
    [
        Task,
        Dispatch
    ],
    [],
)