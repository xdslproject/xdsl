from xdsl.irdl import irdl_op_definition, IRDLOperation, region_def, var_result_def, VarOpResult, Attribute, Operation, Block, traits_def
from xdsl.ir import Region, Dialect
from collections.abc import Sequence
from xdsl.traits import SingleBlockImplicitTerminator, HasParent, IsTerminator
from xdsl.dialects.utils import AbstractYieldOperation

@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "hida_func.yield"

    traits = traits_def(lambda: frozenset([HasParent(TaskOp, DispatchOp), IsTerminator()]))

@irdl_op_definition
class DispatchOp(IRDLOperation):
    name = "hida_func.dispatch"

    region: Region = region_def()

    def __init__(self, block : Block):
        block.add_op(YieldOp())
        region = Region(block)
        super().__init__(regions=[region])


@irdl_op_definition
class TaskOp(IRDLOperation):
    name = "hida_func.task"

    region: Region = region_def()
    res: VarOpResult = var_result_def()

    traits = frozenset([SingleBlockImplicitTerminator(YieldOp), HasParent(DispatchOp)])

    def __init__(self, ops: Sequence[Operation], res_types: Sequence[Attribute]):
        region = Region(Block(ops))
        super().__init__(regions=[region], result_types=[res_types])

HIDA_func = Dialect(
    "hida_func",
    [
        TaskOp,
        DispatchOp,
    ],
    [],
)