from collections.abc import Sequence

from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import Dialect, Operation, Region
from xdsl.irdl import (
    Attribute,
    Block,
    IRDLOperation,
    irdl_op_definition,
    opt_result_def,
    region_def,
    traits_def,
)
from xdsl.traits import HasParent, IsTerminator, SingleBlockImplicitTerminator


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "hida_func.yield"

    traits = traits_def(
        lambda: frozenset([HasParent(TaskOp, DispatchOp), IsTerminator()])
    )


@irdl_op_definition
class DispatchOp(IRDLOperation):
    # TODO: add support for results

    name = "hida_func.dispatch"

    region = region_def()
    _results = opt_result_def()
    # _results = opt_result_def()

    traits = frozenset([SingleBlockImplicitTerminator(YieldOp)])

    def __init__(
        self, operations: list[Operation], result_types: list[Attribute | None]
    ):
        if not result_types:
            result_types = [None]
        super().__init__(regions=[Region(Block(operations))], result_types=result_types)

    # assembly_format = "attr-dict-with-keyword ( `:` type($_results)^ )? $region"


@irdl_op_definition
class TaskOp(IRDLOperation):
    name = "hida_func.task"

    region = region_def()
    _results = opt_result_def()

    traits = frozenset([SingleBlockImplicitTerminator(YieldOp), HasParent(DispatchOp)])

    def __init__(self, ops: Sequence[Operation], res_types: Sequence[Attribute]):
        region = Region(Block(ops))
        super().__init__(regions=[region], result_types=[res_types])

    assembly_format = "attr-dict-with-keyword ( `:` type($_results)^ )? $region"


HIDA_func = Dialect(
    "hida_func",
    [TaskOp, DispatchOp, YieldOp],
    [],
)
