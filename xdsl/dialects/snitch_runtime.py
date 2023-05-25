from abc import ABC
from xdsl.irdl import irdl_op_definition, IRDLOperation
from xdsl.ir import OpResult
from xdsl.dialects.builtin import IntegerType, Signedness
from typing import Annotated

u32 = IntegerType(data=32,signedness=Signedness.UNSIGNED)

class SnitchRuntimeBaseOp(IRDLOperation, ABC):
    pass


class SnitchRuntimeGetInfo(SnitchRuntimeBaseOp, ABC):
    """
    A base class for snitch runtime functions that get a certain value at runtime
    """
    result: Annotated[OpResult, u32]
    def __init__(
        self,
    ):
        super().__init__(
            operands = [],
            result_types=[u32]
        )

class SnitchRuntimeBarrier(SnitchRuntimeBaseOp, ABC):
    """
    A base class for snitch runtime barriers
    """
    def __init__(
        self,
    ):
        super().__init__(
            operands = [],
            result_types=[]
        )


@irdl_op_definition
class ClusterNumOp(SnitchRuntimeGetInfo):
    name = "snrt.cluster_num"


@irdl_op_definition
class ClusterHwBarrierOp(SnitchRuntimeBarrier):
    name = "snrt.cluster_hw_barrier"
