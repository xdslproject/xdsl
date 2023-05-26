from abc import ABC
from xdsl.irdl import irdl_op_definition, IRDLOperation, Operand, Operation, SSAValue
from xdsl.ir import OpResult, Dialect
from xdsl.dialects.builtin import IntegerType, Signedness, IndexType
from typing import Annotated

u32 = IntegerType(data=32, signedness=Signedness.UNSIGNED)
u64 = IntegerType(data=64, signedness=Signedness.UNSIGNED)
tx_id = u32


class SnitchRuntimeBaseOp(IRDLOperation, ABC):
    """
    A base class for ops in the Snitch Runtime dialect.
    The Snitch Runtime dialect models the snitch runtime which contains low level
    routines to manage system level aspects of snitch systems.

    This dialect is modeled after:
    https://github.com/pulp-platform/snitch/tree/b9fe5550e26ea878fb734cfc37d161f564252305/sw/snRuntime
    """

    pass


class SnitchRuntimeGetInfo(SnitchRuntimeBaseOp, ABC):
    """
    A base class for snitch runtime functions that get a certain value at runtime
    """

    result: Annotated[OpResult, u32]

    def __init__(
        self,
    ):
        super().__init__(operands=[], result_types=[u32])


class SnitchRuntimeBarrier(SnitchRuntimeBaseOp, ABC):
    """
    A base class for snitch runtime barriers
    """

    def __init__(
        self,
    ):
        super().__init__(operands=[], result_types=[])


@irdl_op_definition
class ClusterNumOp(SnitchRuntimeGetInfo):
    """
    Probe the amount of clusters
    """

    name = "snrt.cluster_num"


@irdl_op_definition
class ClusterHwBarrierOp(SnitchRuntimeBarrier):
    """
    Synchronize cores in a cluster with a hardware barrier
    """

    name = "snrt.cluster_hw_barrier"


@irdl_op_definition
class DmaStart1DWideptrOp(SnitchRuntimeBaseOp):
    """
    Initiate an asynchronous 1D DMA transfer with wide 64-bit pointers
    """

    name = "snrt.dma_start_1d_wideptr"
    src: Annotated[Operand, u64]
    dst: Annotated[Operand, u64]
    size: Annotated[Operand, IndexType]
    transfer_id: Annotated[OpResult, tx_id]

    def __init__(
        self,
        src: Operation | SSAValue,
        dst: Operation | SSAValue,
        size: Operation | SSAValue,
    ):
        super().__init__(operands=[src, dst, size], result_types=[tx_id])


@irdl_op_definition
class DmaStart1DOp(SnitchRuntimeBaseOp):
    """
    Initiate an asynchronous 1D DMA transfer
    """

    name = "snrt.dma_start_1d"
    dst: Annotated[Operand, u32]
    src: Annotated[Operand, u32]
    size: Annotated[Operand, IndexType]
    transfer_id: Annotated[OpResult, tx_id]

    def __init__(
        self,
        dst: Operation | SSAValue,
        src: Operation | SSAValue,
        size: Operation | SSAValue,
    ):
        super().__init__(operands=[dst, src, size], result_types=[tx_id])


@irdl_op_definition
class DmaStart2DWideptrOp(SnitchRuntimeBaseOp):
    """
    Initiate an asynchronous 2D DMA transfer with wide 64-bit pointers
    """

    name = "snrt.dma_start_2d_wideptr"
    dst: Annotated[Operand, u64]
    src: Annotated[Operand, u64]
    dst_stride: Annotated[Operand, IndexType]
    src_stride: Annotated[Operand, IndexType]
    size: Annotated[Operand, IndexType]
    repeat: Annotated[Operand, IndexType]
    transfer_id: Annotated[OpResult, tx_id]

    def __init__(
        self,
        dst: Operation | SSAValue,
        src: Operation | SSAValue,
        dst_stride: Operation | SSAValue,
        src_stride: Operation | SSAValue,
        size: Operation | SSAValue,
        repeat: Operation | SSAValue,
    ):
        super().__init__(
            operands=[dst, src, dst_stride, src_stride, size, repeat],
            result_types=[tx_id],
        )


@irdl_op_definition
class DmaStart2DOp(SnitchRuntimeBaseOp):
    """
    Initiate an asynchronous 2D DMA transfer
    """

    name = "snrt.dma_start_2d"
    dst: Annotated[Operand, u32]
    src: Annotated[Operand, u32]
    dst_stride: Annotated[Operand, IndexType]
    src_stride: Annotated[Operand, IndexType]
    size: Annotated[Operand, IndexType]
    repeat: Annotated[Operand, IndexType]
    transfer_id: Annotated[OpResult, tx_id]

    def __init__(
        self,
        dst: Operation | SSAValue,
        src: Operation | SSAValue,
        dst_stride: Operation | SSAValue,
        src_stride: Operation | SSAValue,
        size: Operation | SSAValue,
        repeat: Operation | SSAValue,
    ):
        super().__init__(
            operands=[dst, src, dst_stride, src_stride, size, repeat],
            result_types=[tx_id],
        )


@irdl_op_definition
class DmaWaitOp(SnitchRuntimeBaseOp):
    """
    Block until a transfer finishes
    """

    name = "snrt.dma_wait"
    transfer_id: Annotated[Operand, tx_id]

    def __init__(self, transfer_id: Operation | SSAValue):
        super().__init__(operands=[transfer_id], result_types=[])


@irdl_op_definition
class DmaWaitAllOp(SnitchRuntimeBaseOp):
    """
    Block until all operations on the DMA cease
    """

    name = "snrt.dma_wait_all"

    def __init__(
        self,
    ):
        super().__init__(operands=[], result_types=[])


SnitchRuntime = Dialect(
    [
        ClusterNumOp,
        ClusterHwBarrierOp,
        DmaStart1DWideptrOp,
        DmaStart1DOp,
        DmaStart2DWideptrOp,
        DmaStart2DOp,
        DmaWaitOp,
        DmaWaitAllOp,
    ],
    [],
)
