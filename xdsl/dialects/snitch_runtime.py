from abc import ABC
from xdsl.irdl import (
    irdl_op_definition,
    IRDLOperation,
    Operand,
    Operation,
    SSAValue,
    ConstraintVar,
    Attribute,
)
from xdsl.ir import OpResult, Dialect
from xdsl.dialects.builtin import IntegerType, Signedness, IndexType
from typing import Annotated, Generic, TypeVar

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


_T = TypeVar("_T", bound=Attribute)


class DmaStart1DBaseOp(SnitchRuntimeBaseOp, Generic[_T]):
    """
    Initiate an asynchronous 1D DMA transfer
    """

    T = Annotated[Attribute, ConstraintVar("T"), _T]
    dst: Annotated[Operand, T]
    src: Annotated[Operand, T]
    size: Annotated[Operand, IndexType]
    transfer_id: Annotated[OpResult, tx_id]

    def __init__(
        self,
        dst: Operation | SSAValue,
        src: Operation | SSAValue,
        size: Operation | SSAValue,
    ):
        super().__init__(operands=[dst, src, size], result_types=[tx_id])


class DmaStart2DBaseOp(SnitchRuntimeBaseOp, Generic[_T]):
    """
    Generic base class for starting asynchronous 2D DMA transfers
    """

    T = Annotated[Attribute, ConstraintVar("T"), _T]
    dst: Annotated[Operand, T]
    src: Annotated[Operand, T]
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


Dma1DWideptrOp = DmaStart1DBaseOp[Annotated[Attribute, u64]]
Dma1DOp = DmaStart1DBaseOp[Annotated[Attribute, u32]]
Dma2DWideptrOp = DmaStart2DBaseOp[Annotated[Attribute, u64]]
Dma2DOp = DmaStart2DBaseOp[Annotated[Attribute, u32]]


@irdl_op_definition
class DmaStart1DOp(Dma1DOp):
    """
    Initiate an asynchronous 1D DMA transfer with 32-bits pointers
    """

    name = "snrt.dma_start_1d"


@irdl_op_definition
class DmaStart1DWideptrOp(Dma1DWideptrOp):
    """
    Initiate an asynchronous 1D DMA transfer with 64-bits wide pointers
    """

    name = "snrt.dma_start_1d_wideptr"


@irdl_op_definition
class DmaStart2DOp(Dma2DOp):
    """
    Initiate an asynchronous 2D DMA transfer with 32-bits pointers
    """

    name = "snrt.dma_start_2d"


@irdl_op_definition
class DmaStart2DWideptrOp(Dma2DWideptrOp):
    """
    Initiate an asynchronous 2D DMA transfer with 64-bits wide pointers
    """

    name = "snrt.dma_start_2d_wideptr"


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
class DmaWaitAllOp(SnitchRuntimeBarrier):
    """
    Block until all operations on the DMA cease
    """

    name = "snrt.dma_wait_all"


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
