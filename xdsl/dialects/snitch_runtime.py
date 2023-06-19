from abc import ABC
from xdsl.irdl import (
    irdl_op_definition,
    IRDLOperation,
    Operand,
    Operation,
    SSAValue,
    operand_def,
    result_def,
    ConstraintVar,
)
from xdsl.ir import OpResult, Dialect, Attribute
from xdsl.dialects.builtin import i32, i64, IndexType
from typing import Generic, TypeVar, Annotated

tx_id = i32
slice_t_begin = i64
slice_t_end = i64


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

    result: OpResult = result_def(i32)

    def __init__(
        self,
    ):
        super().__init__(operands=[], result_types=[i32])


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


@irdl_op_definition
class BarrierRegPtrOp(SnitchRuntimeGetInfo):
    """
    Get pointer to barrier register
    """

    name = "snrt.barrier_reg_ptr"


class GetMemoryInfoBaseOp(SnitchRuntimeBaseOp, ABC):
    """
    Generic base class for operations returning memory slices
    """

    slice_begin: OpResult = result_def(slice_t_begin)
    slice_end: OpResult = result_def(slice_t_end)

    def __init__(
        self,
    ):
        super().__init__(operands=[], result_types=[slice_t_begin, slice_t_end])


@irdl_op_definition
class GlobalMemoryOp(GetMemoryInfoBaseOp):
    """
    Get start address of global memory
    """

    name = "snrt.global_memory"


@irdl_op_definition
class ClusterMemoryOp(GetMemoryInfoBaseOp):
    """
    Get start address of the cluster's TCDM memory
    """

    name = "snrt.cluster_memory"


@irdl_op_definition
class ZeroMemoryOp(GetMemoryInfoBaseOp):
    """
    Get start address of the cluster's zero memory
    """

    name = "snrt.zero_memory"


class DmaStart1DBaseOp(SnitchRuntimeBaseOp, Generic[_T], ABC):
    """
    Initiate an asynchronous 1D DMA transfer
    """

    T = Annotated[Attribute, ConstraintVar("T"), _T]
    dst: Operand = operand_def(T)
    src: Operand = operand_def(T)
    size: Operand = operand_def(IndexType)
    transfer_id: OpResult = result_def(tx_id)

    def __init__(
        self,
        dst: Operation | SSAValue,
        src: Operation | SSAValue,
        size: Operation | SSAValue,
    ):
        super().__init__(operands=[dst, src, size], result_types=[tx_id])


class DmaStart2DBaseOp(SnitchRuntimeBaseOp, Generic[_T], ABC):
    """
    Generic base class for starting asynchronous 2D DMA transfers
    """

    T = Annotated[Attribute, ConstraintVar("T"), _T]
    dst: Operand = operand_def(T)
    src: Operand = operand_def(T)
    dst_stride: Operand = operand_def(IndexType)
    src_stride: Operand = operand_def(IndexType)
    size: Operand = operand_def(IndexType)
    repeat: Operand = operand_def(IndexType)
    transfer_id: OpResult = result_def(tx_id)

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
class DmaStart1DOp(DmaStart1DBaseOp[Annotated[Attribute, i32]]):
    """
    Initiate an asynchronous 1D DMA transfer with 32-bits pointers
    """

    name = "snrt.dma_start_1d"


@irdl_op_definition
class DmaStart1DWideptrOp(DmaStart1DBaseOp[Annotated[Attribute, i64]]):
    """
    Initiate an asynchronous 1D DMA transfer with 64-bits wide pointers
    """

    name = "snrt.dma_start_1d_wideptr"


@irdl_op_definition
class DmaStart2DOp(DmaStart2DBaseOp[Annotated[Attribute, i32]]):
    """
    Initiate an asynchronous 2D DMA transfer with 32-bits pointers
    """

    name = "snrt.dma_start_2d"


@irdl_op_definition
class DmaStart2DWideptrOp(DmaStart2DBaseOp[Annotated[Attribute, i64]]):
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
    transfer_id: Operand = operand_def(tx_id)

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
        BarrierRegPtrOp,
        GlobalMemoryOp,
        ClusterMemoryOp,
        ZeroMemoryOp,
        DmaStart1DWideptrOp,
        DmaStart1DOp,
        DmaStart2DWideptrOp,
        DmaStart2DOp,
        DmaWaitOp,
        DmaWaitAllOp,
    ],
    [],
)
