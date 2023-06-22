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
class GlobalCoreBaseHartidOp(SnitchRuntimeGetInfo):
    """
    Get the current core's global base Hart ID
    """

    name = "snrt.global_core_base_hartid"


@irdl_op_definition
class GlobalCoreIdxOp(SnitchRuntimeGetInfo):
    """
    Regardless of core type, return global core index, equal to the Hart ID of the current core - global base Hart ID of the cluster 
    """

    name = "snrt.global_core_idx"


@irdl_op_definition
class GlobalCoreNumOp(SnitchRuntimeGetInfo):
    """
    Return total amount of cores including DMA cores per cluster
    """

    name = "snrt.global_core_num"


@irdl_op_definition
class GlobalComputeCoreIdxOp(SnitchRuntimeGetInfo):
    """
    For compute core, return global core index
    """

    name = "snrt.global_compute_core_idx"


@irdl_op_definition
class GlobalComputeCoreNumOp(SnitchRuntimeGetInfo):
    """
    Return total amount of compute cores per cluster
    """

    name = "snrt.global_compute_core_num"


@irdl_op_definition
class GlobalDmCoreIdxOp(SnitchRuntimeGetInfo):
    """
    For DMA core, return global core index
    """

    name = "snrt.global_dm_core_idx"


@irdl_op_definition
class GlobalDmCoreNumOp(SnitchRuntimeGetInfo):
    """
    Return total amount of DMA cores
    """

    name = "snrt.global_dm_core_num"


@irdl_op_definition
class ClusterCoreBaseHartidOp(SnitchRuntimeGetInfo):
    """
    Return Base Hart ID for this cluster
    """

    name = "snrt.cluster_core_base_hartid"


@irdl_op_definition
class ClusterCoreIdxOp(SnitchRuntimeGetInfo):
    """
    Return cluster identifier
    """

    name = "snrt.cluster_core_idx"


@irdl_op_definition
class ClusterCoreNumOp(SnitchRuntimeGetInfo):
    """
    Return total amount of cores for the current cluster
    """

    name = "snrt.cluster_core_num"


@irdl_op_definition
class ClusterComputeCoreIdxOp(SnitchRuntimeGetInfo):
    """
    For compute cores return core ID within a cluster
    """

    name = "snrt.cluster_compute_core_idx"


@irdl_op_definition
class ClusterComputeCoreNumOp(SnitchRuntimeGetInfo):
    """
    Return number of compute cores for the current cluster
    """

    name = "snrt.cluster_compute_core_num"


@irdl_op_definition
class ClusterDmCoreIdxOp(SnitchRuntimeGetInfo):
    """
    For DMA cores, return core ID within a cluster, currently hardcoded to number of all cores - 1
    """

    name = "snrt.cluster_dm_core_idx"


@irdl_op_definition
class ClusterDmCoreNumOp(SnitchRuntimeGetInfo):
    """
    Return amount of DMA cores in this cluster, in the current runtime, this is hardcoded to 1
    """

    name = "snrt.cluster_dm_core_num"


@irdl_op_definition
class ClusterIdxOp(SnitchRuntimeGetInfo):
    """
    Return i32 identifier for the cluster this core is a part of
    """

    name = "snrt.cluster_idx"


@irdl_op_definition
class ClusterNumOp(SnitchRuntimeGetInfo):
    """
    Probe the amount of clusters
    """

    name = "snrt.cluster_num"


@irdl_op_definition
class IsComputeCoreOp(SnitchRuntimeGetInfo):
    """
    Return non-zero integer if current snitch core is a compute core
    """

    name = "snrt.is_compute_core"


@irdl_op_definition
class IsDmCoreOp(SnitchRuntimeGetInfo):
    """
    Return non-zero integer if current snitch core is a DMA core
    """

    name = "snrt.is_dm_core"


@irdl_op_definition
class ClusterHwBarrierOp(SnitchRuntimeBarrier):
    """
    Synchronize cores in a cluster with a hardware barrier
    """

    name = "snrt.cluster_hw_barrier"


@irdl_op_definition
class ClusterSwBarrierOp(SnitchRuntimeBarrier):
    """
    Synchronize with compute cores after loading data
    """

    name = "snrt.cluster_sw_barrier"


@irdl_op_definition
class GlobalBarrierOp(SnitchRuntimeBarrier):
    """
    Synchronize clusters globally with a global software barrier
    """

    name = "snrt.global_barrier"


_T = TypeVar("_T", bound=Attribute)


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
        GlobalCoreBaseHartidOp,
        GlobalCoreIdxOp,
        GlobalCoreNumOp,
        GlobalComputeCoreIdxOp,
        GlobalComputeCoreNumOp,
        GlobalDmCoreIdxOp,
        GlobalDmCoreNumOp,
        ClusterCoreBaseHartidOp,
        ClusterCoreIdxOp,
        ClusterCoreNumOp,
        ClusterComputeCoreIdxOp,
        ClusterComputeCoreNumOp,
        ClusterDmCoreIdxOp,
        ClusterDmCoreNumOp,
        ClusterIdxOp,
        ClusterNumOp,
        IsComputeCoreOp,
        IsComputeDmOp,
        ClusterHwBarrierOp,
        ClusterSwBarrierOp,
        GlobalBarrierOp,
        DmaStart1DWideptrOp,
        DmaStart1DOp,
        DmaStart2DWideptrOp,
        DmaStart2DOp,
        DmaWaitOp,
        DmaWaitAllOp,
    ],
    [],
)
