from abc import ABC, abstractmethod
from typing import Annotated, Generic, Sequence, TypeVar

from xdsl.dialects.builtin import IndexType, i32, i64
from xdsl.ir import Attribute, Dialect, OpResult
from xdsl.ir.core import Operation, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    ConstraintVar,
    IRDLOperation,
    Operand,
    VarOperand,
    irdl_op_definition,
    operand_def,
    result_def,
    var_operand_def,
)
from xdsl.utils.exceptions import VerifyException

# Transfer ID
tx_id = i32
# Indicates address range in memory, a "memory slice"
slice_t_begin = i64
slice_t_end = i64


class SnitchRuntimeBaseOperation(IRDLOperation, ABC):
    """
    A base class for ops in the Snitch Runtime dialect.
    The Snitch Runtime dialect models the snitch runtime which contains low level
    routines to manage system level aspects of snitch systems.

    This dialect is modeled after:
    https://github.com/pulp-platform/snitch/tree/b9fe5550e26ea878fb734cfc37d161f564252305/sw/snRuntime
    """

    pass


class SnitchRuntimeGetInfo(SnitchRuntimeBaseOperation, ABC):
    """
    A base class for snitch runtime functions that get a certain value at runtime
    """

    result: OpResult = result_def(i32)

    def __init__(
        self,
    ):
        super().__init__(result_types=[i32])


class NoOperandNoResultBaseOperation(SnitchRuntimeBaseOperation, ABC):
    """
    A base class for operations with no operands nor results
    """

    def __init__(
        self,
    ):
        super().__init__()


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
class ClusterHwBarrierOp(NoOperandNoResultBaseOperation):
    """
    Synchronize cores in a cluster with a hardware barrier
    """

    name = "snrt.cluster_hw_barrier"


@irdl_op_definition
class ClusterSwBarrierOp(NoOperandNoResultBaseOperation):
    """
    Synchronize with compute cores after loading data
    """

    name = "snrt.cluster_sw_barrier"


@irdl_op_definition
class GlobalBarrierOp(NoOperandNoResultBaseOperation):
    """
    Synchronize clusters globally with a global software barrier
    """

    name = "snrt.global_barrier"


_T = TypeVar("_T", bound=Attribute)


@irdl_op_definition
class BarrierRegPtrOp(SnitchRuntimeGetInfo):
    """
    Get pointer to barrier register
    """

    name = "snrt.barrier_reg_ptr"


class GetMemoryInfoBaseOperation(SnitchRuntimeBaseOperation, ABC):
    """
    Generic base class for operations returning memory slices
    """

    slice_begin: OpResult = result_def(slice_t_begin)
    slice_end: OpResult = result_def(slice_t_end)

    def __init__(
        self,
    ):
        super().__init__(result_types=[slice_t_begin, slice_t_end])


@irdl_op_definition
class GlobalMemoryOp(GetMemoryInfoBaseOperation):
    """
    Get start address of global memory
    """

    name = "snrt.global_memory"


@irdl_op_definition
class ClusterMemoryOp(GetMemoryInfoBaseOperation):
    """
    Get start address of the cluster's TCDM memory
    """

    name = "snrt.cluster_memory"


@irdl_op_definition
class ZeroMemoryOp(GetMemoryInfoBaseOperation):
    """
    Get start address of the cluster's zero memory
    """

    name = "snrt.zero_memory"


class DmaStart1DBaseOperation(SnitchRuntimeBaseOperation, Generic[_T], ABC):
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


class DmaStart2DBaseOperation(SnitchRuntimeBaseOperation, Generic[_T], ABC):
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
class DmaStart1DOp(DmaStart1DBaseOperation[Annotated[Attribute, i32]]):
    """
    Initiate an asynchronous 1D DMA transfer with 32-bits pointers
    """

    name = "snrt.dma_start_1d"


@irdl_op_definition
class DmaStart1DWideptrOp(DmaStart1DBaseOperation[Annotated[Attribute, i64]]):
    """
    Initiate an asynchronous 1D DMA transfer with 64-bits wide pointers
    """

    name = "snrt.dma_start_1d_wideptr"


@irdl_op_definition
class DmaStart2DOp(DmaStart2DBaseOperation[Annotated[Attribute, i32]]):
    """
    Initiate an asynchronous 2D DMA transfer with 32-bits pointers
    """

    name = "snrt.dma_start_2d"


@irdl_op_definition
class DmaStart2DWideptrOp(DmaStart2DBaseOperation[Annotated[Attribute, i64]]):
    """
    Initiate an asynchronous 2D DMA transfer with 64-bits wide pointers
    """

    name = "snrt.dma_start_2d_wideptr"


@irdl_op_definition
class DmaWaitOp(SnitchRuntimeBaseOperation):
    """
    Block until a transfer finishes
    """

    name = "snrt.dma_wait"
    transfer_id: Operand = operand_def(tx_id)

    def __init__(self, transfer_id: Operation | SSAValue):
        super().__init__(operands=[transfer_id])


@irdl_op_definition
class DmaWaitAllOp(NoOperandNoResultBaseOperation):
    """
    Block until all operations on the DMA cease
    """

    name = "snrt.dma_wait_all"


"""
The number of data movers determines the number of
independent memory address patterns a core can keep track
of. Since the data movers are tied to individual registers,
there need to be at least the same number of registers with
stream semantics as there are data movers. Multiple SSRs
may address the same data mover, for example to use the
data mover both in integer and FP instructions.
for more info check https://arxiv.org/pdf/1911.08356.pdf

The different SSR data movers.
    SNRT_SSR_DM0 = 0,
    SNRT_SSR_DM1 = 1,
    SNRT_SSR_DM2 = 2,
"""
ssr_dm = i32

"""
The different dimensions - those determine how many levels of nesting a loop can have, used in read and write operations.
The snitch system handles cases with up to 4, but this can be extended.
    SNRT_SSR_1D = 0,
    SNRT_SSR_2D = 1,
    SNRT_SSR_3D = 2,
    SNRT_SSR_4D = 3,
"""
ssr_dim = i32


class SsrLoopBaseOp(SnitchRuntimeBaseOperation, ABC):
    """
    Configure an SSR data mover for an n-dimensional loop nest.
    bounds (limits of loop) and strides (increments of size) are ordered from inner-most to outer-most loops.

    for example:
    for (i = 0; i < 5; i++) { //bounds[1] = 5 and strides[1] = 1
        for (j = 0; j < 6; j+=2) { //bounds[0] = 6 and strides[0] = 2
            ...
        }
    }
    """

    data_mover: Operand = operand_def(ssr_dm)
    bounds: VarOperand = var_operand_def(IndexType)
    strides: VarOperand = var_operand_def(IndexType)
    irdl_options = [AttrSizedOperandSegments()]

    def verify_(self) -> None:
        if len(self.bounds) != len(self.strides):
            raise VerifyException(
                f"the length of bounds ({len(self.bounds)}) and strides ({len(self.strides)}) must be equal."
            )
        if len(self.strides) != self.num:
            raise VerifyException(
                f"Epected {self.num} bounds and strides, got {len(self.strides)}"
            )

    def __init__(
        self,
        data_mover: Operation | SSAValue,
        bounds: Sequence[Operation | SSAValue],
        strides: Sequence[Operation | SSAValue],
    ):
        super().__init__(operands=[data_mover, bounds, strides])

    @property
    @abstractmethod
    def num(self) -> int:
        raise NotImplementedError()


@irdl_op_definition
class SsrLoop1dOp(SsrLoopBaseOp):
    """
    Configure an SSR data mover for a 1D loop nest.
    """

    name = "snrt.ssr_loop_1d"

    @property
    def num(self) -> int:
        return 1


@irdl_op_definition
class SsrLoop2dOp(SsrLoopBaseOp):
    """
    Configure an SSR data mover for a 2D loop nest.
    """

    name = "snrt.ssr_loop_2d"

    @property
    def num(self) -> int:
        return 2


@irdl_op_definition
class SsrLoop3dOp(SsrLoopBaseOp):
    """
    Configure an SSR data mover for a 3D loop nest.
    """

    name = "snrt.ssr_loop_3d"

    @property
    def num(self) -> int:
        return 3


@irdl_op_definition
class SsrLoop4dOp(SsrLoopBaseOp):
    """
    Configure an SSR data mover for a 4D loop nest.
    """

    name = "snrt.ssr_loop_4d"

    @property
    def num(self) -> int:
        return 4


@irdl_op_definition
class SsrRepeatOp(SnitchRuntimeBaseOperation, ABC):
    """
    Configure the repetition count for a stream.
    """

    name = "snrt.ssr_repeat"
    dm: Operand = operand_def(ssr_dm)
    count: Operand = operand_def(IndexType)

    def __init__(
        self,
        dm: Operation | SSAValue,
        count: Operation | SSAValue,
    ):
        super().__init__(operands=[dm, count])


@irdl_op_definition
class SsrEnableOp(NoOperandNoResultBaseOperation):
    """
    Enable SSR.
    """

    name = "snrt.ssr_enable"


@irdl_op_definition
class SsrDisableOp(NoOperandNoResultBaseOperation):
    """
    Disable SSR.
    """

    name = "snrt.ssr_disable"


class SsrReadWriteBaseOperation(SnitchRuntimeBaseOperation, ABC):
    dm: Operand = operand_def(ssr_dm)
    dim: Operand = operand_def(ssr_dim)
    ptr: Operand = operand_def(i32)

    def __init__(
        self,
        dm: Operation | SSAValue,
        dim: Operation | SSAValue,
        ptr: Operation | SSAValue,
    ):
        super().__init__(operands=[dm, dim, ptr])


@irdl_op_definition
class SsrReadOp(SsrReadWriteBaseOperation):
    """
    Start a streaming read with a given dimensionality.
    """

    name = "snrt.ssr_read"


@irdl_op_definition
class SsrWriteOp(SsrReadWriteBaseOperation):
    """
    Start a streaming write with a given dimensionality.
    """

    name = "snrt.ssr_write"


@irdl_op_definition
class FpuFenceOp(NoOperandNoResultBaseOperation):
    """
    Synchronize the integer and float pipelines.
    """

    name = "snrt.fpu_fence"


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
        IsDmCoreOp,
        ClusterHwBarrierOp,
        ClusterSwBarrierOp,
        GlobalBarrierOp,
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
        SsrLoop1dOp,
        SsrLoop2dOp,
        SsrLoop3dOp,
        SsrLoop4dOp,
        SsrRepeatOp,
        SsrEnableOp,
        SsrDisableOp,
        SsrReadOp,
        SsrWriteOp,
        FpuFenceOp,
    ],
    [],
)
