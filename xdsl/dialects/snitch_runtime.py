from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Annotated, Generic

from typing_extensions import TypeVar

from xdsl.dialects.builtin import (
    I32,
    I64,
    IndexType,
    IntegerAttr,
    IntegerType,
    i1,
    i32,
    i64,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    ConstraintVar,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import NoMemoryEffect
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
    https://github.com/pulp-platform/snitch_cluster/tree/main/sw/snRuntime
    """

    pass


class SnitchRuntimeGetInfo(SnitchRuntimeBaseOperation, ABC):
    """
    A base class for snitch runtime functions that get a certain value at runtime
    """

    result = result_def(i32)

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
    ):
        super().__init__(result_types=[i32])


class SnitchRuntimeGetInfoBool(SnitchRuntimeBaseOperation, ABC):
    """
    A base class for snitch runtime functions that get a certain value at runtime
    """

    result = result_def(i1)

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
    ):
        super().__init__(result_types=[i1])


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
    Regardless of core type, return global core index, equal to the Hart ID of the
    current core - global base Hart ID of the cluster
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
class GlobalDmCoreNumOp(SnitchRuntimeGetInfo):
    """
    Return total amount of DMA cores
    """

    name = "snrt.global_dm_core_num"


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
class IsComputeCoreOp(SnitchRuntimeGetInfoBool):
    """
    Return non-zero integer if current snitch core is a compute core
    """

    name = "snrt.is_compute_core"


@irdl_op_definition
class IsDmCoreOp(SnitchRuntimeGetInfoBool):
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

    slice_begin = result_def(slice_t_begin)
    slice_end = result_def(slice_t_end)

    traits = traits_def(NoMemoryEffect())

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
    dst = operand_def(_T)
    src = operand_def(_T)
    # Pylance was complaining about the below.
    # size = operand_def(Annotated[Attribute, i32])
    size = operand_def(i32)
    transfer_id = result_def(tx_id)

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
    dst = operand_def(_T)
    src = operand_def(_T)
    dst_stride = operand_def(i32)
    src_stride = operand_def(i32)
    size = operand_def(i32)
    repeat = operand_def(i32)
    transfer_id = result_def(tx_id)

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
class DmaStart1DOp(DmaStart1DBaseOperation[I32]):
    """
    Initiate an asynchronous 1D DMA transfer with 32-bits pointers
    """

    name = "snrt.dma_start_1d"


@irdl_op_definition
class DmaStart1DWideptrOp(DmaStart1DBaseOperation[I64]):
    """
    Initiate an asynchronous 1D DMA transfer with 64-bits wide pointers
    """

    name = "snrt.dma_start_1d_wideptr"


@irdl_op_definition
class DmaStart2DOp(DmaStart2DBaseOperation[I32]):
    """
    Initiate an asynchronous 2D DMA transfer with 32-bits pointers
    """

    name = "snrt.dma_start_2d"


@irdl_op_definition
class DmaStart2DWideptrOp(DmaStart2DBaseOperation[I64]):
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
    transfer_id = operand_def(tx_id)

    def __init__(self, transfer_id: Operation | SSAValue):
        super().__init__(operands=[transfer_id])


@irdl_op_definition
class DmaWaitAllOp(NoOperandNoResultBaseOperation):
    """
    Block until all operations on the DMA cease
    """

    name = "snrt.dma_wait_all"


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

    data_mover = operand_def(i32)
    bounds = var_operand_def(IndexType)
    strides = var_operand_def(IndexType)
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
    dm = prop_def(IntegerAttr[IntegerType])
    count = operand_def(i32)

    def __init__(
        self,
        dm: int | IntegerAttr[IntegerType],
        count: Operation | SSAValue,
    ):
        if isinstance(dm, int):
            dm = IntegerAttr(dm, i32)

        super().__init__(operands=[count], properties={"dm": dm})


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
    dm = prop_def(IntegerAttr[IntegerType])
    dim = prop_def(IntegerAttr[IntegerType])
    ptr = operand_def(i32)

    def __init__(
        self,
        dm: int | IntegerAttr[IntegerType],
        dim: int | IntegerAttr[IntegerType],
        ptr: Operation | SSAValue,
    ):
        if isinstance(dm, int):
            dm = IntegerAttr(dm, i32)

        if isinstance(dim, int):
            dim = IntegerAttr(dim, i32)

        super().__init__(
            operands=[ptr],
            properties={
                "dm": dm,
                "dim": dim,
            },
        )


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
    "snrt",
    [
        GlobalCoreBaseHartidOp,
        GlobalCoreIdxOp,
        GlobalCoreNumOp,
        GlobalComputeCoreIdxOp,
        GlobalComputeCoreNumOp,
        GlobalDmCoreNumOp,
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
