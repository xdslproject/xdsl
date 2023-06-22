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
from xdsl.ir import Block, OpResult, Dialect, Attribute, Operation, Region, SSAValue
from xdsl.dialects.builtin import i32, i64, IndexType
from typing import Generic, Mapping, Sequence, TypeVar, Annotated

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


"""
The different SSR data movers.
    SNRT_SSR_DM0 = 0,
    SNRT_SSR_DM1 = 1,
    SNRT_SSR_DM2 = 2,
"""
ssr_dm = i32

"""
The different dimensions.
    SNRT_SSR_1D = 0,
    SNRT_SSR_2D = 1,
    SNRT_SSR_3D = 2,
    SNRT_SSR_4D = 3,
"""
ssr_dim = i32


@irdl_op_definition
class SsrLoop1dOp(SnitchRuntimeBaseOp, ABC):
    """
    Configure an SSR data mover for a 1D loop nest.
    """

    name = "snrt.ssr_loop_1d"
    dm: Operand = operand_def(ssr_dm)
    b0: Operand = operand_def(IndexType)
    i0: Operand = operand_def(IndexType)

    def __init__(
        self,
        dm: Operation | SSAValue,
        b0: Operation | SSAValue,
        i0: Operation | SSAValue,
    ):
        super().__init__(operands=[dm, b0, i0], result_types=[])


@irdl_op_definition
class SsrLoop2dOp(SnitchRuntimeBaseOp, ABC):
    """
    Configure an SSR data mover for a 2D loop nest.
    """

    name = "snrt.ssr_loop_2d"
    dm: Operand = operand_def(ssr_dm)
    b0: Operand = operand_def(IndexType)
    b1: Operand = operand_def(IndexType)
    i0: Operand = operand_def(IndexType)
    i1: Operand = operand_def(IndexType)

    def __init__(
        self,
        dm: Operation | SSAValue,
        b0: Operation | SSAValue,
        b1: Operation | SSAValue,
        i0: Operation | SSAValue,
        i1: Operation | SSAValue,
    ):
        super().__init__(operands=[dm, b0, b1, i0, i1], result_types=[])


@irdl_op_definition
class SsrLoop3dOp(SnitchRuntimeBaseOp, ABC):
    """
    Configure an SSR data mover for a 3D loop nest.
    """

    name = "snrt.ssr_loop_3d"
    dm: Operand = operand_def(ssr_dm)
    b0: Operand = operand_def(IndexType)
    b1: Operand = operand_def(IndexType)
    b2: Operand = operand_def(IndexType)
    i0: Operand = operand_def(IndexType)
    i1: Operand = operand_def(IndexType)
    i2: Operand = operand_def(IndexType)

    def __init__(
        self,
        dm: Operation | SSAValue,
        b0: Operation | SSAValue,
        b1: Operation | SSAValue,
        b2: Operation | SSAValue,
        i0: Operation | SSAValue,
        i1: Operation | SSAValue,
        i2: Operation | SSAValue,
    ):
        super().__init__(operands=[dm, b0, b1, b2, i0, i1, i2], result_types=[])


@irdl_op_definition
class SsrLoop4dOp(SnitchRuntimeBaseOp, ABC):
    """
    Configure an SSR data mover for a 4D loop nest.
    b0: Inner-most bound (limit of loop)
    b3: Outer-most bound (limit of loop)
    s0: increment size of inner-most loop
    """

    name = "snrt.ssr_loop_4d"
    dm: Operand = operand_def(ssr_dm)
    b0: Operand = operand_def(IndexType)
    b1: Operand = operand_def(IndexType)
    b2: Operand = operand_def(IndexType)
    b3: Operand = operand_def(IndexType)
    i0: Operand = operand_def(IndexType)
    i1: Operand = operand_def(IndexType)
    i2: Operand = operand_def(IndexType)
    i3: Operand = operand_def(IndexType)

    def __init__(
        self,
        dm: Operation | SSAValue,
        b0: Operation | SSAValue,
        b1: Operation | SSAValue,
        b2: Operation | SSAValue,
        b3: Operation | SSAValue,
        i0: Operation | SSAValue,
        i1: Operation | SSAValue,
        i2: Operation | SSAValue,
        i3: Operation | SSAValue,
    ):
        super().__init__(operands=[dm, b0, b1, b2, b3, i0, i1, i2, i3], result_types=[])


@irdl_op_definition
class SsrRepeatOp(SnitchRuntimeBaseOp, ABC):
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
        super().__init__(operands=[dm, count], result_types=[])


@irdl_op_definition
class SsrEnableOp(SnitchRuntimeBaseOp, ABC):
    """
    Enable SSR.
    """

    name = "snrt.ssr_enable"

    def __init__(
        self,
    ):
        super().__init__(operands=[], result_types=[])


@irdl_op_definition
class SsrDisableOp(SnitchRuntimeBaseOp, ABC):
    """
    Disable SSR.
    """

    name = "snrt.ssr_disable"

    def __init__(
        self,
    ):
        super().__init__(operands=[], result_types=[])


@irdl_op_definition
class SsrReadOp(SnitchRuntimeBaseOp, ABC):
    """
    Start a streaming read.
    """

    name = "snrt.ssr_read"
    dm: Operand = operand_def(ssr_dm)
    dim: Operand = operand_def(ssr_dim)
    ptr: Operand = operand_def(i32)

    def __init__(
        self,
        dm: Operation | SSAValue,
        dim: Operation | SSAValue,
        ptr: Operation | SSAValue,
    ):
        super().__init__(operands=[dm, dim, ptr], result_types=[])


@irdl_op_definition
class SsrWriteOp(SnitchRuntimeBaseOp, ABC):
    """
    Start a streaming write.
    """

    name = "snrt.ssr_write"
    dm: Operand = operand_def(ssr_dm)
    dim: Operand = operand_def(ssr_dim)
    ptr: Operand = operand_def(i32)

    def __init__(
        self,
        dm: Operation | SSAValue,
        dim: Operation | SSAValue,
        ptr: Operation | SSAValue,
    ):
        super().__init__(operands=[dm, dim, ptr], result_types=[])


@irdl_op_definition
class FpuFenceOp(SnitchRuntimeBaseOp, ABC):
    """
    Synchronize the integer and float pipelines.
    """

    name = "snrt.fpu_fence"

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
