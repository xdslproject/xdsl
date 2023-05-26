from xdsl.dialects import snitch_runtime
from xdsl.utils.test_value import TestSSAValue
from xdsl.dialects.builtin import IntegerType, Signedness, IndexType

u32 = IntegerType(data=32, signedness=Signedness.UNSIGNED)


def test_snrt_op():
    cluster_num = snitch_runtime.ClusterNumOp()
    size = TestSSAValue(IndexType())
    stride = TestSSAValue(IndexType())
    repeat = TestSSAValue(IndexType())
    src_64 = TestSSAValue(IntegerType(data=64, signedness=Signedness.UNSIGNED))
    dst_64 = TestSSAValue(IntegerType(data=64, signedness=Signedness.UNSIGNED))
    src_32 = TestSSAValue(IntegerType(data=32, signedness=Signedness.UNSIGNED))
    dst_32 = TestSSAValue(IntegerType(data=32, signedness=Signedness.UNSIGNED))
    cluster_hw_barrier = snitch_runtime.ClusterHwBarrierOp()
    dma_start_1d_wideptr = snitch_runtime.DmaStart1DWidePtr(
        src=src_64, dst=dst_64, size=size
    )
    dma_start_1d = snitch_runtime.DmaStart1DWidePtr(src=src_32, dst=dst_32, size=size)
    dma_wait_all = snitch_runtime.DmaWaitAll()
    dma_start_2d_wideptr = snitch_runtime.DmaStart2DWidePtr(
        src=src_64,
        dst=dst_64,
        size=size,
        src_stride=stride,
        dst_stride=stride,
        repeat=repeat,
    )
    dma_start_2d_wideptr = snitch_runtime.DmaStart2DWidePtr(
        src=src_32,
        dst=dst_32,
        size=size,
        src_stride=stride,
        dst_stride=stride,
        repeat=repeat,
    )
    dma_wait = snitch_runtime.DmaWait(dma_start_2d_wideptr.results[0])
    print(cluster_num)
    print(cluster_hw_barrier)
    print(dma_start_1d_wideptr)
    print(dma_wait_all)
    print(dma_start_2d_wideptr)
    print(dma_wait)
