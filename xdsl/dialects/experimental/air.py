"""
Port of the AMD Xilinx AIR dialect for programming the AIEs on the AMD Xilinx Versal FPGA architecture.
This is a higher-level dialect than the AIE dialect. It is used to program Versal cards over PCIe.
AIE is a hardened systolic array present in the Versal devices. The dialect describes netlists of AIE
components and it can be lowered to the processor's assembly using the vendor's compiler. A description
of the original dialect can be found here https://xilinx.github.io/mlir-air/AIRDialect.html
"""

from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "air.alloc"


@irdl_op_definition
class ChannelOp(IRDLOperation):
    name = "air.channel"


@irdl_op_definition
class ChannelGetOp(IRDLOperation):
    name = "air.channel.get"


@irdl_op_definition
class ChannelPutOp(IRDLOperation):
    name = "air.channel.put"


@irdl_op_definition
class CustomOp(IRDLOperation):
    name = "air.custom"


@irdl_op_definition
class DeallocOp(IRDLOperation):
    name = "air.dealloc"


@irdl_op_definition
class DmaMemcpyNdOp(IRDLOperation):
    name = "air.dma_memcpy_nd"


@irdl_op_definition
class ExecuteOp(IRDLOperation):
    name = "air.execute"


@irdl_op_definition
class ExecuteTerminatorOp(IRDLOperation):
    name = "air.execute_terminator"


@irdl_op_definition
class HerdOp(IRDLOperation):
    name = "air.herd"


@irdl_op_definition
class HerdTerminatorOp(IRDLOperation):
    name = "air.herd_terminator"


@irdl_op_definition
class LaunchOp(IRDLOperation):
    name = "air.launch"


@irdl_op_definition
class LaunchTerminatorOp(IRDLOperation):
    name = "air.launch_terminator"


@irdl_op_definition
class HerdPipelineOp(IRDLOperation):
    name = "air.pipeline"


@irdl_op_definition
class PipelineGetOp(IRDLOperation):
    name = "air.pipeline.get"


@irdl_op_definition
class PipelinePutOp(IRDLOperation):
    name = "air.pipeline.put"


@irdl_op_definition
class PipelineStageOp(IRDLOperation):
    name = "air.pipeline.stage"


@irdl_op_definition
class PipelineTerminatorOp(IRDLOperation):
    name = "air.pipeline.terminator"


@irdl_op_definition
class PipelineYieldOp(IRDLOperation):
    name = "air.pipeline.yield"


@irdl_op_definition
class SegmentOp(IRDLOperation):
    name = "air.segment"


@irdl_op_definition
class SegmentTerminatorOp(IRDLOperation):
    name = "air.segment_terminator"


@irdl_op_definition
class WaitAllOp(IRDLOperation):
    name = "air.wait_all"


AIR = Dialect(
    "air",
    [
        AllocOp,
        ChannelOp,
        ChannelGetOp,
        ChannelPutOp,
        CustomOp,
        DeallocOp,
        DmaMemcpyNdOp,
        ExecuteOp,
        ExecuteTerminatorOp,
        HerdOp,
        HerdTerminatorOp,
        LaunchOp,
        LaunchTerminatorOp,
        HerdPipelineOp,
        PipelineGetOp,
        PipelinePutOp,
        PipelineStageOp,
        PipelineTerminatorOp,
        PipelineYieldOp,
        SegmentOp,
        SegmentTerminatorOp,
        WaitAllOp,
    ],
    [],
)
