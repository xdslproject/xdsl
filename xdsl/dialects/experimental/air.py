"""
Port of the AMD Xilinx AIR dialect for programming the AIEs on the AMD Xilinx Versal FPGA architecture.
This is a higher-level dialect than the AIE dialect. It is used to program Versal cards over PCIe.
AIE is a hardened systolic array present in the Versal devices. The dialect describes netlists of AIE
components and it can be lowered to the processor's assembly using the vendor's compiler. A description
of the original dialect can be found here https://xilinx.github.io/mlir-air/AIRDialect.html
"""

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    ArrayAttr,
    IndexType,
    MemRefType,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    var_operand_def,
    var_result_def,
)


@irdl_attr_definition
class AsyncTokenAttr(ParametrizedAttribute, TypeAttribute):
    name = "air.async.token"


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "air.alloc"

    async_dependencies = var_operand_def(AsyncTokenAttr)

    async_token = result_def(AsyncTokenAttr)
    result = result_def(MemRefType[Attribute])

    def __init__(
        self,
        async_dependencies: Operation,
        element_type: Attribute,
        shape: ArrayAttr[AnyIntegerAttr],
    ):
        memref_type = MemRefType.from_element_type_and_shape(element_type, shape)
        super().__init__(
            operands=[async_dependencies], result_types=[AsyncTokenAttr(), memref_type]
        )


@irdl_op_definition
class ChannelOp(IRDLOperation):
    name = "air.channel"

    sym_name = attr_def(StringAttr)
    size = attr_def(ArrayAttr)

    def __init__(
        self, sym_name: StringAttr, size: ArrayAttr[AnyIntegerAttr]
    ):  # TODO: add verify to check 64-bit integer array attribute
        super().__init__(attributes={"sym_name": sym_name, "size": size})


@irdl_op_definition
class ChannelGetOp(IRDLOperation):
    name = "air.channel.get"

    chan_name = attr_def(SymbolRefAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr)
    indices = var_operand_def(AsyncTokenAttr)
    dst = operand_def(MemRefType[Attribute])

    def __init__(
        self,
        chan_name: SymbolRefAttr,
        async_dependencies: list[Operation | SSAValue],
        indices: list[Operation | SSAValue],
        dst: Operation | SSAValue,
    ):
        super().__init__(
            attributes={"chan_name": chan_name},
            operands=[async_dependencies, indices, dst],
        )


@irdl_op_definition
class ChannelPutOp(IRDLOperation):
    name = "air.channel.put"

    chan_name = attr_def(SymbolRefAttr)

    async_dependencies = var_operand_def(AsyncTokenAttr)
    indices = var_operand_def(IndexType)
    src = operand_def(MemRefType[Attribute])
    src_offsets = var_operand_def(IndexType)
    src_sizes = var_operand_def(IndexType)
    src_strides = var_operand_def(IndexType)

    def __init__(
        self,
        chan_name: SymbolRefAttr,
        async_dependencies: list[Operation | SSAValue],
        indices: list[Operation | SSAValue],
        src: Operation | SSAValue,
        src_offsets: list[Operation | SSAValue],
        src_sizes: list[Operation | SSAValue],
        src_strides: list[Operation | SSAValue],
    ):
        super().__init__(
            attributes={"chan_name": chan_name},
            operands=[
                async_dependencies,
                indices,
                src,
                src_offsets,
                src_sizes,
                src_strides,
            ],
        )


@irdl_op_definition
class CustomOp(IRDLOperation):
    name = "air.custom"

    symbol = attr_def(SymbolRefAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr)
    custom_operands = var_operand_def(Attribute)

    async_token = result_def(AsyncTokenAttr)

    def __init__(
        self,
        symbol: SymbolRefAttr,
        async_dependencies: list[Operation | SSAValue],
        custom_operands: list[Operation | SSAValue],
    ):
        super().__init__(
            attributes={"symbol": symbol},
            operands=[async_dependencies, custom_operands],
            result_types=[AsyncTokenAttr()],
        )


@irdl_op_definition
class DeallocOp(IRDLOperation):
    name = "air.dealloc"

    async_dependencies = var_operand_def(AsyncTokenAttr)
    memref = operand_def(MemRefType[Attribute])

    async_token = result_def(AsyncTokenAttr)

    def __init__(
        self,
        async_dependencies: list[Operation | SSAValue],
        memref: Operation | SSAValue,
    ):
        super().__init__(
            operands=[async_dependencies, memref], result_types=[AsyncTokenAttr()]
        )


@irdl_op_definition
class DmaMemcpyNdOp(IRDLOperation):
    name = "air.dma_memcpy_nd"

    async_dependencies = var_operand_def(AsyncTokenAttr)
    dst = operand_def(MemRefType)
    dst_offsets = var_operand_def(IndexType)
    dst_sizes = var_operand_def(IndexType)
    dst_strides = var_operand_def(IndexType)
    src = operand_def(MemRefType[Attribute])
    src_offsets = var_operand_def(IndexType)
    src_sizes = var_operand_def(IndexType)
    src_strides = var_operand_def(IndexType)

    async_token = result_def(AsyncTokenAttr)

    def __init__(
        self,
        async_dependencies: list[Operation | SSAValue],
        dst: Operation | SSAValue,
        dst_offsets: list[Operation | SSAValue],
        dst_sizes: list[Operation | SSAValue],
        dst_strides: list[Operation | SSAValue],
        src: Operation | SSAValue,
        src_offsets: list[Operation | SSAValue],
        src_sizes: list[Operation | SSAValue],
        src_strides: list[Operation | SSAValue],
    ):
        super().__init__(
            operands=[
                async_dependencies,
                dst,
                dst_offsets,
                dst_sizes,
                dst_strides,
                src,
                src_offsets,
                src_sizes,
                src_strides,
            ],
            result_types=[AsyncTokenAttr()],
        )


@irdl_op_definition
class ExecuteOp(IRDLOperation):
    name = "air.execute"

    async_dependencies = var_operand_def(AsyncTokenAttr)
    async_token = result_def(AsyncTokenAttr)
    results = var_result_def(Attribute)

    def __init__(
        self, async_dependencies: list[Operation | SSAValue], result_types: Attribute
    ):
        super().__init__(
            operands=[async_dependencies], result_types=[AsyncTokenAttr(), result_types]
        )


@irdl_op_definition
class ExecuteTerminatorOp(IRDLOperation):
    name = "air.execute_terminator"

    results = var_result_def(Attribute)

    def __init__(self):
        super().__init__(result_types=[Attribute()])


@irdl_op_definition
class HerdOp(IRDLOperation):
    name = "air.herd"

    sym_name = attr_def(StringAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr)
    sizes = var_operand_def(IndexType)
    herd_operands = var_operand_def(Attribute)
    async_token = result_def(AsyncTokenAttr)

    def __init__(
        self,
        sym_name: StringAttr,
        async_dependencies: list[Operation | SSAValue],
        sizes: list[Operation | SSAValue],
        herd_operands: list[Operation | SSAValue],
    ):
        super().__init__(
            attributes={"sym_name": sym_name},
            operands=[async_dependencies, sizes, herd_operands],
            result_types=[AsyncTokenAttr()],
        )


@irdl_op_definition
class HerdTerminatorOp(IRDLOperation):
    name = "air.herd_terminator"


@irdl_op_definition
class LaunchOp(IRDLOperation):
    name = "air.launch"

    sym_name = attr_def(StringAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr)
    sizes = var_operand_def(IndexType())
    launch_operands = var_operand_def(Attribute)
    async_token = result_def(AsyncTokenAttr)

    def __init__(
        self,
        sym_name: StringAttr,
        async_dependencies: list[Operation | SSAValue],
        sizes: list[Operation | SSAValue],
        launch_operands: list[Operation | SSAValue],
    ):
        super().__init__(
            attributes={"sym_name": sym_name},
            operands=[async_dependencies, sizes, launch_operands],
            result_types=[AsyncTokenAttr()],
        )


@irdl_op_definition
class LaunchTerminatorOp(IRDLOperation):
    name = "air.launch_terminator"


@irdl_op_definition
class HerdPipelineOp(IRDLOperation):
    name = "air.pipeline"


@irdl_op_definition
class PipelineGetOp(IRDLOperation):
    name = "air.pipeline.get"

    src0 = operand_def(Attribute)
    src1 = operand_def(Attribute)
    results = var_result_def(Attribute)

    def __init__(
        self,
        src0: Operation | SSAValue,
        src1: Operation | SSAValue,
        result_types: list[Attribute],
    ):
        super().__init__(operands=[src0, src1], result_types=result_types)


@irdl_op_definition
class PipelinePutOp(IRDLOperation):
    name = "air.pipeline.put"

    dst0 = operand_def(Attribute)
    dst1 = operand_def(Attribute)
    opers = var_operand_def(Attribute)

    def __init__(
        self,
        dst0: Operation | SSAValue,
        dst1: Operation | SSAValue,
        opers: list[Operation | SSAValue],
    ):
        super().__init__(operands=[dst0, dst1, opers])


@irdl_op_definition
class PipelineStageOp(IRDLOperation):
    name = "air.pipeline.stage"

    opers = var_operand_def(Attribute)
    results = var_result_def(Attribute)

    def __init__(
        self, opers: list[Operation | SSAValue], result_types: list[Attribute]
    ):
        super().__init__(operands=[opers], result_types=result_types)


@irdl_op_definition
class PipelineTerminatorOp(IRDLOperation):
    name = "air.pipeline.terminator"

    opers = var_operand_def(Attribute)

    def __init__(self, opers: list[Operation | SSAValue]):
        super().__init__(operands=[opers])


@irdl_op_definition
class PipelineYieldOp(IRDLOperation):
    name = "air.pipeline.yield"

    opers = var_operand_def(Attribute)

    def __init__(self, opers: list[Operation | SSAValue]):
        super().__init__(operands=[opers])


@irdl_op_definition
class SegmentOp(IRDLOperation):
    name = "air.segment"

    sym_name = attr_def(StringAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr)
    sizes = var_operand_def(IndexType)
    segment_operands = var_operand_def(Attribute)
    async_token = result_def(AsyncTokenAttr)

    def __init__(
        self,
        sym_name: StringAttr,
        async_dependencies: list[Operation | SSAValue],
        sizes: list[Operation | SSAValue],
        segment_operands: list[Operation | SSAValue],
    ):
        super().__init__(
            attributes={"sym_name": sym_name},
            operands=[async_dependencies, sizes, segment_operands],
            result_types=[AsyncTokenAttr()],
        )


@irdl_op_definition
class SegmentTerminatorOp(IRDLOperation):
    name = "air.segment_terminator"


@irdl_op_definition
class WaitAllOp(IRDLOperation):
    name = "air.wait_all"

    async_dependencies = var_operand_def(AsyncTokenAttr)
    async_token = result_def(AsyncTokenAttr)

    def __init__(self, async_dependencies: list[Operation | SSAValue]):
        super().__init__(operands=async_dependencies, result_types=[AsyncTokenAttr()])


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
    [AsyncTokenAttr],
)
