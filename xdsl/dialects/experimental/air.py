"""
Port of the AMD Xilinx AIR dialect for programming the AIEs on the AMD Xilinx Versal FPGA architecture.
This is a higher-level dialect than the AIE dialect. It is used to program Versal cards over PCIe.
AIE is a hardened systolic array present in the Versal devices. The dialect describes netlists of AIE
components and it can be lowered to the processor's assembly using the vendor's compiler. A description
of the original dialect can be found here https://xilinx.github.io/mlir-air/AIRDialect.html
"""

from __future__ import annotations

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
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AttrSizedOperandSegments,
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_region_def,
    opt_result_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    SingleBlockImplicitTerminator,
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

    sym_name = attr_def(SymbolRefAttr)
    size = attr_def(ArrayAttr)

    def __init__(
        self, sym_name: StringAttr | SymbolRefAttr, size: ArrayAttr[AnyIntegerAttr]
    ):  # TODO: add verify to check 64-bit integer array attribute
        if isinstance(sym_name, StringAttr):
            sym_name = SymbolRefAttr(sym_name)
        super().__init__(attributes={"sym_name": sym_name, "size": size})

    assembly_format = "$sym_name $size attr-dict"


@irdl_op_definition
class ChannelGetOp(IRDLOperation):
    name = "air.channel.get"

    chan_name = attr_def(SymbolRefAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr)
    indices = var_operand_def(AsyncTokenAttr)
    dst = operand_def(MemRefType[Attribute])

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

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

    async_token = opt_result_def(AsyncTokenAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

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
            result_types=[AsyncTokenAttr()],
        )

    @classmethod
    def parse(cls, parser: Parser) -> ChannelPutOp:
        parser.parse_keyword("async")
        async_dependencies: list[Operation | SSAValue] = []
        if parser.parse_optional_characters("["):
            while not parser.parse_optional_characters("]"):
                parser.parse_operand()
                parser.parse_optional_characters(",")
        chan_name = SymbolRefAttr(parser.parse_symbol_name())
        indices: list[Operation | SSAValue] = []
        if parser.parse_optional_characters("["):
            while not parser.parse_optional_characters("]"):
                indices.append(parser.parse_operand())
                parser.parse_optional_characters(",")
        parser.parse_characters("(")
        src = parser.parse_operand()
        src_offsets: list[Operation | SSAValue] = []
        if parser.parse_optional_characters("["):
            while not parser.parse_optional_characters("]"):
                src_offsets.append(parser.parse_operand())
                parser.parse_optional_characters(",")
        src_sizes: list[Operation | SSAValue] = []
        if parser.parse_optional_characters("["):
            while not parser.parse_optional_characters("]"):
                src_sizes.append(parser.parse_operand())
                parser.parse_optional_characters(",")
        src_strides: list[Operation | SSAValue] = []
        if parser.parse_optional_characters("["):
            while not parser.parse_optional_characters("]"):
                src_strides.append(parser.parse_operand())
                parser.parse_optional_characters(",")

        parser.parse_characters(")")
        parser.parse_optional_attr_dict()
        parser.parse_characters(":")
        parser.parse_characters("(")
        parser.parse_type()
        parser.parse_characters(")")

        return ChannelPutOp(
            chan_name,
            async_dependencies,
            indices,
            src,
            src_offsets,
            src_sizes,
            src_strides,
        )


@irdl_op_definition
class CustomOp(IRDLOperation):
    name = "air.custom"

    symbol = attr_def(SymbolRefAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr)
    custom_operands = var_operand_def(Attribute)

    async_token = result_def(AsyncTokenAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

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

    async_token = opt_result_def(AsyncTokenAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        async_dependencies: list[Operation | SSAValue] | None,
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

    def print(self, printer: Printer):
        if self.async_dependencies:
            printer.print(self.async_dependencies)
        printer.print("(")
        printer.print(self.dst)
        printer.print("[")
        printer.print(self.dst_offsets)
        printer.print("]")
        printer.print("[")
        printer.print(self.dst_sizes)
        printer.print("]")
        printer.print(self.dst_strides)
        printer.print(", ")
        printer.print(self.src)
        printer.print("[")
        printer.print(self.src_offsets)
        printer.print("]")
        printer.print("[")
        printer.print(self.src_sizes)
        printer.print("]")
        printer.print("[")
        printer.print(self.src_strides)
        printer.print("]")
        printer.print(")")
        printer.print(" : ")
        printer.print("(")
        printer.print(self.dst.type)
        printer.print(", ")
        printer.print(self.src.type)
        printer.print(")")

    @classmethod
    def parse(cls, parser: Parser) -> DmaMemcpyNdOp:
        parser.parse_characters("(")
        dst = parser.parse_operand()
        parser.parse_characters("[")
        dst_offsets: list[Operation | SSAValue] = []
        while not parser.parse_optional_characters("]"):
            dst_offsets.append(parser.parse_operand())
            parser.parse_optional_characters(",")
        parser.parse_characters("[")
        dst_sizes: list[Operation | SSAValue] = []
        while not parser.parse_optional_characters("]"):
            dst_sizes.append(parser.parse_operand())
            parser.parse_optional_characters(",")
        parser.parse_characters("[")
        dst_strides: list[Operation | SSAValue] = []
        while not parser.parse_optional_characters("]"):
            dst_strides.append(parser.parse_operand())
            parser.parse_optional_characters(",")
        parser.parse_characters(",")
        src = parser.parse_operand()
        parser.parse_characters("[")
        src_offsets: list[Operation | SSAValue] = []
        while not parser.parse_optional_characters("]"):
            src_offsets.append(parser.parse_operand())
            parser.parse_optional_characters(",")
        parser.parse_characters("[")
        src_sizes: list[Operation | SSAValue] = []
        while not parser.parse_optional_characters("]"):
            src_sizes.append(parser.parse_operand())
            parser.parse_optional_characters(",")
        parser.parse_characters("[")
        src_strides: list[Operation | SSAValue] = []
        while not parser.parse_optional_characters("]"):
            src_strides.append(parser.parse_operand())
            parser.parse_optional_characters(",")
        parser.parse_characters(")")
        parser.parse_optional_attr_dict()
        parser.parse_characters(":")
        parser.parse_characters("(")
        parser.parse_type()
        parser.parse_characters(",")
        parser.parse_type()
        parser.parse_characters(")")

        return DmaMemcpyNdOp(
            None,
            dst,
            dst_offsets,
            dst_sizes,
            dst_strides,
            src,
            src_offsets,
            src_sizes,
            src_strides,
        )


@irdl_op_definition
class ExecuteOp(IRDLOperation):
    name = "air.execute"

    async_dependencies = var_operand_def(AsyncTokenAttr)
    async_token = result_def(AsyncTokenAttr)
    results_ = var_result_def(Attribute)
    body = region_def()

    traits = traits_def(
        lambda: frozenset([SingleBlockImplicitTerminator(ExecuteTerminatorOp)])
    )  # TODO: solve dependency

    def __init__(
        self,
        async_dependencies: list[Operation | SSAValue],
        result_types: list[Attribute] | None,
        body: Region,
    ):
        super().__init__(
            operands=[async_dependencies],
            result_types=[AsyncTokenAttr(), result_types],
            regions=[body],
        )

    @classmethod
    def parse(cls, parser: Parser) -> ExecuteOp:
        async_dependencies: list[Operation | SSAValue] = []
        if parser.parse_optional_characters("["):
            while not parser.parse_optional_characters("]"):
                async_dependencies.append(parser.parse_operand())
                parser.parse_optional_characters(",")

        result_types: list[Attribute] = []
        if parser.parse_optional_characters("->"):
            parser.parse_characters("(")
            while not parser.parse_optional_characters(")"):
                result_types.append(parser.parse_type())
                parser.parse_optional_characters(",")

        body = parser.parse_region()

        return ExecuteOp(async_dependencies, result_types, body)


@irdl_op_definition
class ExecuteTerminatorOp(IRDLOperation):
    name = "air.execute_terminator"

    results_op = var_operand_def(
        AnyAttr()
    )  # even though this is an operand they decided to name it "result" in the original specification

    traits = frozenset([HasParent(ExecuteOp), IsTerminator()])

    def __init__(self, results_op: list[Operation | SSAValue]):
        super().__init__(operands=[results_op])

    @classmethod
    def parse(cls, parser: Parser) -> ExecuteTerminatorOp:
        results_op: list[Operation | SSAValue] = []
        while not parser.parse_optional_characters(":"):
            results_op.append(parser.parse_operand())
        while parser.parse_optional_type():
            parser.parse_optional_characters(",")

        return ExecuteTerminatorOp(results_op)


@irdl_op_definition
class HerdTerminatorOp(IRDLOperation):
    name = "air.herd_terminator"

    traits = traits_def(lambda: frozenset([HasParent(HerdOp), IsTerminator()]))

    assembly_format = "attr-dict"


@irdl_op_definition
class HerdOp(IRDLOperation):
    name = "air.herd"

    sym_name = opt_attr_def(StringAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr())
    sizes = var_operand_def(IndexType())
    herd_operands = var_operand_def(AnyAttr())
    async_token = opt_result_def(AsyncTokenAttr)
    region = opt_region_def()

    traits = frozenset(
        [IsolatedFromAbove(), SingleBlockImplicitTerminator(HerdTerminatorOp)]
    )

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        sym_name: StringAttr | None,
        async_dependencies: list[Operation | SSAValue] | None,
        sizes: list[Operation | SSAValue],
        herd_operands: list[Operation | SSAValue] | None,
        region: Region | None,
    ):
        super().__init__(
            attributes={"sym_name": sym_name},
            operands=[async_dependencies, sizes, herd_operands],
            result_types=[AsyncTokenAttr()],
            regions=[region],
        )

    def print(self, printer: Printer):
        printer.print(" tile")
        printer.print("(")
        if len(self.sizes) == 1:
            printer.print("%tx")
        if len(self.sizes) == 2:
            printer.print("%tx, %ty")
        if len(self.sizes) == 3:
            printer.print("%tx, %ty, %tz")
        printer.print(")")
        printer.print(" in ")
        printer.print("(")
        if len(self.sizes) == 1:
            printer.print("%\\size_x = ")
        if len(self.sizes) == 2:
            printer.print("%size_x = ")
            printer.print(self.sizes[0])
            printer.print(", ")
            printer.print("%size_y = ")
            printer.print(self.sizes[1])
        if len(self.sizes) == 3:
            printer.print("%\\size_x, %\\size_y, %\\size_z")
        printer.print(")")

        if self.herd_operands:
            printer.print(" args")
            printer.print("(")
            if len(self.sizes) == 1:
                printer.print("%\\ext0 = ")
            if len(self.sizes) == 2:
                printer.print("%ext0 = ")
                printer.print(self.herd_operands[0])
                printer.print(", ")
                printer.print("ext1 = ")
                printer.print(self.herd_operands[1])
            if len(self.sizes) == 3:
                printer.print("%ext0 = ")

        printer.print(" : ")
        for n_arg in range(len(self.herd_operands)):
            printer.print(self.herd_operands[n_arg].type)
            if n_arg < len(self.herd_operands) - 1:
                printer.print(",")

        printer.print(")")
        if self.region:
            printer.print_region(self.region)

    @classmethod
    def parse(cls, parser: Parser) -> HerdOp:
        sym_name = parser.parse_optional_symbol_name()
        async_dependencies: list[Operation | SSAValue] = []
        if parser.parse_optional_keyword("async"):
            parser.parse_optional_characters("[")
            while not parser.parse_optional_characters("]"):
                async_dependencies.append(parser.parse_operand())
                parser.parse_optional_characters(",")
        parser.parse_keyword("tile")
        arg_list = parser.parse_op_args_list()
        parser.parse_keyword("in")

        parser.parse_characters("(")
        tile_size_lst: list[Operation | SSAValue] = []
        for n_arg in range(len(arg_list)):
            parser.parse_optional_argument(False)
            parser.parse_characters("=")
            tile_size = parser.parse_operand()
            tile_size_lst.append(tile_size)
            if n_arg < len(arg_list) - 1:
                parser.parse_characters(",")

        parser.parse_characters(")")
        arguments_lst: list[Parser.Argument] = []

        operands_lst: list[Operation | SSAValue] = []
        if parser.parse_optional_keyword("args"):
            parser.parse_characters("(")
            while True:
                argument = parser.parse_argument(expect_type=False)
                parser.parse_characters("=")
                operand = parser.parse_operand()

                # The type of the block argument is not known until the operand has been parsed, e.g. %ext0 = %arg0
                argument = argument.resolve(operand.type)
                arguments_lst.append(argument)
                operands_lst.append(operand)

                if not parser.parse_optional_characters(","):
                    break
            parser.parse_characters(")")

            parser.parse_characters(":")

            for n_arg in range(len(operands_lst)):
                parser.parse_type()
                if n_arg < len(arg_list) - 1:
                    parser.parse_characters(",")

        parser.parse_keyword("attributes")
        parser.parse_optional_attr_dict()
        region = parser.parse_optional_region(arguments_lst)

        return HerdOp(sym_name, async_dependencies, tile_size_lst, operands_lst, region)


@irdl_op_definition
class LaunchTerminatorOp(IRDLOperation):
    name = "air.launch_terminator"

    traits = traits_def(lambda: frozenset([HasParent(LaunchOp)]))


@irdl_op_definition
class LaunchOp(IRDLOperation):
    name = "air.launch"

    sym_name = attr_def(StringAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr())
    sizes = var_operand_def(IndexType())
    launch_operands = var_operand_def(AnyAttr())
    async_token = result_def(AsyncTokenAttr)

    traits = frozenset(
        [IsolatedFromAbove(), SingleBlockImplicitTerminator(LaunchTerminatorOp)]
    )

    irdl_options = [AttrSizedOperandSegments()]

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
class HerdPipelineOp(IRDLOperation):
    name = "air.pipeline"

    traits = frozenset([HasParent(HerdOp)])


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

    traits = frozenset([HasParent(HerdPipelineOp)])

    def __init__(
        self, opers: list[Operation | SSAValue], result_types: list[Attribute]
    ):
        super().__init__(operands=[opers], result_types=result_types)


@irdl_op_definition
class PipelineTerminatorOp(IRDLOperation):
    name = "air.pipeline.terminator"

    opers = var_operand_def(Attribute)

    traits = frozenset([HasParent(HerdPipelineOp), IsTerminator()])

    def __init__(self, opers: list[Operation | SSAValue]):
        super().__init__(operands=[opers])


@irdl_op_definition
class PipelineYieldOp(IRDLOperation):
    name = "air.pipeline.yield"

    opers = var_operand_def(Attribute)

    traits = frozenset([HasParent(PipelineStageOp), IsTerminator()])

    def __init__(self, opers: list[Operation | SSAValue]):
        super().__init__(operands=[opers])


@irdl_op_definition
class SegmentTerminatorOp(IRDLOperation):
    name = "air.segment_terminator"

    traits = traits_def(lambda: frozenset([HasParent(SegmentOp), IsTerminator()]))


@irdl_op_definition
class SegmentOp(IRDLOperation):
    name = "air.segment"

    sym_name = attr_def(StringAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr())
    sizes = var_operand_def(IndexType())
    segment_operands = var_operand_def(AnyAttr())
    async_token = result_def(AsyncTokenAttr)

    traits = frozenset(
        [IsolatedFromAbove(), SingleBlockImplicitTerminator(SegmentTerminatorOp)]
    )

    irdl_options = [AttrSizedOperandSegments()]

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
class WaitAllOp(IRDLOperation):
    name = "air.wait_all"

    async_dependencies = var_operand_def(AsyncTokenAttr)
    async_token = result_def(AsyncTokenAttr)

    def __init__(self, async_dependencies: list[Operation | SSAValue | None]):
        super().__init__(operands=async_dependencies, result_types=[AsyncTokenAttr()])

    @classmethod
    def parse(cls, parser: Parser) -> WaitAllOp:
        parser.parse_keyword("async")
        async_dependencies: list[Operation | SSAValue | None] = []
        if parser.parse_optional_characters("["):
            while parser.parse_optional_characters("]"):
                async_dependencies.append(parser.parse_operand())
                parser.parse_optional_characters(",")
        else:
            async_dependencies = [None]

        return WaitAllOp(async_dependencies)


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
