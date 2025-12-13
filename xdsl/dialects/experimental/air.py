"""
Port of the AMD Xilinx AIR dialect for programming the AIEs on the AMD Xilinx Versal FPGA architecture.
This is a higher-level dialect than the AIE dialect. It is used to program Versal cards over PCIe.
AIE is a hardened systolic array present in the Versal devices. The dialect describes netlists of AIE
components and it can be lowered to the processor's assembly using the vendor's compiler. A description
of the original dialect can be found here https://xilinx.github.io/mlir-air/AIRDialect.html
"""

from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    MemRefType,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.dialects.utils import AbstractYieldOperation
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
    AttrSizedOperandSegments,
    IRDLOperation,
    ParsePropInAttrDict,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    opt_prop_def,
    opt_region_def,
    opt_result_def,
    prop_def,
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
    result = result_def(MemRefType)

    def __init__(
        self,
        async_dependencies: Operation,
        element_type: Attribute,
        shape: ArrayAttr[IntAttr],
    ):
        memref_type = MemRefType(element_type, shape)
        super().__init__(
            operands=[async_dependencies], result_types=[AsyncTokenAttr(), memref_type]
        )


@irdl_op_definition
class ChannelOp(IRDLOperation):
    name = "air.channel"

    sym_name = prop_def(SymbolRefAttr)
    size = prop_def(ArrayAttr)

    def __init__(
        self, sym_name: SymbolRefAttr, size: ArrayAttr[IntegerAttr]
    ):  # TODO: add verify to check 64-bit integer array attribute
        super().__init__(properties={"sym_name": sym_name, "size": size})

    assembly_format = "$sym_name $size attr-dict"


@irdl_op_definition
class ChannelGetOp(IRDLOperation):
    name = "air.channel.get"

    chan_name = attr_def(SymbolRefAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr())
    indices = var_operand_def(AsyncTokenAttr())
    dst = operand_def(MemRefType)
    dst_offsets = var_operand_def(IndexType())
    dst_sizes = var_operand_def(IndexType())
    dst_strides = var_operand_def(IndexType())

    async_token = result_def(AsyncTokenAttr())

    irdl_options = (AttrSizedOperandSegments(),)

    def __init__(
        self,
        chan_name: SymbolRefAttr,
        async_dependencies: list[Operation | SSAValue],
        indices: list[Operation | SSAValue],
        dst: Operation | SSAValue,
        dst_offsets: list[Operation | SSAValue],
        dst_sizes: list[Operation | SSAValue],
        dst_strides: list[Operation | SSAValue],
    ):
        super().__init__(
            attributes={"chan_name": chan_name},
            operands=[
                async_dependencies,
                indices,
                dst,
                dst_offsets,
                dst_sizes,
                dst_strides,
            ],
        )

    assembly_format = "(`async` `[` $async_dependencies^ `]`)? $chan_name `[` $indices `]` `(` $dst `[` $dst_offsets `]``[` $dst_sizes `]``[` $dst_strides `]` `)` attr-dict `:` `(` type($dst) `)`"  # noqa: E501


@irdl_op_definition
class ChannelPutOp(IRDLOperation):
    name = "air.channel.put"

    chan_name = attr_def(SymbolRefAttr)

    async_dependencies = var_operand_def(AsyncTokenAttr())
    indices = var_operand_def(IndexType())
    src = operand_def(MemRefType)
    src_offsets = var_operand_def(IndexType())
    src_sizes = var_operand_def(IndexType())
    src_strides = var_operand_def(IndexType())

    async_token = result_def(AsyncTokenAttr())

    irdl_options = (AttrSizedOperandSegments(),)

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
            properties={"chan_name": chan_name},
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

    assembly_format = "(`async` `[` $async_dependencies^ `]`)? $chan_name `[` $indices `]` `(` $src `[` $src_offsets `]``[` $src_sizes `]``[` $src_strides `]` `)` attr-dict `:` `(` type($src) `)`"  # noqa: E501


@irdl_op_definition
class CustomOp(IRDLOperation):
    name = "air.custom"

    symbol = attr_def(SymbolRefAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr)
    custom_operands = var_operand_def(Attribute)

    async_token = result_def(AsyncTokenAttr)

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

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
    memref = operand_def(MemRefType)

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

    async_dependencies = var_operand_def(AsyncTokenAttr())
    dst = operand_def(MemRefType)
    dst_offsets = var_operand_def(IndexType())
    dst_sizes = var_operand_def(IndexType())
    dst_strides = var_operand_def(IndexType())
    src = operand_def(MemRefType)
    src_offsets = var_operand_def(IndexType())
    src_sizes = var_operand_def(IndexType())
    src_strides = var_operand_def(IndexType())

    async_token = result_def(AsyncTokenAttr())

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

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
        print(
            async_dependencies,
            dst,
            dst_offsets,
            dst_sizes,
            dst_strides,
            src,
            src_offsets,
            src_sizes,
            src_strides,
        )
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

    assembly_format = "(`async` $async_dependencies^)? `(` $dst `[` $dst_offsets `]``[` $dst_sizes `]``[` $dst_strides `]` `,` $src `[` $src_offsets `]``[` $src_sizes `]``[` $src_strides `]` `)`  attr-dict `:` `(` type($dst) `,` type($src) `)`"  # noqa: E501


@irdl_op_definition
class ExecuteOp(IRDLOperation):
    name = "air.execute"

    async_dependencies = var_operand_def(AsyncTokenAttr)
    async_token = result_def(AsyncTokenAttr)
    results_ = var_result_def(Attribute)
    body = region_def()

    traits = lazy_traits_def(
        lambda: (SingleBlockImplicitTerminator(ExecuteTerminatorOp),)
    )

    def __init__(
        self,
        async_dependencies: list[SSAValue] | None,
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
        async_dependencies = parser.parse_optional_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_operand
        )

        result_types: list[Attribute] | None = []
        if parser.parse_optional_characters("->"):
            result_types = parser.parse_optional_comma_separated_list(
                parser.Delimiter.PAREN, parser.parse_type
            )

        body = parser.parse_region()

        return ExecuteOp(async_dependencies, result_types, body)


@irdl_op_definition
class ExecuteTerminatorOp(IRDLOperation):
    name = "air.execute_terminator"

    # even though this is an operand they decided to name it "result" in the original specification
    results_op = var_operand_def()

    traits = traits_def(HasParent(ExecuteOp), IsTerminator())

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

    traits = lazy_traits_def(lambda: (HasParent(HerdOp), IsTerminator()))

    assembly_format = "attr-dict"


@irdl_op_definition
class HerdOp(IRDLOperation):
    name = "air.herd"

    sym_name = opt_prop_def(StringAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr())
    sizes = var_operand_def(IndexType())
    herd_operands = var_operand_def()
    async_token = opt_result_def(AsyncTokenAttr)
    region = opt_region_def()

    traits = traits_def(
        IsolatedFromAbove(), SingleBlockImplicitTerminator(HerdTerminatorOp)
    )

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    def __init__(
        self,
        sym_name: StringAttr | None,
        async_dependencies: list[SSAValue] | None,
        sizes: list[Operation | SSAValue],
        herd_operands: list[Operation | SSAValue] | None,
        region: Region | None,
    ):
        super().__init__(
            properties={"sym_name": sym_name},
            operands=[async_dependencies, sizes, herd_operands],
            result_types=[AsyncTokenAttr()],
            regions=[region],
        )

    def print(self, printer: Printer):
        printer.print_string(" tile")
        with printer.in_parens():
            if len(self.sizes) == 1:
                printer.print_string("%tx")
            if len(self.sizes) == 2:
                printer.print_string("%tx, %ty")
            if len(self.sizes) == 3:
                printer.print_string("%tx, %ty, %tz")
        printer.print_string(" in ")
        with printer.in_parens():
            if len(self.sizes) == 1:
                printer.print_string("%size_x = ")
                printer.print_ssa_value(self.sizes[0])
            if len(self.sizes) == 2:
                printer.print_string("%size_x = ")
                printer.print_ssa_value(self.sizes[0])
                printer.print_string(", ")
                printer.print_string("%size_y = ")
                printer.print_ssa_value(self.sizes[1])
            if len(self.sizes) == 3:
                printer.print_string("%size_x = ")
                printer.print_ssa_value(self.sizes[0])
                printer.print_string(", ")
                printer.print_string("%size_y = ")
                printer.print_ssa_value(self.sizes[1])
                printer.print_string(", ")
                printer.print_string("%size_z = ")
                printer.print_ssa_value(self.sizes[2])

        if self.herd_operands:
            printer.print_string(" args")
            with printer.in_parens():
                if len(self.herd_operands) == 1:
                    printer.print_string("%ext0 = ")
                    printer.print_ssa_value(self.herd_operands[0])
                if len(self.herd_operands) == 2:
                    printer.print_string("%ext0 = ")
                    printer.print_ssa_value(self.herd_operands[0])
                    printer.print_string(", ")
                    printer.print_string("%ext1 = ")
                    printer.print_ssa_value(self.herd_operands[1])
                if len(self.herd_operands) == 3:
                    printer.print_string("%ext0 = ")
                    printer.print_ssa_value(self.herd_operands[0])
                    printer.print_string(", ")
                    printer.print_string("%ext1 = ")
                    printer.print_ssa_value(self.herd_operands[1])
                    printer.print_string(", ")
                    printer.print_string("%ext2 = ")
                    printer.print_ssa_value(self.herd_operands[2])

                printer.print_string(" : ")
                printer.print_list(
                    self.herd_operands, lambda o: printer.print_attribute(o.type), ","
                )

        if self.region:
            printer.print_region(self.region)

    @classmethod
    def parse(cls, parser: Parser) -> HerdOp:
        sym_name = parser.parse_optional_symbol_name()
        async_dependencies = parser.parse_optional_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_operand
        )
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
                parser.parse_optional_characters(",")

        parser.parse_keyword("attributes")
        parser.parse_optional_attr_dict()
        region = parser.parse_optional_region(arguments_lst)

        return HerdOp(sym_name, async_dependencies, tile_size_lst, operands_lst, region)


@irdl_op_definition
class LaunchTerminatorOp(IRDLOperation):
    name = "air.launch_terminator"

    traits = lazy_traits_def(
        lambda: (
            HasParent(LaunchOp),
            IsTerminator(),
        )
    )


@irdl_op_definition
class LaunchOp(IRDLOperation):
    name = "air.launch"

    sym_name = opt_prop_def(StringAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr())
    sizes = var_operand_def(IndexType())
    launch_operands = var_operand_def()
    async_token = result_def(AsyncTokenAttr)
    body = opt_region_def()

    traits = traits_def(
        IsolatedFromAbove(), SingleBlockImplicitTerminator(LaunchTerminatorOp)
    )

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    def __init__(
        self,
        sym_name: StringAttr | None,
        async_dependencies: list[SSAValue] | None,
        sizes: list[Operation | SSAValue],
        launch_operands: list[Operation | SSAValue],
        body: Region | None,
    ):
        super().__init__(
            attributes={"sym_name": sym_name},
            operands=[async_dependencies, sizes, launch_operands],
            result_types=[AsyncTokenAttr()],
            regions=[body],
        )

    @classmethod
    def parse(cls, parser: Parser) -> LaunchOp:
        sym_name = parser.parse_optional_symbol_name()

        async_dependencies = parser.parse_optional_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_operand
        )

        block_args_lst: list[Parser.Argument] = []
        if parser.parse_optional_characters("("):
            while not parser.parse_optional_characters(")"):
                b_arg = parser.parse_argument(expect_type=False)
                b_arg = b_arg.resolve(IndexType())
                block_args_lst.append(b_arg)
                parser.parse_optional_characters(",")

        parser.parse_keyword("in")
        parser.parse_characters("(")

        arguments_lst: list[Parser.Argument] = []

        sizes_operands_lst: list[Operation | SSAValue] = []
        while not parser.parse_optional_characters(")"):
            argument = parser.parse_argument(expect_type=False)
            parser.parse_characters("=")
            operand = parser.parse_operand()

            argument = argument.resolve(operand.type)
            arguments_lst.append(argument)
            sizes_operands_lst.append(operand)

            parser.parse_optional_characters(",")

        launch_operands_lst: list[Operation | SSAValue] = []
        attr_dict: dict[str, Attribute] = dict()
        if parser.parse_optional_keyword("args"):
            parser.parse_characters("(")
            while True:
                argument = parser.parse_argument(expect_type=False)
                parser.parse_characters("=")
                operand = parser.parse_operand()

                argument = argument.resolve(operand.type)
                block_args_lst.append(argument)
                launch_operands_lst.append(operand)

                if not parser.parse_optional_characters(","):
                    break
            parser.parse_characters(")")

            parser.parse_characters(":")
            for _ in range(len(launch_operands_lst)):
                parser.parse_type()
                parser.parse_optional_characters(",")

            if parser.parse_optional_keyword("attributes"):
                attr_dict = parser.parse_optional_attr_dict()

        body = parser.parse_optional_region(block_args_lst)

        launch_op = LaunchOp(
            sym_name, async_dependencies, sizes_operands_lst, launch_operands_lst, body
        )
        launch_op.attributes |= attr_dict

        return launch_op


@irdl_op_definition
class HerdPipelineOp(IRDLOperation):
    name = "air.pipeline"

    body = opt_region_def()

    traits = traits_def(HasParent(HerdOp))

    def __init__(self, body: None | Region):
        super().__init__(regions=[body])

    @classmethod
    def parse(cls, parser: Parser) -> HerdPipelineOp:
        attr_dict: dict[str, Attribute] = dict()
        if parser.parse_optional_keyword("attributes"):
            attr_dict = parser.parse_optional_attr_dict()

        body = parser.parse_optional_region()

        herd_pipeline_op = HerdPipelineOp(body)
        herd_pipeline_op.attributes |= attr_dict

        return herd_pipeline_op


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
    result = var_result_def()

    body = opt_region_def()

    traits = traits_def(HasParent(HerdPipelineOp))

    def __init__(
        self,
        opers: list[Operation | SSAValue],
        result_types: Sequence[Attribute],
        body: None | Region,
    ):
        if not result_types:
            result_types = []
        super().__init__(operands=[opers], result_types=[result_types], regions=[body])

    @classmethod
    def parse(cls, parser: Parser) -> PipelineStageOp:
        kernel_ops_lst: list[Operation | SSAValue] = []
        result_types_lst: list[Attribute] = []

        arguments_lst: list[Parser.Argument] = []
        pipeline_operands_lst: list[Operation | SSAValue] = []
        if parser.parse_optional_keyword("args"):
            parser.parse_characters("(")
            while not parser.parse_optional_characters(")"):
                argument = parser.parse_argument(expect_type=False)
                parser.parse_characters("=")
                operand = parser.parse_operand()
                pipeline_operands_lst.append(operand)

                argument = argument.resolve(operand.type)
                arguments_lst.append(argument)

                parser.parse_optional_characters(",")

            if parser.parse_optional_characters(":"):
                for _ in range(len(arguments_lst)):
                    parser.parse_type()
                    parser.parse_optional_characters(",")

        body = parser.parse_optional_region(arguments=arguments_lst)
        if parser.parse_optional_characters(":"):
            result_types_lst.append(parser.parse_type())
            parser.parse_optional_characters(",")

        return PipelineStageOp(kernel_ops_lst, result_types_lst, body)


@irdl_op_definition
class PipelineTerminatorOp(AbstractYieldOperation[Attribute]):
    name = "air.pipeline.terminator"

    traits = traits_def(HasParent(HerdPipelineOp), IsTerminator())


@irdl_op_definition
class PipelineYieldOp(AbstractYieldOperation[Attribute]):
    name = "air.pipeline.yield"

    traits = traits_def(HasParent(PipelineStageOp), IsTerminator())


@irdl_op_definition
class SegmentTerminatorOp(IRDLOperation):
    name = "air.segment_terminator"

    traits = lazy_traits_def(lambda: (HasParent(SegmentOp), IsTerminator()))


@irdl_op_definition
class SegmentOp(IRDLOperation):
    name = "air.segment"

    sym_name = opt_prop_def(StringAttr)
    async_dependencies = var_operand_def(AsyncTokenAttr())
    sizes = var_operand_def(IndexType())
    segment_operands = var_operand_def()
    async_token = result_def(AsyncTokenAttr)

    body = opt_region_def()

    traits = traits_def(
        IsolatedFromAbove(), SingleBlockImplicitTerminator(SegmentTerminatorOp)
    )

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    def __init__(
        self,
        sym_name: None | StringAttr,
        async_dependencies: list[Operation | SSAValue],
        sizes: list[Operation | SSAValue],
        segment_operands: list[Operation | SSAValue],
        body: None | Region,
    ):
        super().__init__(
            attributes={"sym_name": sym_name},
            operands=[async_dependencies, sizes, segment_operands],
            result_types=[AsyncTokenAttr()],
            regions=[body],
        )

    @classmethod
    def parse(cls, parser: Parser) -> SegmentOp:
        sym_name = parser.parse_optional_symbol_name()
        async_dependencies: list[Operation | SSAValue] = []
        sizes: list[Operation | SSAValue] = []
        parser.parse_optional_keyword("async")
        # TODO: unclear from the tests how to parse async. Follow the C++ code for the original custom parser
        if parser.parse_optional_keyword("unroll"):
            pass  # TODO: unclear from the tests how to parse unroll. Follow the C++ code for the original custom parser
        arguments_lst: list[Parser.Argument] = []
        segment_operands_lst: list[Operation | SSAValue] = []
        if parser.parse_optional_keyword("args"):
            if parser.parse_optional_characters("("):
                while True:
                    argument = parser.parse_argument(expect_type=False)
                    parser.parse_characters("=")
                    operand = parser.parse_operand()

                    argument = argument.resolve(operand.type)
                    arguments_lst.append(argument)
                    segment_operands_lst.append(operand)

                    if not parser.parse_optional_characters(","):
                        break
                parser.parse_characters(")")

            parser.parse_characters(":")

            for _ in range(len(segment_operands_lst)):
                parser.parse_type()
                parser.parse_optional_characters(",")

        attr_dict: dict[str, Attribute] = dict()
        if parser.parse_optional_keyword("attributes"):
            attr_dict = parser.parse_optional_attr_dict()

        body = parser.parse_optional_region()

        segment_op = SegmentOp(
            sym_name, async_dependencies, sizes, segment_operands_lst, body
        )
        segment_op.attributes |= attr_dict
        return segment_op


@irdl_op_definition
class WaitAllOp(IRDLOperation):
    name = "air.wait_all"

    async_dependencies = var_operand_def(AsyncTokenAttr)
    async_token = result_def(AsyncTokenAttr)

    def __init__(self, async_dependencies: list[SSAValue] | None):
        super().__init__(operands=[async_dependencies], result_types=[AsyncTokenAttr()])

    def print(self, printer: Printer):
        printer.print_string(" async ")
        if self.async_dependencies:
            printer.print_list(self.async_dependencies, printer.print_ssa_value)

    @classmethod
    def parse(cls, parser: Parser) -> WaitAllOp:
        parser.parse_keyword("async")
        async_dependencies = parser.parse_optional_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_operand
        )

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
