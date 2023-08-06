from dataclasses import dataclass

from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
)
from xdsl.ir import Block, MLContext, Region, Operation, OpResult
from xdsl.irdl import VarOperand, VarOpResult
from xdsl.dialects.func import FuncOp, Return, Call
from xdsl.dialects import builtin, func
from xdsl.dialects.builtin import i32, IndexType, f64
from xdsl.dialects.arith import Constant

from xdsl.dialects.experimental.hls import (
    PragmaPipeline,
    PragmaUnroll,
    PragmaDataflow,
    HLSStream,
    HLSStreamRead,
    HLSStreamWrite,
    HLSYield,
    HLSExtractStencilValue,
)
from xdsl.dialects import llvm
from xdsl.dialects.llvm import (
    AllocaOp,
    LLVMPointerType,
    GEPOp,
    LLVMStructType,
    LoadOp,
    StoreOp,
)

from xdsl.passes import ModulePass

from xdsl.dialects.scf import ParallelOp, For, Yield

from typing import cast, Any
from xdsl.utils.hints import isa
from xdsl.builder import Builder


@dataclass
class LowerHLSStreamWrite(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op
        self.push_stencil_declaration = False
        self.push_duplicate_declaration = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: HLSStreamWrite, rewriter: PatternRewriter, /):
        elem = op.element
        elem_type = op.element.typ
        p_elem_type = LLVMPointerType.typed(elem_type)
        p_struct_elem_type = op.operands[0].typ

        if "duplicate" in op.attributes:
            if not self.push_duplicate_declaration:
                push_func = func.FuncOp.external(
                    "llvm.fpga.fifo.push.duplicate", [elem_type, p_elem_type], []
                )

                self.module.body.block.add_op(push_func)
                self.push_duplicate_declaration = True

            gep = GEPOp.get(op.stream, [0, 0], result_type=p_elem_type)
            push_call = func.Call.get("llvm.fpga.fifo.push.duplicate", [elem, gep], [])

        else:
            if not self.push_stencil_declaration:
                push_func = func.FuncOp.external(
                    "llvm.fpga.fifo.push.stencil", [elem_type, p_elem_type], []
                )

                self.module.body.block.add_op(push_func)
                self.push_stencil_declaration = True

            gep = GEPOp.get(op.stream, [0, 0], result_type=p_elem_type)
            push_call = func.Call.get("llvm.fpga.fifo.push.stencil", [elem, gep], [])

        rewriter.replace_matched_op([gep, push_call])


@dataclass
class LowerHLSStreamRead(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op
        self.pop_stencil_declaration = False
        self.pop_write_data_declaration = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: HLSStreamRead, rewriter: PatternRewriter, /):
        # The stream is an alloca of a struct of the elem_type. elem_type must be extracted from the struct
        p_struct_elem_type = op.operands[0].typ

        elem_type = p_struct_elem_type.type.types
        p_elem_type = LLVMPointerType.typed(*elem_type)

        if "write_data" in op.attributes:
            if not self.pop_write_data_declaration:
                pop_func = func.FuncOp.external(
                    "llvm.fpga.fifo.pop.write_data",
                    [p_elem_type],
                    [op.res.typ],
                )

                self.module.body.block.add_op(pop_func)

                self.pop_write_data_declaration = True
            size = Constant.from_int_and_width(1, i32)

            alloca = AllocaOp.get(size, *elem_type)

            gep = GEPOp.get(op.stream, [0, 0], result_type=p_elem_type)

            pop_call = func.Call.get(
                "llvm.fpga.fifo.pop.write_data", [gep], [op.res.typ]
            )

        else:
            if not self.pop_stencil_declaration:
                pop_func = func.FuncOp.external(
                    "llvm.fpga.fifo.pop.stencil",
                    [p_elem_type],
                    [op.res.typ],
                )

                self.module.body.block.add_op(pop_func)

                self.pop_stencil_declaration = True
            size = Constant.from_int_and_width(1, i32)

            alloca = AllocaOp.get(size, *elem_type)

            gep = GEPOp.get(op.stream, [0, 0], result_type=p_elem_type)

            pop_call = func.Call.get("llvm.fpga.fifo.pop.stencil", [gep], [op.res.typ])

        current_parent = op.parent_op()
        while not isinstance(current_parent, FuncOp):
            current_parent = current_parent.parent_op()

        store = StoreOp.get(pop_call, alloca)
        load = LoadOp.get(alloca)

        # rewriter.insert_op_at_start(alloca, current_parent.body.blocks[0])
        current_parent.body.blocks[0].insert_op_before(
            alloca, current_parent.body.blocks[0].first_op
        )
        current_parent.body.blocks[0].insert_op_before(
            size, current_parent.body.blocks[0].first_op
        )

        # rewriter.replace_matched_op([size, alloca, gep, pop_call, store, load])
        rewriter.replace_matched_op([gep, pop_call, store, load])


@dataclass
class LowerHLSStreamToAlloca(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op
        self.set_stream_depth_declaration = False
        self.set_stream_size_qualifier_double_declaration = False
        self.set_stream_size_qualifier_stencil_declaration = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: HLSStream, rewriter: PatternRewriter, /):
        # We need to make sure that the type gets updated in the operations using the stream
        uses = []

        for use in op.result.uses:
            uses.append(use)

        hls_elem_type = op.elem_type
        stream_type = LLVMPointerType.typed(
            LLVMStructType.from_type_list([hls_elem_type])
        )

        if not self.set_stream_depth_declaration:
            stream_depth_func = llvm.FuncOp(
                "llvm.fpga.set.stream.depth",
                llvm.LLVMFunctionType([], is_variadic=True),
                linkage=llvm.LinkageAttr("external"),
            )
            self.module.body.block.add_op(stream_depth_func)

            self.set_stream_depth_declaration = True

        # As can be seen on the compiled synthetic stream benchmark of the FPL paper
        size = Constant.from_int_and_width(1, i32)
        alloca = AllocaOp.get(size, LLVMStructType.from_type_list([hls_elem_type]))
        gep = GEPOp.get(
            alloca, [0, 0], result_type=LLVMPointerType.typed(hls_elem_type)
        )
        depth = Constant.from_int_and_width(0, i32)
        depth_call = llvm.CallOp("llvm.fpga.set.stream.depth", gep, depth)

        stream_size = Constant.from_int_and_width(32, i32)

        if hls_elem_type == f64:
            stream_size_call = Call.get(
                "stream_size_qualifier_double", [op, stream_size], []
            )
            if not self.set_stream_size_qualifier_double_declaration:
                stream_size_qualifier_double = FuncOp.external(
                    "stream_size_qualifier_double", [stream_type, i32], []
                )
                self.module.body.block.add_op(stream_size_qualifier_double)
                self.set_stream_size_qualifier_double_declaration = True
        else:
            stream_size_call = Call.get(
                "stream_size_qualifier_stencil", [op, stream_size], []
            )
            if not self.set_stream_size_qualifier_stencil_declaration:
                stream_size_qualifier_double = FuncOp.external(
                    "stream_size_qualifier_stencil", [stream_type, i32], []
                )
                self.module.body.block.add_op(stream_size_qualifier_double)
                self.set_stream_size_qualifier_stencil_declaration = True

        start_df_call = Call.get("_start_df_call", [], [i32])
        end_df_call = Call.get("_end_df_call", [], [])

        rewriter.insert_op_after_matched_op(
            [
                start_df_call,
                depth,
                gep,
                depth_call,
                stream_size,
                stream_size_call,
                end_df_call,
            ]
        )
        rewriter.replace_matched_op([size, alloca])

        for use in uses:
            use.operation.operands[use.index].typ = alloca.res.typ

            # This is specially important when the stream is an argument of ApplyOp
            if use.operation.regions:
                block_arg = use.operation.regions[0].block.args[use.index]
                block_arg.typ = alloca.res.typ


@dataclass
class PragmaPipelineToFunc(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op
        self.declared_pipeline_names = set()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PragmaPipeline, rewriter: PatternRewriter, /):
        # TODO: can we retrieve data directly without having to go through IntegerAttr -> IntAttr?
        # ii : i32 = op.ii.owner.value.value.data
        ii = cast(Any, op.ii.owner).value.value.data

        ret1 = Return()
        block1 = Block(arg_types=[])
        block1.add_ops([ret1])
        region1 = Region(block1)

        pipeline_func_name = f"_pipeline_{ii}_"
        func1 = FuncOp.external(pipeline_func_name, [], [])

        call1 = Call.get(func1.sym_name.data, [], [])

        if pipeline_func_name not in self.declared_pipeline_names:
            self.module.body.block.add_op(func1)
            self.declared_pipeline_names.add(pipeline_func_name)

        rewriter.replace_matched_op(call1)


@dataclass
class PragmaUnrollToFunc(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PragmaUnroll, rewriter: PatternRewriter, /):
        # TODO: can we retrieve data directly without having to go through IntegerAttr -> IntAttr?
        factor = cast(Any, op.factor.owner).value.value.data

        ret1 = Return()
        block1 = Block(arg_types=[])
        block1.add_ops([ret1])
        region1 = Region(block1)
        func1 = FuncOp.from_region(f"_unroll_{factor}_", [], [], region1)

        call1 = Call.get(func1.sym_name.data, [], [])

        self.module.body.block.add_op(func1)

        rewriter.replace_matched_op(call1)


# @dataclass
# class PragmaDataflowToFunc(RewritePattern):
#    def __init__(self, op: builtin.ModuleOp):
#        self.module = op
#
#    @op_type_rewrite_pattern
#    def match_and_rewrite(self, op: PragmaDataflow, rewriter: PatternRewriter, /):
#        # TODO: can we retrieve data directly without having to go through IntegerAttr -> IntAttr?
#        ret1 = Return()
#        block1 = Block(arg_types=[])
#        block1.add_ops([ret1])
#        region1 = Region(block1)
#        func1 = FuncOp.from_region(f"_dataflow", [], [], region1)
#
#        call1 = Call.get(func1.sym_name.data, [], [])
#
#        self.module.body.block.add_op(func1)
#
#        rewriter.replace_matched_op(call1)


@dataclass
class SCFParallelToHLSPipelinedFor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ParallelOp, rewriter: PatternRewriter, /):
        ii = Constant.from_int_and_width(1, i32)
        hls_pipeline_op: Operation = PragmaPipeline(ii)

        lb: VarOperand = op.lowerBound
        ub: VarOperand = op.upperBound
        step: VarOperand = op.step
        res: VarOpResult = op.res

        for i in range(len(lb)):
            cast(OpResult, lb[i]).op.detach()
            cast(OpResult, ub[i]).op.detach()
            cast(OpResult, step[i]).op.detach()

        # We generate a For loop for each induction variable in the Parallel loop.
        # We start by wrapping the parallel block in a region for the For loop and keep
        # wrapping in for loops until we have exhausted the induction variables
        parallel_block = op.body.detach_block(0)

        if res != []:
            parallel_block.insert_arg(res[0].typ, 1)
            cast(Operation, parallel_block.last_op).detach()
            yieldop = Yield.get(res[0].op)
            parallel_block.add_op(yieldop)

        for_region = Region([parallel_block])

        for i in range(len(lb) - 1):
            for_region.block.erase_arg(for_region.block.args[i])

        if res != []:
            for_op = For.get(lb[-1], ub[-1], step[-1], [res[0].op], for_region)
        else:
            for_op = For.get(lb[-1], ub[-1], step[-1], [], for_region)

        for i in range(len(lb) - 2, -1, -1):
            for_region = Region(Block([for_op]))

            for_region.block.insert_arg(IndexType(), 0)

            for_region.block.insert_op_before(
                cast(OpResult, lb[i + 1]).op, cast(Operation, for_region.block.first_op)
            )
            for_region.block.insert_op_after(
                cast(OpResult, ub[i + 1]).op, cast(OpResult, lb[i + 1]).op
            )
            for_region.block.insert_op_after(
                cast(OpResult, step[i + 1]).op, cast(OpResult, ub[i + 1]).op
            )
            yieldop = Yield.get()
            for_region.block.add_op(yieldop)
            for_op = For.get(lb[i], ub[i], step[i], [], for_region)

        for_region.block.insert_op_before(
            hls_pipeline_op, cast(Operation, for_region.block.first_op)
        )
        for_region.block.insert_op_after(ii, cast(Operation, for_region.block.first_op))

        cast(Block, op.parent_block()).insert_op_before(
            cast(OpResult, lb[0]).op,
            cast(Operation, cast(Block, op.parent_block()).first_op),
        )
        cast(Block, op.parent_block()).insert_op_after(
            cast(OpResult, ub[0]).op, cast(OpResult, lb[0]).op
        )
        cast(Block, op.parent_block()).insert_op_after(
            cast(OpResult, step[0]).op, cast(OpResult, ub[0]).op
        )

        rewriter.replace_matched_op([for_op])


@dataclass
class LowerDataflow(RewritePattern):
    module: builtin.ModuleOp
    declared_df_functions: bool = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PragmaDataflow, rewriter: PatternRewriter, /):
        if not self.declared_df_functions:
            start_df_func = FuncOp.external("_start_df_call", [], [i32])
            end_df_func = FuncOp.external("_end_df_call", [], [])

            self.module.body.block.add_op(start_df_func)
            self.module.body.block.add_op(end_df_func)

            self.declared_df_functions = True

        start_df_call = Call.get("_start_df_call", [], [i32])
        end_df_call = Call.get("_end_df_call", [], [])

        rewriter.insert_op_before_matched_op(start_df_call)
        rewriter.insert_op_after_matched_op(end_df_call)

        dataflow_ops = [op for op in op.body.block.ops if not isinstance(op, HLSYield)]
        for df_op in reversed(dataflow_ops):
            df_op.detach()
            rewriter.insert_op_after_matched_op(df_op)

        rewriter.erase_matched_op()


@dataclass
class LowerHLSExtractStencilValue(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: HLSExtractStencilValue, rewriter: PatternRewriter, /
    ):
        indices = [attr.data for attr in op.position.data]

        stencil = op.container.op.ptr
        # result_hls_read = op.container
        # p_stencil = op.container.
        struct_types = stencil.typ.type.types
        array_type = struct_types.data[0]
        values = GEPOp.get(
            stencil, [0, 0], result_type=LLVMPointerType.typed(array_type)
        )
        first_array = GEPOp.get(
            values, [0, indices[1]], result_type=LLVMPointerType.typed(array_type.type)
        )
        second_array = GEPOp.get(
            first_array,
            [0, indices[2]],
            result_type=LLVMPointerType.typed(array_type.type.type),
        )
        third_array = GEPOp.get(
            second_array,
            [0, indices[3]],
            result_type=LLVMPointerType.typed(array_type.type.type.type),
        )
        point = LoadOp.get(third_array)

        rewriter.replace_matched_op(
            [values, first_array, second_array, third_array, point]
        )


@dataclass
class GetHLSStreamInDataflow(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: HLSStream, rewriter: PatternRewriter, /):
        hls_yield = HLSYield.get()

        @Builder.region
        def empty_region(builder: Builder):
            builder.insert(hls_yield)

        dataflow = PragmaDataflow(empty_region)
        rewriter.insert_op_before_matched_op(dataflow)
        op.detach()
        dataflow.body.blocks[0].insert_op_before(op, hls_yield)


@dataclass
class LowerHLSPass(ModulePass):
    name = "lower-hls"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        def gen_greedy_walkers(
            passes: list[RewritePattern],
        ) -> list[PatternRewriteWalker]:
            # Creates a greedy walker for each pass, so that they can be run sequentially even after
            # matching
            walkers: list[PatternRewriteWalker] = []

            for i in range(len(passes)):
                walkers.append(
                    PatternRewriteWalker(
                        GreedyRewritePatternApplier([passes[i]]), apply_recursively=True
                    )
                )

            return walkers

        hlsstream_df = PatternRewriteWalker(
            GreedyRewritePatternApplier([GetHLSStreamInDataflow()]),
            apply_recursively=False,
            walk_reverse=False,
        )
        # hlsstream_df.rewrite_module(op)

        walkers = gen_greedy_walkers(
            [
                # SCFParallelToHLSPipelinedFor(),
                PragmaPipelineToFunc(op),
                PragmaUnrollToFunc(op),
                # PragmaDataflowToFunc(op),
                LowerDataflow(op),
                LowerHLSStreamToAlloca(op),
                LowerHLSStreamRead(op),
                LowerHLSStreamWrite(op),
                LowerHLSExtractStencilValue(),
            ]
        )

        for walker in walkers:
            walker.rewrite_module(op)
