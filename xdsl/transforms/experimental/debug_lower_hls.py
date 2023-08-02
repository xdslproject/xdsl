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
from xdsl.dialects.builtin import i32, IndexType, IntegerType, StringAttr
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
    FuncOp,
    LLVMFunctionType,
    GlobalOp,
    AddressOfOp,
)

from xdsl.passes import ModulePass

from xdsl.dialects.scf import ParallelOp, For, Yield

from typing import cast, Any
from xdsl.utils.hints import isa

i8 = IntegerType(8)
p_i8 = LLVMPointerType.typed(i8)
string_printf = "Test\n"
string_type = llvm.LLVMArrayType.from_size_and_type(len(string_printf), i8)
p_string_type = LLVMPointerType.typed(string_type)


@dataclass
class HLSStreamWriteToPrintf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: HLSStreamWrite, rewriter: PatternRewriter, /):
        global_str = AddressOfOp.get("str", p_string_type)

        elem = op.element
        elem_type = op.element.typ
        p_elem_type = LLVMPointerType.typed(elem_type)
        p_struct_elem_type = op.operands[0].typ

        call_printf = llvm.CallOp("printf", global_str)

        rewriter.replace_matched_op([global_str, call_printf])


@dataclass
class HLSStreamReadToPrintf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: HLSStreamRead, rewriter: PatternRewriter, /):
        elem_type = op.operands[0].op.elem_type
        global_str = AddressOfOp.get("str", p_string_type)

        call_printf = llvm.CallOp("printf", global_str)

        size_alloca = Constant.from_int_and_width(1, i32)
        alloca = AllocaOp.get(size_alloca, elem_type)
        load = LoadOp.get(alloca)

        rewriter.replace_matched_op(
            [global_str, call_printf, size_alloca, alloca, load]
        )


@dataclass
class TrivialIntrinsicsCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Call, rewriter: PatternRewriter, /):
        callee_name = op.callee.root_reference.data

        if callee_name in ["load_data", "shift_buffer", "write_data"]:
            parent_dataflow = op.parent_op()
            rewriter.erase_matched_op()
            hls_yield = parent_dataflow.body.blocks[0].last_op
            hls_yield.detach()
            hls_yield.erase()
            parent_dataflow.detach()
            parent_dataflow.erase()


@dataclass
class TrivialHLSStreamCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: HLSStream, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()


@dataclass
class TrivialDataflowCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PragmaDataflow, rewriter: PatternRewriter, /):
        hls_yield = op.body.blocks[0].last_op
        hls_yield.detach()
        hls_yield.erase()

        rewriter.inline_block_before_matched_op(op.body.blocks[0])
        rewriter.erase_matched_op()


@dataclass
class TrivialPipelineCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PragmaPipeline, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()


@dataclass
class TrivialHLSExtractStencilValueCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: HLSExtractStencilValue, rewriter: PatternRewriter, /
    ):
        rewriter.erase_matched_op()


@dataclass
class DebugLowerHLSPass(ModulePass):
    name = "debug-lower-hls"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        printf_type = LLVMFunctionType(
            inputs=[p_string_type], output=None, is_variadic=True
        )

        printf = FuncOp("printf", printf_type, linkage=llvm.LinkageAttr("external"))

        op.body.block.add_op(printf)

        global_str = llvm.GlobalOp.get(
            string_type,
            "str",
            "internal",
            0,
            True,
            value=StringAttr(string_printf),
            alignment=8,
            unnamed_addr=0,
        )
        print(global_str)
        op.body.block.add_op(global_str)

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

        walkers = gen_greedy_walkers(
            [
                HLSStreamWriteToPrintf(),
                HLSStreamReadToPrintf(),
                TrivialIntrinsicsCleanup(),
                TrivialHLSStreamCleanup(),
                TrivialDataflowCleanup(),
                TrivialPipelineCleanup(),
                TrivialHLSExtractStencilValueCleanup(),
            ]
        )

        for walker in walkers:
            walker.rewrite_module(op)
