from dataclasses import dataclass

from xdsl.dialects import builtin, llvm
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import IntegerType, StringAttr, i32
from xdsl.dialects.experimental.hls import (
    HLSExtractStencilValue,
    HLSStream,
    HLSStreamRead,
    HLSStreamWrite,
    PragmaDataflow,
    PragmaPipeline,
)
from xdsl.dialects.func import Call
from xdsl.dialects.llvm import (
    AddressOfOp,
    AllocaOp,
    GEPOp,
    LLVMFunctionType,
    LLVMPointerType,
    LoadOp,
)
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

i8 = IntegerType(8)
p_i8 = LLVMPointerType.typed(i8)
string_read_duplicate = "READ DUPLICATE\n"
string_read_compute = "READ COMPUTE\n"
string_dont_care = "DONT CARE\n"
string_read_duplicate_type = llvm.LLVMArrayType.from_size_and_type(
    len(string_read_duplicate), i8
)
string_read_compute_type = llvm.LLVMArrayType.from_size_and_type(
    len(string_read_compute), i8
)
string_dont_care_type = llvm.LLVMArrayType.from_size_and_type(len(string_dont_care), i8)
p_string_read_duplicate_type = LLVMPointerType.typed(string_read_duplicate_type)
p_string_read_compute_type = LLVMPointerType.typed(string_read_compute_type)
p_string_dont_care_type = LLVMPointerType.typed(string_dont_care_type)


@dataclass
class HLSStreamWriteToPrintf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: HLSStreamWrite, rewriter: PatternRewriter, /):
        global_str = AddressOfOp.get("string_dont_care", p_string_dont_care_type)

        elem_type = op.element.type
        LLVMPointerType.typed(elem_type)
        op.operands[0].type

        p_global_str = GEPOp.get(global_str, [0, 0], result_type=p_i8)

        call_printf = llvm.CallOp("printf", p_global_str)

        rewriter.replace_matched_op([global_str, p_global_str, call_printf])


@dataclass
class HLSStreamReadToPrintf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: HLSStreamRead, rewriter: PatternRewriter, /):
        elem_type = op.operands[0].op.elem_type
        if "write_data" in op.attributes:
            global_str = AddressOfOp.get(
                "string_read_compute", p_string_read_compute_type
            )
        else:
            global_str = AddressOfOp.get(
                "string_read_duplicate", p_string_read_duplicate_type
            )

        call_printf = llvm.CallOp("printf", global_str)

        size_alloca = Constant.from_int_and_width(1, i32)
        alloca = AllocaOp.get(size_alloca, elem_type)
        load = LoadOp.get(alloca)

        p_global_str = GEPOp.get(global_str, [0, 0], result_type=p_i8)

        call_printf = llvm.CallOp("printf", p_global_str)

        rewriter.replace_matched_op(
            [global_str, p_global_str, call_printf, size_alloca, alloca, load]
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
        printf_type = LLVMFunctionType(inputs=[p_i8], output=None, is_variadic=True)

        printf = llvm.FuncOp(
            "printf", printf_type, linkage=llvm.LinkageAttr("external")
        )

        op.body.block.add_op(printf)

        global_read_duplicate = llvm.GlobalOp.get(
            string_read_duplicate_type,
            "string_read_duplicate",
            "internal",
            0,
            True,
            value=StringAttr(string_read_duplicate),
            alignment=8,
            unnamed_addr=0,
        )
        global_read_compute = llvm.GlobalOp.get(
            string_read_compute_type,
            "string_read_compute",
            "internal",
            0,
            True,
            value=StringAttr(string_read_compute),
            alignment=8,
            unnamed_addr=0,
        )
        global_dont_care = llvm.GlobalOp.get(
            string_dont_care_type,
            "string_dont_care",
            "internal",
            0,
            True,
            value=StringAttr(string_dont_care),
            alignment=8,
            unnamed_addr=0,
        )
        op.body.block.add_op(global_read_duplicate)
        op.body.block.add_op(global_read_compute)
        op.body.block.add_op(global_dont_care)

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
