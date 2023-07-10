from abc import ABC

from xdsl.dialects import func, snitch_runtime
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.dialects.snitch_runtime import tx_id
from xdsl.ir import Attribute, MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerGetInfoOpToFunc(RewritePattern, ABC):
    """
    Rewrite pattern that matches on all SnitchRuntimeGetInfo ops, since they have
    the same function signature.

    Note: Takes the name of the op and replaces "snrt." with "snrt_" to link with
    snrt.h
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.SnitchRuntimeGetInfo, rewriter: PatternRewriter, /
    ):
        func_call = func.Call.get("snrt_" + op.name[5:], [], [i32])
        rewriter.replace_matched_op(func_call)


class LowerBarrierOpToFunc(RewritePattern, ABC):
    """
    Rewrite pattern that matches on all SnitchRuntimeBarrier ops, since they have
    the same function signature.

    Note: Takes the name of the op and replaces "snrt." with "snrt_" to link with
    snrt.h
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: snitch_runtime.NoOperandNoResultBaseOperation,
        rewriter: PatternRewriter,
        /,
    ):
        func_call = func.Call.get("snrt_" + op.name[5:], [], [])
        rewriter.replace_matched_op(func_call)


class LowerDma1DOpToFunc(RewritePattern, ABC):
    """
    Rewrite pattern that matches on DmaStart1DOp instances and lowers to external
    function calls.
    Works on both DmaStart1DOp and DmaStart1DWideptrOp.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: snitch_runtime.DmaStart1DOp | snitch_runtime.DmaStart1DWideptrOp,
        rewriter: PatternRewriter,
    ):
        func_call = func.Call.get(
            "snrt_" + op.name[5:], [op.dst, op.src, op.size], [tx_id]
        )
        rewriter.replace_matched_op(func_call)


class LowerDma2DOpToFunc(RewritePattern, ABC):
    """
    Rewrite pattern that matches on DmaStart2DOp instances and lowers to external
    function calls
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: snitch_runtime.DmaStart2DOp | snitch_runtime.DmaStart2DWideptrOp,
        rewriter: PatternRewriter,
    ):
        func_call = func.Call.get(
            "snrt_" + op.name[5:],
            [op.dst, op.src, op.dst_stride, op.src_stride, op.size, op.repeat],
            [tx_id],
        )
        rewriter.replace_matched_op(func_call)


class AddExternalFuncs(RewritePattern, ABC):
    """
    Looks for snrt ops and adds an external func call to it for LLVM to link in
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, module: ModuleOp, rewriter: PatternRewriter):
        funcs_to_emit: dict[str, tuple[list[Attribute], list[Attribute]]] = dict()
        for op in module.walk():
            if not isinstance(op, func.Call):
                continue
            if "snrt" not in op.callee.string_value():
                continue
            funcs_to_emit[op.callee.string_value()] = (
                [arg.type for arg in op.arguments],
                [res.type for res in op.results],
            )

        for name, types in funcs_to_emit.items():
            arg, res = types
            rewriter.insert_op_at_end(
                func.FuncOp.external(name, arg, res), module.body.block
            )


class LowerSnitchRuntimePass(ModulePass):
    name = "lower-snrt-to-func"

    # lower to func.call
    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker1 = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerGetInfoOpToFunc(),
                    LowerBarrierOpToFunc(),
                    LowerDma1DOpToFunc(),
                    LowerDma2DOpToFunc(),
                ]
            ),
            apply_recursively=True,
        )
        walker2 = PatternRewriteWalker(AddExternalFuncs())
        walker1.rewrite_module(op)
        walker2.rewrite_module(op)
