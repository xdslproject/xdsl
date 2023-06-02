from abc import ABC
from xdsl.ir import Operation, SSAValue, MLContext, Attribute
from xdsl.dialects.builtin import i32, ModuleOp
from xdsl.dialects import snitch_runtime, func
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
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
        rewriter.replace_matched_op(*self.lower(op))

    def lower(
        self, op: snitch_runtime.SnitchRuntimeGetInfo
    ) -> tuple[Operation, list[SSAValue]]:
        return func.Call.get("snrt_" + op.name[5:], [], [i32]), [op.result]


class LowerBarrierOpToFunc(RewritePattern, ABC):
    """
    Rewrite pattern that matches on all SnitchRuntimeBarrier ops, since they have
    the same function signature.

    Note: Takes the name of the op and replaces "snrt." with "snrt_" to link with
    snrt.h
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.SnitchRuntimeBarrier, rewriter: PatternRewriter, /
    ):
        rewriter.replace_matched_op(*self.lower(op))

    def lower(
        self, op: snitch_runtime.SnitchRuntimeBarrier
    ) -> tuple[Operation, list[SSAValue]]:
        return func.Call.get("snrt_" + op.name[5:], [], []), []


class AddExternalFuncs(RewritePattern, ABC):
    """
    Looks for snrt ops and adds an external func call to it for LLVM to link in
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, module: ModuleOp, rewriter: PatternRewriter, /):
        funcs_to_emit: dict[str, tuple[list[Attribute], list[Attribute]]] = dict()
        for op in module.walk():
            if not isinstance(op, func.Call):
                continue
            if "snrt" not in op.callee.string_value():
                continue
            funcs_to_emit[op.callee.string_value()] = (
                [arg.typ for arg in op.arguments],
                [res.typ for res in op.results],
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
                [LowerGetInfoOpToFunc(), LowerBarrierOpToFunc()]
            ),
            apply_recursively=True,
        )
        walker2 = PatternRewriteWalker(AddExternalFuncs())
        walker1.rewrite_module(op)
        walker2.rewrite_module(op)
