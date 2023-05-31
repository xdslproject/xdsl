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


class LowerClusterNumOp(RewritePattern, ABC):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.ClusterNumOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_matched_op(*self.lower(op))

    def lower(
        self, op: snitch_runtime.ClusterNumOp
    ) -> tuple[Operation, list[SSAValue]]:
        return func.Call.get("snrt_cluster_num", [], [i32]), [op.result]


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
            GreedyRewritePatternApplier([LowerClusterNumOp()]), apply_recursively=True
        )
        walker2 = PatternRewriteWalker(AddExternalFuncs())
        walker1.rewrite_module(op)
        walker2.rewrite_module(op)
