from dataclasses import dataclass

from xdsl.dialects import builtin
from xdsl.dialects.arith import Maxf
from xdsl.dialects.builtin import f64
from xdsl.dialects.experimental.math import AbsFOp, CopySignOp
from xdsl.dialects.func import Call, FuncOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class ReplaceCopySignOpByXilinxMath(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op
        self.func_def_declaration = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CopySignOp, rewriter: PatternRewriter, /):
        if not self.func_def_declaration:
            func_def = FuncOp.external("llvm.copysign.f64", [f64, f64], [f64])
            self.module.body.block.add_op(func_def)
            self.func_def_declaration = True

        call = Call("llvm.copysign.f64", [op.lhs, op.rhs], [f64])

        rewriter.replace_matched_op([call])


@dataclass
class ReplaceMaxfByXilinxMath(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op
        self.func_def_declaration = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Maxf, rewriter: PatternRewriter, /):
        if not self.func_def_declaration:
            func_def = FuncOp.external("llvm.maxnum.f64", [f64, f64], [f64])
            self.module.body.block.add_op(func_def)
            self.func_def_declaration = True

        call = Call("llvm.maxnum.f64", [op.lhs, op.rhs], [f64])

        rewriter.replace_matched_op([call])


@dataclass
class ReplaceAbsOpByXilinxMath(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op
        self.func_def_declaration = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AbsFOp, rewriter: PatternRewriter, /):
        if not self.func_def_declaration:
            func_def = FuncOp.external("llvm.fabs.f64", [f64], [f64])
            self.module.body.block.add_op(func_def)
            self.func_def_declaration = True

        call = Call("llvm.fabs.f64", [op.operand], [f64])

        rewriter.replace_matched_op([call])


@dataclass
class ReplaceIncompatibleFPGA(ModulePass):
    name = "replace-incompatible-fpga"

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

        walkers = gen_greedy_walkers(
            [
                # ReplaceCopySignOpByEquivalent(),
                ReplaceCopySignOpByXilinxMath(op),
                # ReplaceMaxfOpByEquivalent(),
                ReplaceMaxfByXilinxMath(op),
                # ReplaceAbsOpByEquivalent(),
                ReplaceAbsOpByXilinxMath(op),
            ]
        )

        for walker in walkers:
            walker.rewrite_module(op)
