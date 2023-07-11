from dataclasses import dataclass


from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
)
from xdsl.ir import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.builtin import f64
from xdsl.dialects.arith import Constant, Cmpf, Cmpi, Negf, Maxf
from xdsl.builder import Builder

from xdsl.passes import ModulePass

from xdsl.dialects.scf import Yield, If
from xdsl.dialects.func import Call, FuncOp

from xdsl.dialects.experimental.math import CopySignOp, AbsFOp


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

        call = Call.get("llvm.copysign.f64", [op.lhs, op.rhs], [f64])

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

        call = Call.get("llvm.maxnum.f64", [op.lhs, op.rhs], [f64])

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

        call = Call.get("llvm.fabs.f64", [op.operand], [f64])

        rewriter.replace_matched_op([call])


@dataclass
class ReplaceCopySignOpByEquivalent(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CopySignOp, rewriter: PatternRewriter, /):
        """
        //%751 = "math.copysign"(%745, %750) : (f64, f64) -> f64
          %zero_751 = arith.constant 0.0 : f64
          %compf_751 = "arith.cmpf"(%745, %zero_751) {predicate = 1} : (f64, f64) -> i1
          %compf_751_1 = "arith.cmpf"(%750, %zero_751) {predicate = 1} : (f64, f64) -> i1
          %compi_751 = "arith.cmpi"(%compf_751, %compf_751_1) {predicate = 0} : (i1, i1) -> i1
          %751 = scf.if %compi_751 -> f64 {
              scf.yield %751 : f64
          }
          else {
              %neg_751 = "arith.negf"(%751) : (f64) -> f64
              scf.yield %neg_751 : f64
          }
        """
        zero = Constant.from_float_and_width(0.0, f64)
        compf_magn = Cmpf.get(op.lhs, zero, 4)
        compf_sign = Cmpf.get(op.rhs, zero, 4)
        compi_both = Cmpi.get(compf_magn, compf_sign, 0)

        @Builder.region
        def positive(builder: Builder):
            yieldop = Yield.get(op.lhs)

            builder.insert(yieldop)

        @Builder.region
        def negative(builder: Builder):
            negfop = Negf.get(op.lhs)
            yieldop = Yield.get(negfop)

            builder.insert(negfop)
            builder.insert(yieldop)

        if_neg = If.get(compi_both, [f64], positive, negative)

        rewriter.replace_matched_op(
            [zero, compf_magn, compf_sign, compi_both, if_neg], if_neg.output
        )


@dataclass
class ReplaceMaxfOpByEquivalent(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Maxf, rewriter: PatternRewriter, /):
        compf_ops = Cmpf.get(op.lhs, op.rhs, 2)

        @Builder.region
        def positive(builder: Builder):
            yieldop = Yield.get(op.lhs)

            builder.insert(yieldop)

        @Builder.region
        def negative(builder: Builder):
            yieldop = Yield.get(op.rhs)

            builder.insert(yieldop)

        if_max = If.get(compf_ops, [f64], positive, negative)

        rewriter.replace_matched_op([compf_ops, if_max])


@dataclass
class ReplaceAbsOpByEquivalent(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AbsFOp, rewriter: PatternRewriter, /):
        zero = Constant.from_float_and_width(0.0, f64)
        compf_gt = Cmpf.get(op.operand, zero, 4)

        @Builder.region
        def positive(builder: Builder):
            yieldop = Yield.get(op.operand)

            builder.insert(yieldop)

        @Builder.region
        def negative(builder: Builder):
            negfop = Negf.get(op.operand)
            yieldop = Yield.get(negfop)

            builder.insert(negfop)
            builder.insert(yieldop)

        if_abs = If.get(compf_gt, [f64], positive, negative)

        rewriter.replace_matched_op([zero, compf_gt, if_abs])


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
