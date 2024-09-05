from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import scf
from xdsl.dialects.builtin import FunctionType, ModuleOp
from xdsl.dialects.csl import csl, csl_stencil, csl_wrapper
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass(frozen=True)
class ConvertForLoopToCallGraphPass(RewritePattern):
    """ """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter, /):
        if not self._is_inside_wrapper_outside_apply(op):
            return
        csl.FuncOp("for_body", FunctionType.from_lists([], []))
        pass

    @staticmethod
    def _is_inside_wrapper_outside_apply(op: Operation):
        """Returns if the op is inside `csl_wrapper.module` and contains a `csl_stencil.apply`."""
        is_inside_wrapper = False
        is_inside_apply = False
        has_apply_inside = False

        parent_op = op.parent_op()
        while parent_op:
            if isinstance(parent_op, csl_wrapper.ModuleOp):
                is_inside_wrapper = True
            elif isinstance(parent_op, csl_stencil.ApplyOp):
                is_inside_apply = True
            parent_op = parent_op.parent_op()

        for child_op in op.walk():
            if isinstance(child_op, csl_stencil.ApplyOp):
                has_apply_inside = True
                break

        return is_inside_wrapper and not is_inside_apply and has_apply_inside


@dataclass(frozen=True)
class CslStencilLowerEnclosingLoop(ModulePass):
    """ """

    name = "csl-stencil-lower-enclosing-loops"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertForLoopToCallGraphPass(),
                ]
            )
        )
        module_pass.rewrite_module(op)
