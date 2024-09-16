from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import memref
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.csl import csl_wrapper
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


@dataclass(frozen=True)
class HoistBuffers(RewritePattern):
    """
    Hoists buffers to `csl_wrapper.program_module`-level.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Alloc, rewriter: PatternRewriter, /):
        wrapper = op.parent_op()
        while wrapper and not isinstance(wrapper, csl_wrapper.ModuleOp):
            wrapper = wrapper.parent_op()

        # no action required if this op exists on module-level
        if not wrapper or wrapper == op.parent_op():
            return

        assert len(op.dynamic_sizes) == 0, "not implemented"
        assert len(op.symbol_operands) == 0, "not implemented"

        rewriter.insert_op(
            alloc := op.clone(), InsertPoint.at_start(wrapper.program_module.block)
        )
        rewriter.replace_matched_op([], new_results=[alloc.memref])


@dataclass(frozen=True)
class CslWrapperHoistBuffers(ModulePass):
    """
    Hoists buffers to the `csl_wrapper.program_module`-level.
    """

    name = "csl-wrapper-hoist-buffers"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(HoistBuffers())
        module_pass.rewrite_module(op)
