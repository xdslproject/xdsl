from collections.abc import Iterable
from itertools import chain

from xdsl.context import Context
from xdsl.dialects import affine, scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.traits import IsTerminator, is_side_effect_free, is_speculatable
from xdsl.transforms.common_subexpression_elimination import cse


def hoist_all(
    rewriter: PatternRewriter,
    ops: Iterable[Operation],
    at: InsertPoint,
    value_mapper: dict[SSAValue, SSAValue] | None = None,
):
    if value_mapper is None:
        value_mapper = {}
    for o in ops:
        if o.has_trait(IsTerminator, value_if_unregistered=False):
            continue
        new_op = o.clone(value_mapper=value_mapper)
        value_mapper |= {
            old: new for old, new in zip(o.results, new_op.results, strict=True)
        }
        rewriter.insert_op(new_op, at)
        rewriter.replace_op(o, [], new_op.results)


class AffineIfHoistPattern(RewritePattern):
    """
    Hoist everything out of a pure affine.if.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: affine.IfOp, rewriter: PatternRewriter):
        # Easy bail out for now
        if not (is_speculatable(op) and is_side_effect_free(op)):
            return

        hoist_all(
            rewriter,
            chain(op.then_region.ops, op.else_region.ops),
            InsertPoint.before(op),
        )
        if not rewriter.has_done_action:
            return
        block = op.parent
        if block:
            cse(block, rewriter)


class SCFIfHoistPattern(RewritePattern):
    """
    Hoist everything out of a pure scf.if
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.IfOp, rewriter: PatternRewriter):
        # Easy bail out for now
        if not (is_speculatable(op) and is_side_effect_free(op)):
            return

        hoist_all(
            rewriter,
            chain(op.true_region.ops, op.false_region.ops),
            InsertPoint.before(op),
        )

        # Perf-friendly cleanup
        # None needed if nothing happened
        if not rewriter.has_done_action:
            return
        block = op.parent
        if block:
            # If we hoisted some ops, run CSE on that block to not keep pushing duplicates upward.
            cse(block, rewriter)


class ControlFlowHoistPass(ModulePass):
    """
    Hoist all hoistable ops from control flow ops.
    """

    name = "control-flow-hoist"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AffineIfHoistPattern(),
                    SCFIfHoistPattern(),
                ]
            ),
            walk_regions_first=True,
        ).rewrite_module(op)
