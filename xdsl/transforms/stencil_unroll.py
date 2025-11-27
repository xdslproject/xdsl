from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import cast

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    DynAccessOp,
    IndexAttr,
    IndexOp,
    ReturnOp,
    TempType,
)
from xdsl.ir import Attribute
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


def offseted_block_clone(apply: ApplyOp, unroll_offset: Sequence[int]):
    region = apply.region
    return_op = region.block.last_op
    # ReturnOp is ApplyOp's terminator
    assert isinstance(return_op, ReturnOp)

    offseted = region.clone().detach_block(0)

    for op in offseted.ops:
        match op:
            case AccessOp():
                if op.offset_mapping is None:
                    offset_mapping = list(range(0, len(op.offset)))
                else:
                    offset_mapping = op.offset_mapping
                new_offset = [
                    o + unroll_offset[m]
                    for o, m in zip(op.offset, offset_mapping, strict=True)
                ]

                op.offset = IndexAttr.get(*new_offset)
            case DynAccessOp():
                op.lb += IndexAttr.get(*unroll_offset)
                op.ub += IndexAttr.get(*unroll_offset)
            case IndexOp():
                op.offset += IndexAttr.get(*unroll_offset)
            case _:
                continue

    return offseted


@dataclass
class StencilUnrollPattern(RewritePattern):
    unroll_factor: tuple[int, ...]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        return_op = op.region.block.last_op
        # ReturnOp is ApplyOp's terminator
        assert isinstance(return_op, ReturnOp)

        # Don't unroll already unrolled stencils.
        if return_op.unroll is not None:
            return

        # Don't work on degenerate apply with no result
        if not op.results:
            return

        # Enforced by verification
        res_types = op.result_types
        assert isa(res_types, Sequence[TempType[Attribute]])
        dim = res_types[0].get_num_dims()

        # If unroll factors list is shorter than the dim, fill with ones from the front
        unroll = self.unroll_factor
        if len(unroll) < dim:
            # If unroll factors list is shorter than the dim, fill with ones from the front
            unroll = (1,) * (dim - len(unroll)) + unroll
        elif len(unroll) > dim:
            # If unroll factors list is longer than the dim, pop from the front to keep
            # similar semantics
            unroll = unroll[-dim:]

        # Bail out if nothing to unroll
        if prod(unroll) == 1:
            return

        # Get all the offsetted computations
        offsetted_blocks = [
            offseted_block_clone(op, cast(Sequence[int], offset))
            for offset in product(*(range(u) for u in unroll))
        ]

        # Merge them in one region
        unrolled_block = offsetted_blocks[0]
        unrolled_return = unrolled_block.last_op
        assert isinstance(unrolled_return, ReturnOp)
        assert unrolled_return is not None
        for block in offsetted_blocks[1:]:
            for marg, arg in zip(unrolled_block.args, block.args):
                arg.replace_by(marg)
            for o in block.ops:
                if o is block.last_op:
                    unrolled_return.operands = [*unrolled_return.operands, *o.operands]
                    break
                o.detach()
                unrolled_block.insert_op_before(o, unrolled_return)
        unrolled_return.unroll = IndexAttr.get(*unroll)
        new_apply = ApplyOp.get(op.args, unrolled_block, res_types)
        rewriter.replace_op(op, new_apply)


@dataclass(frozen=True)
class StencilUnrollPass(ModulePass):
    name = "stencil-unroll"

    unroll_factor: tuple[int, ...]

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([StencilUnrollPattern(self.unroll_factor)])
        )
        walker.rewrite_module(op)
