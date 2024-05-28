from typing import cast

from xdsl.dialects import stencil
from xdsl.ir import Attribute, Block, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RedundantOperands(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.ApplyOp, rewriter: PatternRewriter) -> None:
        unique_operands = list[SSAValue]()
        rbargs = list[int]()

        found_duplicate: bool = False

        for i, o in enumerate(op.args):
            try:
                ui = unique_operands.index(o)
                rbargs.append(ui)
                found_duplicate = True
            except ValueError:
                unique_operands.append(o)
                rbargs.append(i)

        if not found_duplicate:
            return

        new = stencil.ApplyOp.get(
            unique_operands,
            block := Block(arg_types=[uo.type for uo in unique_operands]),
            [cast(stencil.TempType[Attribute], r.type) for r in op.res],
        )
        rewriter.inline_block_at_start(
            op.region.block, block, [block.args[i] for i in rbargs]
        )
        rewriter.replace_matched_op(new)


class UnusedResults(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.ApplyOp, rewriter: PatternRewriter) -> None:
        unused = [i for i, r in enumerate(op.res) if len(r.uses) == 0]

        if not unused:
            return

        block = op.region.block
        op.region.detach_block(block)
        old_return = cast(stencil.ReturnOp, block.last_op)

        results = list(op.res)
        return_args = list(old_return.arg)

        for i in reversed(unused):
            results.pop(i)
            return_args.pop(i)

        new = stencil.ApplyOp.get(
            op.args, block, [cast(stencil.TempType[Attribute], r.type) for r in results]
        )

        replace_results: list[SSAValue | None] = list(new.res)
        for i in unused:
            replace_results.insert(i, None)

        rewriter.replace_op(old_return, stencil.ReturnOp.get(return_args))
        rewriter.replace_matched_op(new, replace_results)
