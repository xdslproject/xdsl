from typing import cast

from xdsl.dialects import stencil
from xdsl.ir import Attribute, Block, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.transforms.common_subexpression_elimination import cse


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

        bbargs = op.region.block.args
        for i, a in enumerate(bbargs):
            if rbargs[i] == i:
                continue
            a.replace_by(bbargs[rbargs[i]])

        cse(op.region.block)


class UnusedOperands(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.ApplyOp, rewriter: PatternRewriter) -> None:
        op_args = op.region.block.args
        unused = {a for a in op_args if not a.uses}
        if not unused:
            return
        bbargs = [a for a in op_args if a not in unused]
        bbargs_type = [a.type for a in bbargs]
        operands = [a for i, a in enumerate(op.args) if op_args[i] not in unused]

        for arg in unused:
            op.region.block.erase_arg(arg)

        new = stencil.ApplyOp.get(
            operands,
            block := Block(arg_types=bbargs_type),
            [cast(stencil.TempType[Attribute], r.type) for r in op.res],
        )

        rewriter.inline_block(op.region.block, InsertPoint.at_start(block), block.args)
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
