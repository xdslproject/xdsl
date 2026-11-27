from xdsl.context import Context
from xdsl.dialects import builtin, cf, func
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import BlockInsertPoint, InsertPoint
from xdsl.traits import SymbolTable


class FuncInlinePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter, /):
        if op.parent is None:
            return

        f = SymbolTable.lookup_symbol(op, op.callee)

        if not isinstance(f, func.FuncOp):
            return

        if f.is_declaration:
            return

        body = f.body.clone()
        first_block = body.blocks.first
        last_block = body.blocks.last
        assert first_block is not None
        assert last_block is not None
        term = last_block.ops.last
        assert isinstance(term, func.ReturnOp)

        before = op.parent
        after = before.split_before(op, arg_types=term.arguments.types)

        rewriter.insert_op(
            cf.BranchOp(first_block, *op.arguments), InsertPoint.at_end(before)
        )

        rewriter.inline_region(body, BlockInsertPoint.before(after))

        rewriter.replace_op(term, cf.BranchOp(after, *term.arguments))

        rewriter.replace_op(op, (), after.args)


class FuncInlinePass(ModulePass):
    name = "func-inline"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            FuncInlinePattern(),
            apply_recursively=False,
        ).rewrite_module(op)
