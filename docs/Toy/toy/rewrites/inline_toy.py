from xdsl.ir import MLContext
from xdsl.dialects.builtin import (
    ModuleOp,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    RewritePattern,
    PatternRewriter,
)
from xdsl.transforms.dead_code_elimination import dce

from ..dialects import toy


class InlineFunctions(RewritePattern):
    _func_op_by_name: dict[str, toy.FuncOp] | None = None

    def func_op(self, module: ModuleOp, name: str) -> toy.FuncOp:
        if self._func_op_by_name is None:
            self._func_op_by_name = {
                op.sym_name.data: op for op in module.ops if isinstance(op, toy.FuncOp)
            }
        return self._func_op_by_name[name]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.GenericCallOp, rewriter: PatternRewriter):
        """
        For each generic call, find the function that it calls, and inline it.
        """

        # Get module
        parent = op.parent_op()
        assert isinstance(parent, toy.FuncOp)
        module = parent.parent_op()
        assert isinstance(module, ModuleOp)

        # Clone called function
        impl = self.func_op(module, op.callee.string_value()).clone()
        _ = impl

        # Insert function definition before matched op
        # rewriter.insert_op_before_matched_op(impl)

        # Replace uses of arguments with operands

        # Remove arguments

        # Inline block before function operation

        # Replace return op with casts
        casts: list[toy.CastOp] = []

        # Replace results
        new_results = [cast.res for cast in casts]

        rewriter.replace_op(op, [], new_results)


class InlineToyPass(ModulePass):
    name = "dce"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(InlineFunctions()).rewrite_module(op)
        dce(op)
