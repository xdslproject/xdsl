from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.dead_code_elimination import dce

from ..dialects import toy


class InlineFunctions(RewritePattern):
    _func_op_by_name: dict[str, toy.FuncOp] | None = None

    def lookup_func_op(self, module: ModuleOp, name: str) -> toy.FuncOp:
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
        impl = self.lookup_func_op(module, op.callee.string_value()).clone()
        impl_block = impl.body.block

        # Cast operands to unranked
        inputs = [toy.CastOp(operand) for operand in op.operands]

        # Insert casts before matched op
        rewriter.insert_op_before_matched_op(inputs)

        # Replace block args with operand casts
        for i, arg in zip(inputs, impl_block.args):
            arg.replace_by(i.res)

        # remove block args
        while len(impl_block.args):
            assert not len(impl_block.args[-1].uses)
            rewriter.erase_block_argument(impl_block.args[-1])

        # Inline function definition before matched op
        rewriter.inline_block_before_matched_op(impl_block)

        # Get return from function definition
        return_op = op.prev_op
        assert return_op is not None

        rewriter.replace_matched_op([], return_op.operands)
        rewriter.erase_op(return_op)


class RemoveUnusedPrivateFunctions(RewritePattern):
    _used_funcs: set[str] | None = None

    def should_remove_op(self, op: toy.FuncOp) -> bool:
        if op.sym_visibility != StringAttr("private"):
            return False

        if self._used_funcs is None:
            # Get module
            module = op.parent_op()
            assert isinstance(module, ModuleOp)

            self._used_funcs = {
                op.callee.string_value()
                for op in module.walk()
                if isinstance(op, toy.GenericCallOp)
            }

        return op.sym_name.data not in self._used_funcs

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.FuncOp, rewriter: PatternRewriter):
        if self.should_remove_op(op):
            rewriter.erase_matched_op()


class InlineToyPass(ModulePass):
    """
    A custom pass to inline Toy functions. In MLIR, this is done through an interface, as
    described in the Toy tutorial. As of time of writing, we don't have dialect interfaces
    in xDSL, but would like to add them. This pass should be migrated to the built-in pass
    when they land.

    https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/
    https://github.com/xdslproject/xdsl/issues/957

    """

    name = "inline-toy-functions"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(InlineFunctions()).rewrite_module(op)
        PatternRewriteWalker(RemoveUnusedPrivateFunctions()).rewrite_module(op)
        dce(op)
