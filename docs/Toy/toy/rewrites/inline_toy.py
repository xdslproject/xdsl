from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.traits import CallableOpInterface, SymbolTable
from xdsl.transforms.dead_code_elimination import dce

from ..dialects import toy


class InlineFunctions(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.GenericCallOp, rewriter: PatternRewriter):
        """
        For each generic call, find the function that it calls, and inline it.
        """

        callee = SymbolTable.lookup_symbol(op, op.callee)
        assert callee is not None
        callable_interface = callee.get_trait(CallableOpInterface)
        assert callable_interface is not None

        impl_body = callable_interface.get_callable_region(callee)
        assert len(impl_body.blocks) == 1
        # Clone called function body
        impl_block = impl_body.clone().block

        # Cast operands to unranked
        inputs = [toy.CastOp(operand) for operand in op.operands]

        # Insert casts before matched op
        rewriter.insert_op(inputs)

        # Replace block args with operand casts
        for i, arg in zip(inputs, impl_block.args):
            arg.replace_by(i.res)

        # remove block args
        while len(impl_block.args):
            assert not impl_block.args[-1].uses
            rewriter.erase_block_argument(impl_block.args[-1])

        # Inline function definition before matched op
        rewriter.inline_block(impl_block, InsertPoint.before(op))

        # Get return from function definition
        return_op = op.prev_op
        assert return_op is not None

        rewriter.replace_op(op, [], return_op.operands)
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
            rewriter.erase_op(op)


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

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(InlineFunctions()).rewrite_module(op)
        PatternRewriteWalker(RemoveUnusedPrivateFunctions()).rewrite_module(op)
        dce(op)
