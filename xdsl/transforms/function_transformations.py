from xdsl.context import MLContext
from xdsl.dialects import builtin, func
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ArgNamesToArgAttrsPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        arg_attrs = list(op.arg_attrs or (len(op.args) * [DictionaryAttr({})]))

        new_arg_attrs = ArrayAttr(
            DictionaryAttr(
                {"llvm.name": StringAttr(arg.name_hint), **arg_attr.data}
                if arg.name_hint
                else arg_attr.data
            )
            for arg, arg_attr in zip(op.args, arg_attrs, strict=True)
        )

        if new_arg_attrs != op.arg_attrs:
            op.arg_attrs = new_arg_attrs
            rewriter.has_done_action = True


class FunctionPersistArgNames(ModulePass):
    """
    Persists func.func arg name hints to arg_attrs.

    Such that, for instance
        `func.func @my_func(%arg_name : i32) -> ...`
    becomes
        `func.func @my_func(%arg_name : i32 {"llvm.name" = "arg_name"}) -> ...
    """

    name = "function-persist-arg-names"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            ArgNamesToArgAttrsPass(), apply_recursively=False
        ).rewrite_module(op)
