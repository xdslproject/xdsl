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
        new_arg_attrs: list[DictionaryAttr] = []
        for arg, arg_attr in zip(op.args, arg_attrs):
            if arg.name_hint:
                d_attr = DictionaryAttr(
                    {"llvm.name": StringAttr(arg.name_hint), **arg_attr.data}
                )
            else:
                d_attr = arg_attr
            new_arg_attrs.append(d_attr)

        op.arg_attrs = ArrayAttr(new_arg_attrs)
        rewriter.has_done_action = True


class FunctionPersistArgNames(ModulePass):
    name = "function-persist-arg-names"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            ArgNamesToArgAttrsPass(), apply_recursively=False
        ).rewrite_module(op)
