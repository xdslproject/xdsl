from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker

from .test_query_builder import add_zero_query


@add_zero_query.rewrite
def add_zero(rewriter: PatternRewriter, root: arith.Addi, rhs_input: arith.Constant):
    rewriter.replace_matched_op((), (root.lhs,))


def test_add_zero_rewrite():
    @ModuleOp
    @Builder.implicit_region
    def module():
        with ImplicitBuilder(func.FuncOp("main", ((), ())).body):
            c0 = arith.Constant.from_int_and_width(0, i32).result
            c1 = arith.Constant.from_int_and_width(1, i32).result
            s = arith.Addi(c1, c0).result
            func.Call("func", ((s,)), ())
            func.Return()

    @ModuleOp
    @Builder.implicit_region
    def expected():
        with ImplicitBuilder(func.FuncOp("main", ((), ())).body):
            _c0 = arith.Constant.from_int_and_width(0, i32).result
            c1 = arith.Constant.from_int_and_width(1, i32).result
            func.Call("func", ((c1,)), ())
            func.Return()

    PatternRewriteWalker(add_zero).rewrite_module(module)

    assert str(module) == str(expected)
