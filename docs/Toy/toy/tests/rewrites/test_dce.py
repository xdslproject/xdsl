from xdsl.builder import Builder
from xdsl.dialects.builtin import ModuleOp
from xdsl.pattern_rewriter import PatternRewriteWalker
from ...rewrites.dead_code_elimination import RemoveUnusedOperations
from ...dialects import toy


def test_dce():
    @ModuleOp
    @Builder.implicit_region
    def module():
        a = toy.ConstantOp.from_list([1, 2, 3], [3]).res
        _b = toy.ConstantOp.from_list([1, 2, 3], [3]).res
        toy.PrintOp(a)

    @ModuleOp
    @Builder.implicit_region
    def expected():
        a = toy.ConstantOp.from_list([1, 2, 3], [3]).res
        toy.PrintOp(a)

    PatternRewriteWalker(RemoveUnusedOperations()).rewrite_module(module)

    assert module.is_structurally_equivalent(expected)
