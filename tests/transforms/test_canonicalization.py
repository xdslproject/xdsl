from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.irdl import IRDLOperation, irdl_op_definition, traits_def
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.traits import HasCanonicalizationPatternsTrait
from xdsl.transforms.canonicalize import CanonicalizePass


class Pattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        rewriter.erase_op(op)


class HasCanonicalizationPatternsTrait1(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        return ()


class HasCanonicalizationPatternsTrait2(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        return (Pattern(),)


@irdl_op_definition
class TestOp(IRDLOperation):
    name = "test.op"

    traits = traits_def(
        HasCanonicalizationPatternsTrait1(), HasCanonicalizationPatternsTrait2()
    )


def test_multiple_traits():
    """
    Check that operations with multiple canonicalization patterns traits get patterns
    from all the traits and not just the first one.
    """
    op = TestOp()
    module = ModuleOp([op])
    ctx = Context()
    ctx.load_op(ModuleOp)
    ctx.load_op(TestOp)

    CanonicalizePass().apply(ctx, module)

    assert not module.body.block.ops
