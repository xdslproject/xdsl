from xdsl import ir, irdl
from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.dialects.test import TestOp
from xdsl.interfaces import HasFolderInterface
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.traits import HasCanonicalizationPatternsTrait
from xdsl.transforms.canonicalize import CanonicalizePass


class Pattern(RewritePattern):
    def match_and_rewrite(self, op: ir.Operation, rewriter: PatternRewriter, /):
        rewriter.erase_op(op)


class HasCanonicalizationPatternsTrait1(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        return ()


class HasCanonicalizationPatternsTrait2(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        return (Pattern(),)


@irdl.irdl_op_definition
class TwoTraitsTestOp(irdl.IRDLOperation):
    name = "test.op"

    traits = irdl.traits_def(
        HasCanonicalizationPatternsTrait1(), HasCanonicalizationPatternsTrait2()
    )


def test_multiple_traits():
    """
    Check that operations with multiple canonicalization patterns traits get patterns
    from all the traits and not just the first one.
    """
    op = TwoTraitsTestOp()
    module = ModuleOp([op])
    ctx = Context()
    ctx.load_op(ModuleOp)
    ctx.load_op(TwoTraitsTestOp)

    CanonicalizePass().apply(ctx, module)

    assert not module.body.block.ops


@irdl.irdl_op_definition
class FoldCanonicalizeTestOp(irdl.IRDLOperation, HasFolderInterface):
    name = "fold_canonicalize_test.op"

    arg = irdl.operand_def()
    res = irdl.result_def()

    def fold(self):
        return (self.arg,)


def test_folding():

    body = ir.Region(ir.Block())

    with ImplicitBuilder(body):
        source = TestOp(result_types=(IndexType(),))
        op = FoldCanonicalizeTestOp(operands=(source.res,), result_types=(IndexType(),))
        sink = TestOp(operands=(op.results))

    module = ModuleOp(body)
    module.verify()

    ctx = Context()
    ctx.load_op(ModuleOp)
    ctx.load_op(TwoTraitsTestOp)

    CanonicalizePass().apply(ctx, module)

    assert tuple(sink.operands) == source.results
    assert op.parent is None
