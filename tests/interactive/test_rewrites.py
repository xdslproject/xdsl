from xdsl.context import Context
from xdsl.dialects.builtin import (
    Builtin,
    StringAttr,
)
from xdsl.dialects.test import Test, TestOp
from xdsl.interactive.passes import AvailablePass
from xdsl.interactive.rewrites import get_all_possible_rewrites
from xdsl.parser import Parser
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.individual_rewrite import ApplyIndividualRewritePass


class Rewrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TestOp, rewriter: PatternRewriter):
        if op.attributes["label"] != StringAttr("a"):
            return
        rewriter.replace_matched_op(TestOp(attributes={"label": StringAttr("c")}))


def test_get_all_possible_rewrite():
    # build module
    prog = """
    builtin.module {
    "test.op"() {"label" = "a"} : () -> ()
    "test.op"() {"label" = "a"} : () -> ()
    "test.op"() {"label" = "b"} : () -> ()
    }
    """

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Test)

    parser = Parser(ctx, prog)
    module = parser.parse_module()

    expected_res = [
        AvailablePass(
            display_name='TestOp("test.op"() {label = "a"} : () -> ()):test.op:TestRewrite',
            module_pass=ApplyIndividualRewritePass(1, "test.op", "TestRewrite"),
        ),
        AvailablePass(
            display_name='TestOp("test.op"() {label = "a"} : () -> ()):test.op:TestRewrite',
            module_pass=ApplyIndividualRewritePass(2, "test.op", "TestRewrite"),
        ),
    ]

    res = get_all_possible_rewrites(module, {"test.op": {"TestRewrite": Rewrite()}})
    assert res == expected_res
