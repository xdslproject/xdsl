from xdsl.dialects.builtin import (
    StringAttr,
)
from xdsl.dialects.test import TestOp
from xdsl.interactive.get_all_possible_rewrites import (
    IndexedIndividualRewrite,
    IndividualRewrite,
    get_all_possible_rewrites,
)
from xdsl.ir import MLContext
from xdsl.parser import Parser
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.tools.command_line_tool import get_all_dialects


def test_get_all_possible_rewrite():
    # build module
    prog = """
    builtin.module {
    "test.op"() {"label" = "a"} : () -> ()
    "test.op"() {"label" = "a"} : () -> ()
    "test.op"() {"label" = "b"} : () -> ()
    }
    """

    ctx = MLContext(True)
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)
    parser = Parser(ctx, prog)
    module = parser.parse_module()

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: TestOp, rewriter: PatternRewriter):
            if op.attributes["label"] != StringAttr("a"):
                return
            rewriter.replace_matched_op(TestOp(attributes={"label": StringAttr("c")}))

    expected_res = (
        (
            IndexedIndividualRewrite(
                matched_op=TestOp(),
                operation_index=1,
                rewrite=IndividualRewrite(operation="test.op", pattern="TestRewrite"),
            )
        ),
        (
            IndexedIndividualRewrite(
                matched_op=TestOp(),
                operation_index=2,
                rewrite=IndividualRewrite(operation="test.op", pattern="TestRewrite"),
            )
        ),
    )

    res = get_all_possible_rewrites(module, {"test.op": {"TestRewrite": Rewrite()}})
    assert res == expected_res
