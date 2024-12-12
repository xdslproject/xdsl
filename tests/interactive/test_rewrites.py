from xdsl.context import MLContext
from xdsl.dialects.builtin import (
    Builtin,
    StringAttr,
)
from xdsl.dialects.test import Test, TestOp
from xdsl.interactive.passes import AvailablePass
from xdsl.interactive.rewrites import (
    IndexedIndividualRewrite,
    IndividualRewrite,
    convert_indexed_individual_rewrites_to_available_pass,
    get_all_possible_rewrites,
)
from xdsl.parser import Parser
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms import individual_rewrite
from xdsl.utils.parse_pipeline import parse_pipeline


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

    ctx = MLContext()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Test)

    parser = Parser(ctx, prog)
    module = parser.parse_module()

    expected_res = [
        (
            IndexedIndividualRewrite(
                1, IndividualRewrite(operation="test.op", pattern="TestRewrite")
            )
        ),
        (
            IndexedIndividualRewrite(
                operation_index=2,
                rewrite=IndividualRewrite(operation="test.op", pattern="TestRewrite"),
            )
        ),
    ]

    res = get_all_possible_rewrites(module, {"test.op": {"TestRewrite": Rewrite()}})
    assert res == expected_res


def test_convert_indexed_individual_rewrites_to_available_pass():
    # build module
    prog = """
    builtin.module {
    "test.op"() {"label" = "a"} : () -> ()
    "test.op"() {"label" = "a"} : () -> ()
    "test.op"() {"label" = "b"} : () -> ()
    }
    """

    ctx = MLContext()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Test)
    parser = Parser(ctx, prog)
    module = parser.parse_module()

    rewrites = (
        (
            IndexedIndividualRewrite(
                1, IndividualRewrite(operation="test.op", pattern="TestRewrite")
            )
        ),
        (
            IndexedIndividualRewrite(
                operation_index=2,
                rewrite=IndividualRewrite(operation="test.op", pattern="TestRewrite"),
            )
        ),
    )

    expected_res = tuple(
        (
            AvailablePass(
                display_name='TestOp("test.op"() {"label" = "a"} : () -> ()):test.op:TestRewrite',
                module_pass=individual_rewrite.ApplyIndividualRewritePass,
                pass_spec=list(
                    parse_pipeline(
                        'apply-individual-rewrite{matched_operation_index=1 operation_name="test.op" pattern_name="TestRewrite"}'
                    )
                )[0],
            ),
            AvailablePass(
                display_name='TestOp("test.op"() {"label" = "a"} : () -> ()):test.op:TestRewrite',
                module_pass=individual_rewrite.ApplyIndividualRewritePass,
                pass_spec=list(
                    parse_pipeline(
                        'apply-individual-rewrite{matched_operation_index=2 operation_name="test.op" pattern_name="TestRewrite"}'
                    )
                )[0],
            ),
        )
    )

    res = convert_indexed_individual_rewrites_to_available_pass(rewrites, module)
    assert res == expected_res
