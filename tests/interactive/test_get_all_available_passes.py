from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.test import Test, TestOp
from xdsl.interactive.get_all_available_passes import get_available_pass_list
from xdsl.interactive.passes import AvailablePass
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.individual_rewrite import IndividualRewrite
from xdsl.utils.parse_pipeline import PipelinePassSpec


@dataclass
class ReplacePattern(RewritePattern):
    before: str
    after: str

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TestOp, rewriter: PatternRewriter):
        if op.attributes["key"] == StringAttr(self.before):
            rewriter.replace_matched_op(
                TestOp(attributes={"key": StringAttr(self.after)})
            )


class ReplacePass(ModulePass):
    before: ClassVar[str]
    after: ClassVar[str]

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(ReplacePattern(self.before, self.after)).rewrite_module(op)


class ABPass(ReplacePass):
    name = "ab"
    before = "a"
    after = "b"


class ACPass(ReplacePass):
    name = "ac"
    before = "a"
    after = "c"


class BCPass(ReplacePass):
    name = "bc"
    before = "b"
    after = "c"


def test_get_all_available_passes():
    input_text = """
    "test.op"() {key="a"} : () -> ()
    """

    pass_pipeline = (
        (
            ABPass,
            PipelinePassSpec(name="ab", args={}),
        ),
    )

    expected_res = tuple(
        (
            AvailablePass(
                display_name="bc",
                module_pass=BCPass,
                pass_spec=None,
            ),
            AvailablePass(
                display_name='TestOp("test.op"() {"key" = "b"} : () -> ()):test.op:bd',
                module_pass=IndividualRewrite,
                pass_spec=PipelinePassSpec(
                    "apply-individual-rewrite",
                    {
                        "matched_operation_index": (1,),
                        "operation_name": ("test.op",),
                        "pattern_name": ("bd",),
                    },
                ),
            ),
        )
    )

    all_dialects = ((Test.name, lambda: Test),)
    all_passes = tuple((p.name, p) for p in (ABPass, ACPass, BCPass))

    rewrite_by_names_dict: dict[str, dict[str, RewritePattern]] = {
        "test.op": {
            "bd": ReplacePattern("b", "d"),
        }
    }

    res = get_available_pass_list(
        all_dialects,
        all_passes,
        input_text,
        pass_pipeline,
        condense_mode=True,
        rewrite_by_names_dict=rewrite_by_names_dict,
    )

    assert res == expected_res
