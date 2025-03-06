from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
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
from xdsl.transforms.individual_rewrite import ApplyIndividualRewritePass


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

    def apply(self, ctx: Context, op: ModuleOp) -> None:
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


class BDPass(ReplacePass):
    name = "bd"
    before = "b"
    after = "d"


def test_get_all_available_passes():
    res = get_available_pass_list(
        ((Test.name, lambda: Test),),
        tuple((p.name, p) for p in (ABPass, ACPass, BCPass, BDPass)),
        '"test.op"() {key="a"} : () -> ()',
        # Transforms the above op from "a" to "b" before testing passes
        (ABPass(),),
        condense_mode=True,
        rewrite_by_names_dict={
            "test.op": {
                "ae": ReplacePattern("a", "e"),
                "be": ReplacePattern("b", "e"),
            }
        },
    )

    assert res == tuple(
        (
            AvailablePass(
                display_name="bc",
                module_pass=BCPass(),
            ),
            AvailablePass(
                display_name="bd",
                module_pass=BDPass(),
            ),
            AvailablePass(
                display_name='TestOp("test.op"() {key = "b"} : () -> ()):test.op:be',
                module_pass=ApplyIndividualRewritePass(1, "test.op", "be"),
            ),
        )
    )
