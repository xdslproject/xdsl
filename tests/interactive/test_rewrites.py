from dataclasses import dataclass
from typing import ClassVar

from xdsl import traits
from xdsl.context import Context
from xdsl.dialects.builtin import (
    Builtin,
    ModuleOp,
    StringAttr,
)
from xdsl.interactive.get_all_available_passes import get_available_pass_list
from xdsl.interactive.passes import AvailablePass
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, traits_def
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.individual_rewrite import ApplyIndividualRewritePass


class MyTestOpHasCanonicalizationPatternsTrait(traits.HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        return (Rewrite(),)


@irdl_op_definition
class MyTestOp(IRDLOperation):
    name = "test.rewrites"

    traits = traits_def(MyTestOpHasCanonicalizationPatternsTrait())


class Rewrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MyTestOp, rewriter: PatternRewriter):
        if op.attributes["label"] != StringAttr("b"):
            return
        rewriter.replace_op(op, MyTestOp(attributes={"label": StringAttr("c")}))


def test_get_all_possible_rewrites():
    # build module
    prog = """
    builtin.module {
    "test.rewrites"() {"label" = "a"} : () -> ()
    "test.rewrites"() {"label" = "b"} : () -> ()
    "test.rewrites"() {"label" = "b"} : () -> ()
    }
    """

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_op(MyTestOp)

    parser = Parser(ctx, prog)
    module = parser.parse_module()

    assert ApplyIndividualRewritePass.schedule_space(ctx, module) == (
        ApplyIndividualRewritePass(2, "test.rewrites", "Rewrite"),
        ApplyIndividualRewritePass(3, "test.rewrites", "Rewrite"),
    )


@dataclass
class ReplacePattern(RewritePattern):
    before: str
    after: str

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MyTestOp, rewriter: PatternRewriter):
        if op.attributes["label"] == StringAttr(self.before):
            rewriter.replace_op(
                op, MyTestOp(attributes={"label": StringAttr(self.after)})
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
        (("test", lambda: Dialect("test", [MyTestOp])),),
        tuple(
            (p.name, p)
            for p in (ABPass, ACPass, BCPass, BDPass, ApplyIndividualRewritePass)
        ),
        '"test.rewrites"() {label="a"} : () -> ()',
        # Transforms the above op from "a" to "b" before testing passes
        (ABPass(),),
        condense_mode=True,
    )

    assert res == (
        AvailablePass(BCPass()),
        AvailablePass(BDPass()),
        AvailablePass(ApplyIndividualRewritePass(1, "test.rewrites", "Rewrite")),
    )
