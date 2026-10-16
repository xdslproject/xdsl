"""Apply one directly interpreted Match pattern with a PDLInterp rewriter."""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import match
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.match import run_matcher
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern
from xdsl.utils.exceptions import PassFailedException


@dataclass
class MatchRewritePattern(RewritePattern):
    matcher: match.MatcherOp
    interpreter: Interpreter

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        result = run_matcher(self.matcher, op)
        if result is None:
            return
        PDLInterpFunctions.set_root(self.interpreter, op)
        PDLInterpFunctions.set_rewriter(self.interpreter, rewriter)
        try:
            self.interpreter.call_op(result.rewriter, result.inputs)
        finally:
            PDLInterpFunctions.set_rewriter(self.interpreter, None)
            PDLInterpFunctions.set_root(self.interpreter, None)


class ApplyMatchPass(ModulePass):
    """Apply one top-level, straight-line matcher with one final success."""

    name = "apply-match"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        matchers = [child for child in op.walk() if isinstance(child, match.MatcherOp)]
        if len(matchers) != 1 or matchers[0].parent_op() is not op:
            raise PassFailedException("expected exactly one top-level match.matcher")
        interpreter = Interpreter(op)
        interpreter.register_implementations(PDLInterpFunctions())
        PDLInterpFunctions.set_ctx(interpreter, ctx)
        PatternRewriteWalker(
            MatchRewritePattern(matchers[0], interpreter)
        ).rewrite_module(op)
