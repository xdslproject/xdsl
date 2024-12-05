import os
from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin, pdl
from xdsl.interpreters.eqsat_pdl import EqsatPDLRewritePattern
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    RewritePattern,
)


@dataclass(frozen=True)
class ApplyEqsatPDLPass(ModulePass):
    name = "apply-eqsat-pdl"

    pdl_file: str | None = None

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        payload_module = op
        if self.pdl_file is not None:
            assert os.path.exists(self.pdl_file)
            with open(self.pdl_file) as f:
                pdl_module_str = f.read()
                parser = Parser(ctx, pdl_module_str)
                pdl_module = parser.parse_module()
        else:
            pdl_module = payload_module
        rewrite_patterns: list[RewritePattern] = [
            EqsatPDLRewritePattern(op, ctx, None)
            for op in pdl_module.walk()
            if isinstance(op, pdl.RewriteOp)
        ]
        pattern_applier = GreedyRewritePatternApplier(rewrite_patterns)
        # TODO: remove apply_recursively=False
        PatternRewriteWalker(pattern_applier).rewrite_op(payload_module)
