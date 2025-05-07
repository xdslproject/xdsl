import os
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, pdl
from xdsl.interpreters.pdl import (
    PDLRewritePattern,
)
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    RewritePattern,
)


@dataclass(frozen=True)
class ApplyPDLPass(ModulePass):
    name = "apply-pdl"

    pdl_file: str | None = None

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        if self.pdl_file is not None:
            assert os.path.exists(self.pdl_file)
            with open(self.pdl_file) as f:
                pdl_module_str = f.read()
                parser = Parser(ctx, pdl_module_str)
                pdl_module = parser.parse_module()
        else:
            pdl_module = op
        rewrite_patterns: list[RewritePattern] = [
            PDLRewritePattern(op, ctx, None)
            for op in pdl_module.walk()
            if isinstance(op, pdl.RewriteOp)
        ]
        pattern_applier = GreedyRewritePatternApplier(rewrite_patterns)
        PatternRewriteWalker(pattern_applier).rewrite_module(op)
