import os
from dataclasses import dataclass
from io import StringIO
from typing import cast

from xdsl.context import MLContext
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
        stream = StringIO()
        rewrite_patterns = [
            cast(RewritePattern, PDLRewritePattern(op, ctx, stream))
            for op in pdl_module.walk()
            if isinstance(op, pdl.RewriteOp)
        ]
        pattern_applier = GreedyRewritePatternApplier(rewrite_patterns)
        PatternRewriteWalker(pattern_applier).rewrite_op(payload_module)
        # pattern_rewriter = PatternRewriter(payload_module)
        # pattern_applier.match_and_rewrite(payload_module, pattern_rewriter)
