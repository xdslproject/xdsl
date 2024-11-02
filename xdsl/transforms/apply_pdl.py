import os
from dataclasses import dataclass
from io import StringIO

from xdsl.context import MLContext
from xdsl.dialects import builtin, pdl
from xdsl.interpreters.experimental.pdl import (
    PDLRewritePattern,
)
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter


@dataclass(frozen=True)
class ApplyPDLPass(ModulePass):
    name = "apply-pdl"

    pdl_file: str | None = None

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        payload_module = op
        # Target the file containing the PDL specification
        if self.pdl_file is not None:
            assert os.path.exists(self.pdl_file)
            with open(self.pdl_file) as f:
                pdl_module_str = f.read()
                parser = Parser(ctx, pdl_module_str)
                pdl_module = parser.parse_module()
        else:
            pdl_module = payload_module
        # Gather all the pattern operations
        patterns = [op for op in pdl_module.walk() if isinstance(op, pdl.PatternOp)]
        # Process each pattern
        for pattern in patterns:
            for pdl_op in pattern.walk():
                if isinstance(pdl_op, pdl.RewriteOp):
                    stream = StringIO()
                    pattern_rewriter = PatternRewriter(payload_module)
                    pdl_rewrite_pattern = PDLRewritePattern(pdl_op, ctx, stream)
                    pdl_rewrite_pattern.match_and_rewrite(
                        payload_module, pattern_rewriter
                    )
