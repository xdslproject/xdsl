from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.pyast.utils.builder import PyASTBuilder
from xdsl.frontend.pypdl.transforms.func_to_pdl_rewrite import (
    FuncOpToPdlRewritePattern,
    ReturnOpToPdlRewritePattern,
)
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker


@dataclass
class PyPDLRewriteBuilder(PyASTBuilder):
    """Builder for PDL rewrites from aspects of a Python function."""

    def build(self) -> ModuleOp:
        """Build a PDL rewrite from the builder state."""
        module = super().build()
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    FuncOpToPdlRewritePattern(),
                    ReturnOpToPdlRewritePattern(),
                ]
            )
        ).rewrite_module(module)
        return module
