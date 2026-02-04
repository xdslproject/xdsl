import os
from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import builtin, pdl_interp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.ematch import EmatchFunctions
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriterListener, PatternRewriteWalker
from xdsl.traits import SymbolTable
from xdsl.transforms.apply_pdl_interp import PDLInterpRewritePattern


@dataclass(frozen=True)
class EmatchSaturatePass(ModulePass):
    """
    A pass that applies PDL patterns using equality saturation.
    """

    name = "ematch-saturate"

    pdl_file: str | None = None
    """Path to external PDL file containing patterns. If None, patterns are taken from the input module."""

    max_iterations: int = 20
    """Maximum number of iterations to run the equality saturation algorithm."""

    def _load_pdl_module(self, ctx: Context, op: builtin.ModuleOp) -> builtin.ModuleOp:
        """Load PDL module from file or use the input module."""
        if self.pdl_file is not None:
            assert os.path.exists(self.pdl_file)
            with open(self.pdl_file) as f:
                pdl_module_str = f.read()
                parser = Parser(ctx, pdl_module_str)
                return parser.parse_module()
        else:
            return op

    def _extract_matcher_and_rewriters(
        self, temp_module: builtin.ModuleOp
    ) -> tuple[pdl_interp.FuncOp, pdl_interp.FuncOp]:
        """Extract matcher and rewriter function from converted module."""
        matcher = SymbolTable.lookup_symbol(temp_module, "matcher")
        assert isinstance(matcher, pdl_interp.FuncOp)
        assert matcher is not None, "matcher function not found"

        rewriter_module = cast(
            builtin.ModuleOp, SymbolTable.lookup_symbol(temp_module, "rewriters")
        )
        assert rewriter_module.body.first_block is not None
        rewriter_func = rewriter_module.body.first_block.first_op
        assert isinstance(rewriter_func, pdl_interp.FuncOp)

        return matcher, rewriter_func

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Apply all patterns together (original behavior)."""
        pdl_module = self._load_pdl_module(ctx, op)
        # TODO: convert pdl to pdl-interp if necessary
        pdl_interp_module = pdl_module

        matcher = SymbolTable.lookup_symbol(pdl_interp_module, "matcher")
        assert isinstance(matcher, pdl_interp.FuncOp)
        assert matcher is not None, "matcher function not found"

        # Initialize interpreter and implementations
        interpreter = Interpreter(pdl_interp_module)
        pdl_interp_functions = PDLInterpFunctions()
        ematch_functions = EmatchFunctions()
        PDLInterpFunctions.set_ctx(interpreter, ctx)
        ematch_functions.populate_known_ops(op)
        interpreter.register_implementations(ematch_functions)
        interpreter.register_implementations(pdl_interp_functions)
        rewrite_pattern = PDLInterpRewritePattern(
            matcher, interpreter, pdl_interp_functions
        )

        listener = PatternRewriterListener()
        listener.operation_modification_handler.append(
            ematch_functions.modification_handler
        )
        walker = PatternRewriteWalker(rewrite_pattern, apply_recursively=False)
        walker.listener = listener

        for _i in range(self.max_iterations):
            walker.rewrite_module(op)
            ematch_functions.execute_pending_rewrites(interpreter)

            if not ematch_functions.worklist:
                break

            ematch_functions.rebuild(interpreter)
