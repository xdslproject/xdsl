import os
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, equivalence, pdl_interp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.ematch import EmatchFunctions
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.traits import SymbolTable


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

    def _saturate_graph(
        self,
        graph: equivalence.GraphOp,
        matcher: pdl_interp.FuncOp,
        pdl_interp_module: builtin.ModuleOp,
        ctx: Context,
    ) -> None:
        """Run equality saturation on a single equivalence.graph operation."""
        interpreter = Interpreter(pdl_interp_module)
        pdl_interp_functions = PDLInterpFunctions()
        ematch_functions = EmatchFunctions()
        PDLInterpFunctions.set_ctx(interpreter, ctx)
        ematch_functions.populate_known_ops(graph)
        interpreter.register_implementations(ematch_functions)
        interpreter.register_implementations(pdl_interp_functions)

        rewriter = PatternRewriter(graph)
        rewriter.operation_modification_handler.append(
            ematch_functions.modification_handler
        )
        pdl_interp_functions.set_rewriter(interpreter, rewriter)

        for _i in range(self.max_iterations):
            for root in graph.walk():
                rewriter.current_operation = root
                interpreter.call_op(matcher, (root,))
            pdl_interp_functions.apply_pending_rewrites(interpreter)

            if not ematch_functions.worklist:
                break

            ematch_functions.rebuild(interpreter)

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Apply PDL patterns using equality saturation on each equivalence.graph."""
        pdl_module = self._load_pdl_module(ctx, op)
        # TODO: convert pdl to pdl-interp if necessary
        pdl_interp_module = pdl_module

        matcher = SymbolTable.lookup_symbol(pdl_interp_module, "matcher")
        assert isinstance(matcher, pdl_interp.FuncOp)
        assert matcher is not None, "matcher function not found"

        for graph in op.walk():
            if isinstance(graph, equivalence.GraphOp):
                self._saturate_graph(graph, matcher, pdl_interp_module, ctx)
