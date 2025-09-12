import os
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, eqsat, pdl_interp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.eqsat_pdl_interp import EqsatPDLInterpFunctions
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriterListener, PatternRewriteWalker
from xdsl.traits import SymbolTable
from xdsl.transforms.apply_pdl_interp import PDLInterpRewritePattern
from xdsl.transforms.eqsat_optimize_pdl_interp import MatcherOptimizer

DEFAULT_MAX_ITERATIONS = 20
"""Default number of times to iterate over the module."""


def apply_eqsat_pdl_interp(
    op: builtin.ModuleOp,
    ctx: Context,
    pdl_interp_module: builtin.ModuleOp,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    optimize_matcher: bool = False,
):
    matcher = SymbolTable.lookup_symbol(pdl_interp_module, "matcher")
    assert isinstance(matcher, pdl_interp.FuncOp)
    assert matcher is not None, "matcher function not found"

    if optimize_matcher:
        mo = MatcherOptimizer(matcher)
        mo.optimize()

    # Initialize interpreter and implementations once
    interpreter = Interpreter(pdl_interp_module)
    implementations = EqsatPDLInterpFunctions(ctx)
    implementations.interpreter = interpreter
    implementations.populate_known_ops(op)
    interpreter.register_implementations(implementations)
    rewrite_pattern = PDLInterpRewritePattern(matcher, interpreter, implementations)

    listener = PatternRewriterListener()
    listener.operation_modification_handler.append(implementations.modification_handler)
    walker = PatternRewriteWalker(rewrite_pattern, apply_recursively=False)
    walker.listener = listener

    for i in range(max_iterations):
        print(f"Equality saturation iteration {i + 1} of {max_iterations}")
        n_eclasses = 0
        n_enodes = 0
        for o in op.walk():
            if isinstance(o, eqsat.EClassOp):
                n_eclasses += 1
                n_enodes += len(o.operands)
        print("Egraphs has ", n_eclasses, "e-classes and", n_enodes, "e-nodes")
        print()
        print(op)
        print()
        # Register matches by walking the module
        walker.rewrite_module(op)
        # Execute all pending rewrites that were aggregated during matching
        implementations.execute_pending_rewrites(interpreter)

        if not implementations.worklist:
            break

        implementations.rebuild()


@dataclass(frozen=True)
class ApplyEqsatPDLInterpPass(ModulePass):
    name = "apply-eqsat-pdl-interp"

    pdl_interp_file: str | None = None
    """Path to external file containing pdl_interp matcher and rewriters.
    If None, matcher and rewriters are taken from the input module."""

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    """Maximum number of iterations to run, default 20."""

    optimize_matcher: bool = False
    """When enabled, the matcher is optimized to evaluate equality constraints early."""

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        if self.pdl_interp_file is not None:
            assert os.path.exists(self.pdl_interp_file)
            with open(self.pdl_interp_file) as f:
                pdl_interp_module_str = f.read()
                parser = Parser(ctx, pdl_interp_module_str)
                pdl_interp_module = parser.parse_module()
        else:
            pdl_interp_module = op

        apply_eqsat_pdl_interp(
            op, ctx, pdl_interp_module, self.max_iterations, self.optimize_matcher
        )
