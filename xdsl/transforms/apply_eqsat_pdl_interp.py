import os
from collections.abc import Callable
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, pdl_interp
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl_external,
    register_impls,
)
from xdsl.interpreters.eqsat_pdl_interp import EqsatPDLInterpFunctions
from xdsl.ir import Operation
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriterListener, PatternRewriteWalker
from xdsl.traits import SymbolTable
from xdsl.transforms.apply_pdl_interp import PDLInterpRewritePattern

_DEFAULT_MAX_ITERATIONS = 20
"""Default number of times to iterate over the module."""


# TODO: remove the constraint functions here (https://github.com/xdslproject/xdsl/issues/5391)
@register_impls
class EqsatConstraintFunctions(InterpreterFunctions):
    @impl_external("is_not_unsound")
    def run_is_not_unsound(
        self, interp: Interpreter, _op: Operation, args: PythonValues
    ):
        assert isinstance(op := args[0], Operation)
        return "unsound" not in op.attributes, ()


def apply_eqsat_pdl_interp(
    op: builtin.ModuleOp,
    ctx: Context,
    pdl_interp_module: builtin.ModuleOp,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    callback: Callable[[builtin.ModuleOp], None] | None = None,
):
    matcher = SymbolTable.lookup_symbol(pdl_interp_module, "matcher")
    assert isinstance(matcher, pdl_interp.FuncOp)
    assert matcher is not None, "matcher function not found"

    # Initialize interpreter and implementations once
    interpreter = Interpreter(pdl_interp_module)
    implementations = EqsatPDLInterpFunctions()
    EqsatPDLInterpFunctions.set_ctx(interpreter, ctx)
    implementations.populate_known_ops(op)
    interpreter.register_implementations(implementations)
    interpreter.register_implementations(EqsatConstraintFunctions())
    rewrite_pattern = PDLInterpRewritePattern(matcher, interpreter, implementations)

    listener = PatternRewriterListener()
    listener.operation_modification_handler.append(implementations.modification_handler)
    walker = PatternRewriteWalker(rewrite_pattern, apply_recursively=False)
    walker.listener = listener

    for _i in range(max_iterations):
        # Register matches by walking the module
        walker.rewrite_module(op)
        # Execute all pending rewrites that were aggregated during matching
        implementations.execute_pending_rewrites(interpreter)

        if not implementations.worklist:
            break

        implementations.rebuild(interpreter)
        if callback is not None:
            callback(op)


@dataclass(frozen=True)
class ApplyEqsatPDLInterpPass(ModulePass):
    name = "apply-eqsat-pdl-interp"

    pdl_interp_file: str | None = None
    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    """Maximum number of iterations to run, default 20."""

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        if self.pdl_interp_file is not None:
            assert os.path.exists(self.pdl_interp_file)
            with open(self.pdl_interp_file) as f:
                pdl_interp_module_str = f.read()
                parser = Parser(ctx, pdl_interp_module_str)
                pdl_interp_module = parser.parse_module()
        else:
            pdl_interp_module = op

        apply_eqsat_pdl_interp(op, ctx, pdl_interp_module, self.max_iterations)
