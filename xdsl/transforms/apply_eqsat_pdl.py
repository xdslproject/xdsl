import os
from dataclasses import dataclass
from typing import cast

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import builtin, eqsat_pdl_interp, pdl, pdl_interp
from xdsl.dialects.builtin import StringAttr
from xdsl.interpreter import Interpreter
from xdsl.interpreters.eqsat_pdl_interp import EqsatPDLInterpFunctions
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable
from xdsl.transforms.apply_eqsat_pdl_interp import EqsatConstraintFunctions
from xdsl.transforms.convert_pdl_interp_to_eqsat_pdl_interp import (
    ConvertPDLInterpToEqsatPDLInterpPass,
)
from xdsl.transforms.mlir_opt import MLIROptPass


@dataclass(frozen=True)
class ApplyEqsatPDLPass(ModulePass):
    """
    A pass that applies PDL patterns using equality saturation.
    """

    name = "apply-eqsat-pdl"

    pdl_file: str | None = None
    """Path to external PDL file containing patterns. If None, patterns are taken from the input module."""

    max_iterations: int = 20
    """Maximum number of iterations to run the equality saturation algorithm."""

    individual_patterns: bool = False
    """
    Whether to convert and apply patterns individually rather than all together.

    When True: Each pattern is converted to PDL_interp separately and applied individually
    in each iteration.

    When False (default): All patterns are converted together, potentially producing a more efficient
    matcher by reusing equivalent expressions.
    """

    optimize_matcher: bool = False
    """When enabled, the matcher is optimized to evaluate equality constraints early."""

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

    def _convert_single_pattern(
        self, ctx: Context, pattern_op: pdl.PatternOp
    ) -> builtin.ModuleOp:
        """Convert a single PDL pattern to PDL_interp."""
        pattern_copy = pattern_op.clone()
        temp_module = builtin.ModuleOp([pattern_copy])

        pdl_to_pdl_interp = MLIROptPass(
            arguments=("--convert-pdl-to-pdl-interp", "-allow-unregistered-dialect")
        )
        pdl_to_pdl_interp.apply(ctx, temp_module)
        ConvertPDLInterpToEqsatPDLInterpPass().apply(ctx, temp_module)
        return temp_module

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

    def _apply_individual_patterns(
        self, ctx: Context, op: builtin.ModuleOp, pdl_module: builtin.ModuleOp
    ) -> None:
        """Apply patterns individually in separate iterations."""
        patterns = (
            pattern for pattern in pdl_module.ops if isinstance(pattern, pdl.PatternOp)
        )

        implementations = EqsatPDLInterpFunctions()
        implementations.populate_known_ops(op)

        matchers_module = builtin.ModuleOp([])
        rewriters_module = builtin.ModuleOp([], sym_name=StringAttr("rewriters"))
        matchers_builder = Builder(InsertPoint.at_end(matchers_module.body.block))
        matchers_builder.insert_op(rewriters_module)
        rewriters_builder = Builder(InsertPoint.at_end(rewriters_module.body.block))

        interpreter = Interpreter(matchers_module)
        PDLInterpFunctions.set_ctx(interpreter, ctx)
        interpreter.register_implementations(implementations)
        interpreter.register_implementations(
            vanilla_pdl_interp_implementations := PDLInterpFunctions()
        )
        interpreter.register_implementations(EqsatConstraintFunctions())

        matchers: list[pdl_interp.FuncOp] = []
        for pattern_op in patterns:
            temp_module = self._convert_single_pattern(ctx, pattern_op)
            matcher, rewriter_func = self._extract_matcher_and_rewriters(temp_module)

            assert matcher.body.last_block is not None
            assert isinstance(
                recordmatch := matcher.body.last_block.last_op,
                eqsat_pdl_interp.RecordMatchOp,
            )
            name = (
                pattern_op.sym_name
                if pattern_op.sym_name
                else StringAttr(f"pattern_{len(matchers)}")
            )
            recordmatch.rewriter = builtin.SymbolRefAttr("rewriters", (name,))
            rewriter_func.sym_name = name

            # Detach and insert operations
            matcher.detach()
            matchers_builder.insert_op(matcher)

            rewriter_func.detach()
            rewriters_builder.insert_op(rewriter_func)

            matchers.append(matcher)

        if not op.ops.first:
            return

        rewriter = PatternRewriter(op.ops.first)
        rewriter.operation_modification_handler.append(
            implementations.modification_handler
        )
        vanilla_pdl_interp_implementations.set_rewriter(interpreter, rewriter)

        # Main iteration loop
        for _i in range(self.max_iterations):
            # Apply each pattern individually
            for matcher in matchers:
                for root in op.body.walk():
                    rewriter.current_operation = root
                    interpreter.call_op(matcher, (root,))

            # Execute all pending rewrites
            implementations.execute_pending_rewrites(interpreter)

            if not implementations.worklist:
                break

            implementations.rebuild(interpreter)

    def _apply_combined_patterns(
        self, ctx: Context, op: builtin.ModuleOp, pdl_module: builtin.ModuleOp
    ) -> None:
        """Apply all patterns together (original behavior)."""
        pdl_to_pdl_interp = MLIROptPass(
            arguments=("--convert-pdl-to-pdl-interp", "-allow-unregistered-dialect")
        )
        pdl_to_pdl_interp.apply(ctx, pdl_module)
        pdl_interp_module = pdl_module
        ConvertPDLInterpToEqsatPDLInterpPass().apply(ctx, pdl_interp_module)

        matcher = SymbolTable.lookup_symbol(pdl_interp_module, "matcher")
        assert isinstance(matcher, pdl_interp.FuncOp)
        assert matcher is not None, "matcher function not found"

        # Initialize interpreter and implementations
        interpreter = Interpreter(pdl_interp_module)
        pdl_interp_functions = PDLInterpFunctions()
        eqsat_pdl_interp_functions = EqsatPDLInterpFunctions()
        PDLInterpFunctions.set_ctx(interpreter, ctx)
        eqsat_pdl_interp_functions.populate_known_ops(op)
        interpreter.register_implementations(eqsat_pdl_interp_functions)
        interpreter.register_implementations(pdl_interp_functions)
        interpreter.register_implementations(EqsatConstraintFunctions())

        if not op.ops.first:
            return

        rewriter = PatternRewriter(op.ops.first)
        rewriter.operation_modification_handler.append(
            eqsat_pdl_interp_functions.modification_handler
        )
        pdl_interp_functions.set_rewriter(interpreter, rewriter)

        for _i in range(self.max_iterations):
            for root in op.body.walk():
                rewriter.current_operation = root
                interpreter.call_op(matcher, (root,))
            eqsat_pdl_interp_functions.execute_pending_rewrites(interpreter)

            if not eqsat_pdl_interp_functions.worklist:
                break

            eqsat_pdl_interp_functions.rebuild(interpreter)

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        pdl_module = self._load_pdl_module(ctx, op)

        if self.individual_patterns:
            self._apply_individual_patterns(ctx, op, pdl_module)
        else:
            self._apply_combined_patterns(ctx, op, pdl_module)
