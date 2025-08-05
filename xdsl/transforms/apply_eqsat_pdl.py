import os
import time
from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import builtin, pdl, pdl_interp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.eqsat_pdl_interp import EqsatPDLInterpFunctions
from xdsl.ir import Operation
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriterListener, PatternRewriteWalker
from xdsl.traits import SymbolTable
from xdsl.transforms.apply_pdl_interp import PDLInterpRewritePattern
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

    def _extract_patterns(self, pdl_module: builtin.ModuleOp) -> list[pdl.PatternOp]:
        """Extract all PDL patterns from the module."""
        patterns: list[pdl.PatternOp] = []
        for pattern_op in pdl_module.walk():
            if isinstance(pattern_op, pdl.PatternOp):
                patterns.append(pattern_op)
        return patterns

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

    def _create_master_module(
        self,
        matchers: list[pdl_interp.FuncOp],
        all_rewriter_ops: list[pdl_interp.FuncOp],
    ) -> builtin.ModuleOp:
        """Create master module containing all matchers and rewriters."""
        rewriters_module_ops: list[builtin.ModuleOp] = []
        if all_rewriter_ops:
            rewriters_module = builtin.ModuleOp(cast(list[Operation], all_rewriter_ops))
            rewriters_module.attributes["sym_name"] = builtin.StringAttr("rewriters")
            rewriters_module_ops.append(rewriters_module)

        all_ops = cast(list[Operation], matchers) + cast(
            list[Operation], rewriters_module_ops
        )
        return builtin.ModuleOp(all_ops)

    def _apply_individual_patterns(
        self, ctx: Context, op: builtin.ModuleOp, pdl_module: builtin.ModuleOp
    ) -> None:
        """Apply patterns individually in separate iterations."""
        patterns = self._extract_patterns(pdl_module)
        print(f"Collected {len(patterns)} patterns.")

        # Create shared implementations for all patterns
        shared_implementations = EqsatPDLInterpFunctions(ctx)
        shared_implementations.populate_known_ops(op)

        # Convert all patterns and collect components
        matchers: list[pdl_interp.FuncOp] = []
        all_rewriter_ops: list[pdl_interp.FuncOp] = []
        names: list[builtin.StringAttr] = []

        for pattern_op in patterns:
            temp_module = self._convert_single_pattern(ctx, pattern_op)
            matcher, rewriter_func = self._extract_matcher_and_rewriters(temp_module)

            assert matcher.body.last_block is not None
            assert isinstance(
                recordmatch := matcher.body.last_block.last_op, pdl_interp.RecordMatchOp
            )
            assert pattern_op.sym_name
            recordmatch.rewriter = builtin.SymbolRefAttr(
                "rewriters", (pattern_op.sym_name,)
            )
            names.append(pattern_op.sym_name)
            rewriter_func.sym_name = pattern_op.sym_name

            # Detach and collect operations
            matcher.detach()
            matchers.append(matcher)

            rewriter_func.detach()
            all_rewriter_ops.append(rewriter_func)

        # Create master module and interpreter
        master_module = self._create_master_module(matchers, all_rewriter_ops)
        master_interpreter = Interpreter(master_module)
        master_interpreter.register_implementations(shared_implementations)
        print(f"Collected {len(matchers)} matchers.")

        # Create rewrite patterns
        rewrite_patterns: list[PDLInterpRewritePattern] = []
        for name, matcher in zip(names, matchers):
            rewrite_pattern = PDLInterpRewritePattern(
                matcher, master_interpreter, shared_implementations, name.data
            )
            rewrite_patterns.append(rewrite_pattern)
        print(f"Collected {len(rewrite_patterns)} rewrite patterns.")

        # Initialize listener
        listener = PatternRewriterListener()
        listener.operation_modification_handler.append(
            shared_implementations.modification_handler
        )

        # Main iteration loop
        for _i in range(self.max_iterations):
            print(f"Starting iteration {_i + 1} of {self.max_iterations}")
            # Apply each pattern individually
            for rewrite_pattern in rewrite_patterns:
                assert rewrite_pattern.matcher is not None
                print(f"Applying pattern: {rewrite_pattern.name}", end="", flush=True)
                walker = PatternRewriteWalker(rewrite_pattern, apply_recursively=False)
                walker.listener = listener
                t0 = time.time()
                walker.rewrite_module(op)
                t1 = time.time()
                print(f" - took {t1 - t0:.2f} seconds")

            # Execute all pending rewrites
            shared_implementations.execute_pending_rewrites(master_interpreter)

            if not shared_implementations.worklist:
                break

            shared_implementations.rebuild()

    def _apply_combined_patterns(
        self, ctx: Context, op: builtin.ModuleOp, pdl_module: builtin.ModuleOp
    ) -> None:
        """Apply all patterns together (original behavior)."""
        pdl_to_pdl_interp = MLIROptPass(
            arguments=("--convert-pdl-to-pdl-interp", "-allow-unregistered-dialect")
        )
        pdl_to_pdl_interp.apply(ctx, pdl_module)
        pdl_interp_module = pdl_module

        matcher = SymbolTable.lookup_symbol(pdl_interp_module, "matcher")
        assert isinstance(matcher, pdl_interp.FuncOp)
        assert matcher is not None, "matcher function not found"

        # Initialize interpreter and implementations
        interpreter = Interpreter(pdl_interp_module)
        implementations = EqsatPDLInterpFunctions(ctx)
        implementations.populate_known_ops(op)
        interpreter.register_implementations(implementations)
        rewrite_pattern = PDLInterpRewritePattern(matcher, interpreter, implementations)

        listener = PatternRewriterListener()
        listener.operation_modification_handler.append(
            implementations.modification_handler
        )
        walker = PatternRewriteWalker(rewrite_pattern, apply_recursively=False)
        walker.listener = listener

        for _i in range(self.max_iterations):
            walker.rewrite_module(op)
            implementations.execute_pending_rewrites(interpreter)

            if not implementations.worklist:
                break

            implementations.rebuild()

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        pdl_module = self._load_pdl_module(ctx, op)

        if self.individual_patterns:
            self._apply_individual_patterns(ctx, op, pdl_module)
        else:
            self._apply_combined_patterns(ctx, op, pdl_module)
