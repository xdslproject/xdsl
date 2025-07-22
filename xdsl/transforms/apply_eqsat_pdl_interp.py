import os
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, pdl_interp
from xdsl.dialects.eqsat import EClassOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.eqsat_pdl_interp import EqsatPDLInterpFunctions
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriterListener, PatternRewriteWalker
from xdsl.traits import SymbolTable
from xdsl.transforms.apply_pdl_interp import PDLInterpRewritePattern


@dataclass(frozen=True)
class ApplyEqsatPDLInterpPass(ModulePass):
    name = "apply-eqsat-pdl-interp"

    pdl_interp_file: str | None = None
    max_iterations: int = 20  # Maximum number of iterations to run

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        if self.pdl_interp_file is not None:
            assert os.path.exists(self.pdl_interp_file)
            with open(self.pdl_interp_file) as f:
                pdl_interp_module_str = f.read()
                parser = Parser(ctx, pdl_interp_module_str)
                pdl_interp_module = parser.parse_module()
        else:
            pdl_interp_module = op
        matcher = SymbolTable.lookup_symbol(
            pdl_interp_module, builtin.SymbolRefAttr("matcher")
        )
        assert isinstance(matcher, pdl_interp.FuncOp), "matcher function not found"
        rewriters_module = SymbolTable.lookup_symbol(
            pdl_interp_module, builtin.SymbolRefAttr("rewriters")
        )
        assert isinstance(rewriters_module, builtin.ModuleOp), "rewriters not found"

        # Initialize interpreter and implementations once
        interpreter = Interpreter(pdl_interp_module)
        implementations = EqsatPDLInterpFunctions(ctx)
        implementations.populate_known_ops(op)
        implementations.initialize_reachable_rules(rewriters_module)
        interpreter.register_implementations(implementations)
        rewrite_pattern = PDLInterpRewritePattern(matcher, interpreter, implementations)

        listener = PatternRewriterListener()
        listener.operation_modification_handler.append(
            implementations.modification_handler
        )
        walker = PatternRewriteWalker(rewrite_pattern, apply_recursively=False)
        walker.listener = listener

        for _i in range(self.max_iterations):
            # Register matches by walking the module
            walker.rewrite_module(op)
            for eclass in op.walk():
                if isinstance(eclass, EClassOp):
                    op.verify()

            if not implementations.merge_list:
                break

            implementations.apply_matches()
            for eclass in op.walk():
                if isinstance(eclass, EClassOp):
                    op.verify()
