import os
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, pdl_interp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.eqsat_pdl_interp import EqsatPDLInterpFunctions
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriteWalker
from xdsl.transforms.apply_pdl_interp import PDLInterpRewritePattern


@dataclass(frozen=True)
class ApplyEqsatPDLInterpPass(ModulePass):
    name = "apply-eqsat-pdl-interp"

    pdl_interp_file: str | None = None

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        if self.pdl_interp_file is not None:
            assert os.path.exists(self.pdl_interp_file)
            with open(self.pdl_interp_file) as f:
                pdl_interp_module_str = f.read()
                parser = Parser(ctx, pdl_interp_module_str)
                pdl_interp_module = parser.parse_module()
        else:
            pdl_interp_module = op
        matcher = None
        for cur in pdl_interp_module.walk():
            if isinstance(cur, pdl_interp.FuncOp):
                if cur.sym_name.data == "matcher":
                    matcher = cur
                    break
        assert matcher is not None, "matcher function not found"
        interpreter = Interpreter(op)
        implementations = EqsatPDLInterpFunctions(ctx)
        implementations.populate_known_ops(op)
        interpreter.register_implementations(implementations)
        rewrite_pattern = PDLInterpRewritePattern(matcher, interpreter, implementations)
        PatternRewriteWalker(rewrite_pattern).rewrite_module(op)
