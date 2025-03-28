import os
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, pdl_interp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.parser import Parser
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class ApplyPDLInterpPass(ModulePass):
    name = "apply-pdl-interp"

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
        implementations = PDLInterpFunctions(ctx)
        interpreter.register_implementations(implementations)
        for root in op.walk():
            implementations.clear_rewriter()
            if root == op:
                continue  # can't rewrite the module itself
            interpreter.call_op(matcher, (root,))
