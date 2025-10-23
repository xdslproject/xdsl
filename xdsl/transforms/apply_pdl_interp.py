import os
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, pdl_interp
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Operation
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern


@dataclass
class PDLInterpRewritePattern(RewritePattern):
    """
    A rewrite pattern that uses the pdl_interp dialect for matching and rewriting operations.
    """

    interpreter: Interpreter
    functions: PDLInterpFunctions
    matcher: pdl_interp.FuncOp
    name: None | str = None

    def __init__(
        self,
        matcher: pdl_interp.FuncOp,
        interpreter: Interpreter,
        functions: PDLInterpFunctions,
        name: None | str = None,
    ):
        self.functions = functions
        module = matcher.parent_op()
        assert isinstance(module, ModuleOp)
        self.interpreter = interpreter
        if matcher.sym_name.data != "matcher":
            raise ValueError("Matcher function name must be 'matcher'")
        self.matcher = matcher
        self.name = name

    def match_and_rewrite(self, xdsl_op: Operation, rewriter: PatternRewriter) -> None:
        # Setup the rewriter
        self.functions.set_rewriter(self.interpreter, rewriter)

        # Call the matcher function on the operation
        self.interpreter.call_op(self.matcher, (xdsl_op,))


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
            if isinstance(cur, pdl_interp.FuncOp) and cur.sym_name.data == "matcher":
                matcher = cur
                break
        assert matcher is not None, "matcher function not found"
        interpreter = Interpreter(pdl_interp_module)
        implementations = PDLInterpFunctions()
        PDLInterpFunctions.set_ctx(interpreter, ctx)
        interpreter.register_implementations(implementations)
        rewrite_pattern = PDLInterpRewritePattern(matcher, interpreter, implementations)
        PatternRewriteWalker(rewrite_pattern).rewrite_module(op)
