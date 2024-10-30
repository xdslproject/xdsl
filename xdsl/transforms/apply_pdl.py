import os
from dataclasses import dataclass
from io import StringIO

from xdsl.context import MLContext
from xdsl.dialects import builtin, pdl
from xdsl.interpreter import Interpreter
from xdsl.interpreters.experimental.pdl import (
    PDLMatcher,
    PDLRewriteFunctions,
)
from xdsl.ir import Operation, Region, SSAValue
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter


@dataclass(frozen=True)
class ApplyPDLPass(ModulePass):
    name = "apply-pdl"

    pdl_file: str | None = None

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        payload_module = op
        # Target the file containing the PDL specification
        if self.pdl_file:
            assert os.path.exists(self.pdl_file)
            with open(self.pdl_file) as f:
                pdl_module_str = f.read()
                parser = Parser(ctx, pdl_module_str)
                pdl_module = parser.parse_module()
        else:
            pdl_module = payload_module
        # Gather all the pattern operations
        patterns = [op for op in pdl_module.walk() if isinstance(op, pdl.PatternOp)]
        # Process each pattern
        for pattern in patterns:
            matcher = PDLMatcher()
            rewrites: list[pdl.RewriteOp] = []
            for pdl_op in pattern.walk():
                # Match all the specified operations in the pattern
                if isinstance(pdl_op, pdl.OperationOp) and not isinstance(
                    pdl_op.parent_op(), pdl.RewriteOp
                ):
                    for payload_op in payload_module.walk():
                        matcher.match_operation(pdl_op.results[0], pdl_op, payload_op)
                    if pdl_op.results[0] not in matcher.matching_context:
                        break
                    for constraint_op in pattern.walk():
                        if isinstance(constraint_op, pdl.ApplyNativeConstraintOp):
                            assert matcher.check_native_constraints(constraint_op)
                # Apply the rewrites
                elif isinstance(pdl_op, pdl.RewriteOp):
                    # PDLRewriteFUnctions = the RHS of the rewrite
                    functions = PDLRewriteFunctions(ctx)
                    assert isinstance(pdl_op.root, SSAValue)
                    payload_op = matcher.matching_context[pdl_op.root]
                    assert isinstance(payload_op, Operation)
                    functions.rewriter = PatternRewriter(payload_op)
                    # The interpreter which performs the actual rewriting
                    stream = StringIO()
                    interpreter = Interpreter(payload_module, file=stream)
                    interpreter.register_implementations(functions)
                    interpreter.push_scope("rewrite")
                    interpreter.set_values(matcher.matching_context.items())
                    assert isinstance(pdl_op.body, Region)
                    interpreter.run_ssacfg_region(pdl_op.body, ())
                    interpreter.pop_scope()
                    rewrites.append(pdl_op)
