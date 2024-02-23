"""
A pass that applies the interpreter to operations with no side effects where all the
inputs are constant, replacing the computation with a constant value.
"""


from dataclasses import dataclass
from typing import Any, cast

from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.interpreter import Interpreter
from xdsl.interpreters import register_implementations
from xdsl.ir import Attribute, MLContext, Operation, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)
from xdsl.traits import ConstantLike, Pure
from xdsl.utils.exceptions import InterpretationError


@dataclass
class ConstantFoldInterpPattern(RewritePattern):
    interpreter: Interpreter

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not op.has_trait(Pure):
            # Only rewrite operations that don't have side-effects
            return

        if not all(
            isinstance(operand, OpResult) and operand.op.has_trait(ConstantLike)
            for operand in op.operands
        ):
            # Only rewrite operations where all the operands are constants
            return

        try:
            args = tuple(
                self.interpreter.run_op(cast(OpResult, operand).op, ())[0]
                for operand in op.operands
            )
            results = self.interpreter.run_op(op, args)
        except InterpretationError:
            return

        new_ops = [
            self.constant_op_for_value(interp_result, op_result.type)
            for interp_result, op_result in zip(results, op.results)
        ]

        if any(new_op is None for new_op in new_ops):
            # If we don't know how to create a constant for one of the results, bail
            return

        new_ops = cast(list[Operation], new_ops)

        rewriter.replace_matched_op(new_ops, [new_op.results[0] for new_op in new_ops])

    def constant_op_for_value(
        self, value: Any, value_type: Attribute
    ) -> Operation | None:
        match (value, value_type):
            case int(), IntegerType():
                attr = IntegerAttr(value, value_type)
                return arith.Constant(attr)
            case _:
                return None


class ConstantFoldInterpPass(ModulePass):
    """
    A pass that applies the interpreter to operations with no side effects where all the
    inputs are constant, replacing the computation with a constant value.
    """

    name = "constant-fold-interp"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        interpreter = Interpreter(op)
        # Do not call wgpu interpreter functions for this pass
        register_implementations(interpreter, ctx, include_wgpu=False)
        pattern = ConstantFoldInterpPattern(interpreter)
        PatternRewriteWalker(pattern).rewrite_module(op)
