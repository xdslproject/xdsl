"""
A pass that applies the interpreter to operations with no side effects where all the
inputs are constant, replacing the computation with a constant value.
"""

from dataclasses import dataclass
from typing import Any, cast

from xdsl.context import Context
from xdsl.dialect_interfaces.constant_materialization import (
    ConstantMaterializationInterface,
)
from xdsl.dialects import builtin
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.interpreter import Interpreter
from xdsl.interpreters import register_implementations
from xdsl.ir import Attribute, Operation, OpResult
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
    ctx: Context
    interpreter: Interpreter

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not op.has_trait(Pure):
            # Only rewrite operations that don't have side-effects
            return

        # No need to rewrite operations that are already constant-like
        if op.has_trait(ConstantLike):
            return

        if not all(
            isinstance(operand, OpResult) and operand.op.has_trait(ConstantLike)
            for operand in op.operands
        ):
            # Only rewrite operations where all the operands are constants
            return

        dialect = self.ctx.get_dialect(op.dialect_name())

        if (
            materializer := dialect.get_interface(ConstantMaterializationInterface)
        ) is None:
            return

        try:
            args = tuple(
                self.interpreter.run_op(cast(OpResult, operand).op, ())[0]
                for operand in op.operands
            )
            results = self.interpreter.run_op(op, args)
        except InterpretationError:
            return

        new_ops: list[Operation] = []
        for interp_result, op_result in zip(results, op.results):
            result_attr = self.convert_to_attr(interp_result, op_result.type)
            if result_attr is None:
                return
            new_op = materializer.materialize_constant(result_attr, op_result.type)
            if new_op is None:
                return
            new_ops.append(new_op)

        rewriter.replace_op(op, new_ops, [new_op.results[0] for new_op in new_ops])

    def convert_to_attr(self, value: Any, value_type: Attribute) -> Attribute | None:
        match (value, value_type):
            case int(), IntegerType():
                return IntegerAttr(value, cast(IntegerType, value_type))
            case _:
                return None


class ConstantFoldInterpPass(ModulePass):
    """
    A pass that applies the interpreter to operations with no side effects where all the
    inputs are constant, replacing the computation with a constant value.
    """

    name = "constant-fold-interp"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        interpreter = Interpreter(op)
        register_implementations(interpreter, ctx)
        pattern = ConstantFoldInterpPattern(ctx, interpreter)
        PatternRewriteWalker(pattern).rewrite_module(op)
