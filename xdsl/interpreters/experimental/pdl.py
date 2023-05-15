from dataclasses import dataclass, field
from typing import Any

from xdsl.ir import Attribute, MLContext, TypeAttribute, OpResult, Operation, SSAValue
from xdsl.dialects import pdl
from xdsl.dialects.builtin import IntegerAttr, IntegerType, ModuleOp
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    AnonymousRewritePattern,
)
from xdsl.interpreter import Interpreter, InterpreterFunctions, register_impls, impl
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.hints import isa


@dataclass
class PDLMatcher:
    """
    Tracks the xDSL values corresponding to PDL SSA values during
    interpretation. A new instance is created per operation being checked
    against.
    """

    matching_context: dict[SSAValue, Operation | Attribute | SSAValue] = field(
        default_factory=dict
    )
    """
    For each SSAValue that is an OpResult of an operation in the PDL dialect,
    the corresponding xDSL object.
    """

    def match_operand(
        self, ssa_val: SSAValue, pdl_op: pdl.OperandOp, xdsl_val: SSAValue
    ):
        if ssa_val in self.matching_context:
            return True

        if pdl_op.value_type is not None:
            assert isinstance(pdl_op.value_type, OpResult)
            assert isinstance(pdl_op.value_type.op, pdl.TypeOp)

            if not self.match_type(
                pdl_op.value_type, pdl_op.value_type.op, xdsl_val.typ
            ):
                return False

        self.matching_context[ssa_val] = xdsl_val

        return True

    def match_result(
        self, ssa_val: SSAValue, pdl_op: pdl.ResultOp, xdsl_operand: SSAValue
    ):
        if ssa_val in self.matching_context:
            return self.matching_context[ssa_val] == xdsl_operand

        root_pdl_op_value = pdl_op.parent_
        assert isinstance(root_pdl_op_value, OpResult)
        assert isinstance(root_pdl_op_value.op, pdl.OperationOp)

        if not isinstance(xdsl_operand, OpResult):
            return False

        xdsl_op = xdsl_operand.op

        if not self.match_operation(root_pdl_op_value, root_pdl_op_value.op, xdsl_op):
            return False

        original_op = root_pdl_op_value.op

        index = pdl_op.index.value.data

        if len(original_op.results) <= index:
            return False

        self.matching_context[ssa_val] = xdsl_op.results[index]

        return True

    def match_type(self, ssa_val: SSAValue, pdl_op: pdl.TypeOp, xdsl_attr: Attribute):
        if ssa_val in self.matching_context:
            return self.matching_context[ssa_val] == xdsl_attr

        self.matching_context[ssa_val] = xdsl_attr

        return True

    def match_attribute(
        self,
        ssa_val: SSAValue,
        pdl_op: pdl.AttributeOp,
        attr_name: str,
        xdsl_attr: Attribute,
    ):
        if ssa_val in self.matching_context:
            return self.matching_context[ssa_val] == xdsl_attr

        if pdl_op.value is not None:
            if pdl_op.value != xdsl_attr:
                return False

        if pdl_op.value_type is not None:
            assert isinstance(pdl_op.value_type, OpResult)
            assert isinstance(pdl_op.value_type.op, pdl.TypeOp)

            assert isa(
                xdsl_attr, IntegerAttr[IntegerType]
            ), "Only handle integer types for now"

            if not self.match_type(
                pdl_op.value_type, pdl_op.value_type.op, xdsl_attr.typ
            ):
                return False

        self.matching_context[ssa_val] = xdsl_attr

        return True

    def match_operation(
        self, ssa_val: SSAValue, pdl_op: pdl.OperationOp, xdsl_op: Operation
    ) -> bool:
        if ssa_val in self.matching_context:
            return self.matching_context[ssa_val] == xdsl_op

        if pdl_op.opName is not None:
            if xdsl_op.name != pdl_op.opName.data:
                return False

        attribute_value_names = [avn.data for avn in pdl_op.attributeValueNames.data]

        for avn, av in zip(attribute_value_names, pdl_op.attribute_values):
            assert isinstance(av, OpResult)
            assert isinstance(av.op, pdl.AttributeOp)
            if avn not in xdsl_op.attributes:
                return False

            if not self.match_attribute(av, av.op, avn, xdsl_op.attributes[avn]):
                return False

        pdl_operands = pdl_op.operand_values
        xdsl_operands = xdsl_op.operands

        if len(pdl_operands) != len(xdsl_operands):
            return False

        for pdl_operand, xdsl_operand in zip(pdl_operands, xdsl_operands):
            assert isinstance(pdl_operand, OpResult)
            assert isinstance(pdl_operand.op, pdl.OperandOp | pdl.ResultOp)
            if isinstance(pdl_operand.op, pdl.OperandOp):
                if not self.match_operand(pdl_operand, pdl_operand.op, xdsl_operand):
                    return False
            elif isinstance(pdl_operand.op, pdl.ResultOp):
                if not self.match_result(pdl_operand, pdl_operand.op, xdsl_operand):
                    return False

        pdl_results = pdl_op.type_values
        xdsl_results = xdsl_op.results

        if len(pdl_results) != len(xdsl_results):
            return False

        for pdl_result, xdsl_result in zip(pdl_results, xdsl_results):
            assert isinstance(pdl_result, OpResult)
            assert isinstance(pdl_result.op, pdl.TypeOp)
            if not self.match_type(pdl_result, pdl_result.op, xdsl_result.typ):
                return False

        self.matching_context[ssa_val] = xdsl_op

        return True


@register_impls
@dataclass
class PDLFunctions(InterpreterFunctions):
    """
    Applies the PDL pattern to all ops in the input `ModuleOp`. The rewriter
    jumps straight to the last operation in the pattern, which is expected to
    be a rewrite op. It creates an `AnonymousRewriter`, which runs on all
    operations in the ModuleOp. For each operation, it determines whether the
    operation fits the specified pattern and, if so, assigns the xDSL values
    to the corresponding PDL SSA values, and runs the rewrite operations one by
    one. The implementations in this class are for the RHS of the rewrite.
    """

    ctx: MLContext
    module: ModuleOp
    _rewriter: PatternRewriter | None = field(default=None)

    @property
    def rewriter(self) -> PatternRewriter:
        assert self._rewriter is not None
        return self._rewriter

    @rewriter.setter
    def rewriter(self, rewriter: PatternRewriter):
        self._rewriter = rewriter

    @impl(pdl.PatternOp)
    def run_pattern(
        self, interpreter: Interpreter, op: pdl.PatternOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        block = op.body.block

        if block.is_empty:
            raise InterpretationError("No ops in pattern")

        last_op = block.last_op

        if not isinstance(last_op, pdl.RewriteOp):
            raise InterpretationError(
                "Expected pdl.pattern to be terminated by pdl.rewrite"
            )

        for r_op in block.ops:
            if r_op is last_op:
                break
            # in forward pass, the Python value is the SSA value itself
            if len(r_op.results) != 1:
                raise InterpretationError("PDL ops must have one result")
            result = r_op.results[0]
            interpreter.set_values(((result, r_op),))

        interpreter.run(last_op)

        return ()

    @impl(pdl.RewriteOp)
    def run_rewrite(
        self,
        interpreter: Interpreter,
        pdl_rewrite_op: pdl.RewriteOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        input_module = self.module

        def rewrite(xdsl_op: Operation, rewriter: PatternRewriter) -> None:
            pdl_op_val = pdl_rewrite_op.root
            assert pdl_op_val is not None, "TODO: handle None root op in pdl.RewriteOp"
            assert (
                pdl_rewrite_op.body is not None
            ), "TODO: handle None body op in pdl.RewriteOp"

            (pdl_op,) = interpreter.get_values((pdl_op_val,))
            assert isinstance(pdl_op, pdl.OperationOp)
            matcher = PDLMatcher()
            if not matcher.match_operation(pdl_op_val, pdl_op, xdsl_op):
                return

            interpreter.push_scope("rewrite")
            interpreter.set_values(matcher.matching_context.items())
            self.rewriter = rewriter

            for rewrite_impl_op in pdl_rewrite_op.body.ops:
                interpreter.run(rewrite_impl_op)

            interpreter.pop_scope()

        rewriter = AnonymousRewritePattern(rewrite)

        PatternRewriteWalker(rewriter, apply_recursively=False).rewrite_module(
            input_module
        )

        return ()

    @impl(pdl.OperationOp)
    def run_operation(
        self, interpreter: Interpreter, op: pdl.OperationOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert op.opName is not None
        op_name = op.opName.data
        op_type = self.ctx.get_optional_op(op_name)

        if op_type is None:
            raise InterpretationError(
                f"Could not find op type for name {op_name} in context"
            )

        attribute_value_names = [avn.data for avn in op.attributeValueNames.data]

        # How to deal with operand_segment_sizes?
        # operand_values, attribute_values, type_values = args

        operand_values = interpreter.get_values(op.operand_values)
        for operand in operand_values:
            assert isinstance(operand, SSAValue)

        attribute_values = interpreter.get_values(op.attribute_values)

        for attribute in attribute_values:
            assert isinstance(attribute, Attribute)

        type_values = interpreter.get_values(op.type_values)

        for type_value in type_values:
            assert isinstance(type_value, TypeAttribute)

        attributes = dict(zip(attribute_value_names, attribute_values))

        result_op = op_type.create(
            operands=operand_values, result_types=type_values, attributes=attributes
        )

        return (result_op,)

    @impl(pdl.ReplaceOp)
    def run_replace(
        self, interpreter: Interpreter, op: pdl.ReplaceOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        rewriter = self.rewriter

        (old,) = interpreter.get_values((op.op_value,))

        if op.repl_operation is not None:
            (new_op,) = interpreter.get_values((op.repl_operation,))
            rewriter.replace_op(old, new_op)
        elif len(op.repl_values):
            new_vals = interpreter.get_values(op.repl_values)
            rewriter.replace_op(old, new_ops=[], new_results=list(new_vals))
        else:
            assert False, "Unexpected ReplaceOp"

        return ()

    @impl(ModuleOp)
    def run_module(
        self, interpreter: Interpreter, op: ModuleOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        ops = op.ops
        first_op = ops.first
        if first_op is None or not isinstance(first_op, pdl.PatternOp):
            raise InterpretationError("Expected single pattern op in pdl module")
        return self.run_pattern(interpreter, first_op, args)
