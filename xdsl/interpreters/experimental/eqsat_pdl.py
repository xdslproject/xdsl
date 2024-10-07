from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO, Any

from xdsl.context import MLContext
from xdsl.dialects import eqsat, pdl
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.experimental.pdl import PDLMatcher
from xdsl.ir import Attribute, Operation, OpResult, SSAValue, TypeAttribute
from xdsl.irdl import IRDLOperation
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import InterpretationError


@dataclass
class EqsatPDLMatcher(PDLMatcher):
    value_to_eclass: dict[SSAValue, eqsat.EClassOp] = field(default_factory=dict)

    def match_operand(
        self, ssa_val: SSAValue, pdl_op: pdl.OperandOp, xdsl_val: SSAValue
    ):
        owner = xdsl_val.owner
        assert isinstance(owner, eqsat.EClassOp)
        assert (
            len(owner.operands) == 1
        ), "newly converted eqsat always has 1 element in eclass"
        arg = owner.operands[0]
        res = super().match_operand(ssa_val, pdl_op, arg)
        self.value_to_eclass[arg] = owner
        return res

        # res = super().match_operand(ssa_val, pdl_op, xdsl_val)
        # uses = xdsl_val.uses
        # assert len(uses) == 1, f"Eclass representation of code, uses: {uses}"
        # only_use = next(iter(uses))
        # assert isinstance(only_use.operation, eqsat.EClassOp)
        # self.value_to_eclass[ssa_val] = only_use.operation
        # return res


@dataclass
class EqsatPDLRewritePattern(RewritePattern):
    functions: EqsatPDLRewriteFunctions
    pdl_rewrite_op: pdl.RewriteOp
    interpreter: Interpreter

    def __init__(
        self, pdl_rewrite_op: pdl.RewriteOp, ctx: MLContext, file: IO[str] | None = None
    ):
        pdl_pattern = pdl_rewrite_op.parent_op()
        assert isinstance(pdl_pattern, pdl.PatternOp)
        pdl_module = pdl_pattern.parent_op()
        assert isinstance(pdl_module, ModuleOp)
        self.functions = EqsatPDLRewriteFunctions(ctx)
        self.interpreter = Interpreter(pdl_module, file=file)
        self.interpreter.register_implementations(self.functions)
        self.pdl_rewrite_op = pdl_rewrite_op

    def match_and_rewrite(self, xdsl_op: Operation, rewriter: PatternRewriter) -> None:
        pdl_op_val = self.pdl_rewrite_op.root
        assert pdl_op_val is not None, "TODO: handle None root op in pdl.RewriteOp"
        assert (
            self.pdl_rewrite_op.body is not None
        ), "TODO: handle None body op in pdl.RewriteOp"

        assert isinstance(pdl_op_val, OpResult)
        pdl_op = pdl_op_val.op

        assert isinstance(pdl_op, pdl.OperationOp)
        matcher = EqsatPDLMatcher()
        if not matcher.match_operation(pdl_op_val, pdl_op, xdsl_op):
            return

        parent = self.pdl_rewrite_op.parent_op()
        assert isinstance(parent, pdl.PatternOp)
        for constraint_op in parent.walk():
            if isinstance(constraint_op, pdl.ApplyNativeConstraintOp):
                if not matcher.check_native_constraints(constraint_op):
                    return

        self.interpreter.push_scope("rewrite")
        self.interpreter.set_values(matcher.matching_context.items())
        self.functions.rewriter = rewriter
        self.functions.value_to_eclass = matcher.value_to_eclass

        self.interpreter.run_ssacfg_region(self.pdl_rewrite_op.body, ())

        self.interpreter.pop_scope()
        # Reset values just in case
        self.functions.value_to_eclass = {}


@register_impls
@dataclass
class EqsatPDLRewriteFunctions(InterpreterFunctions):
    """
    The implementations in this class are for the RHS of the rewrite. The SSA values
    referenced within the rewrite block are guaranteed to have been matched with the
    corresponding IR elements. The interpreter context stores the IR elements by SSA
    values.
    """

    ctx: MLContext
    _rewriter: PatternRewriter | None = field(default=None)
    value_to_eclass: dict[SSAValue, eqsat.EClassOp] = field(default_factory=dict)

    @property
    def rewriter(self) -> PatternRewriter:
        assert self._rewriter is not None
        return self._rewriter

    @rewriter.setter
    def rewriter(self, rewriter: PatternRewriter):
        self._rewriter = rewriter

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

        # How to deal with operandSegmentSizes?
        # operand_values, attribute_values, type_values = args

        operand_values = interpreter.get_values(op.operand_values)
        for operand in operand_values:
            assert isinstance(operand, SSAValue)
        eclass_operand_values = tuple(
            self.value_to_eclass[operand].result for operand in operand_values
        )

        attribute_values = interpreter.get_values(op.attribute_values)

        for attribute in attribute_values:
            assert isinstance(attribute, Attribute)

        type_values = interpreter.get_values(op.type_values)

        for type_value in type_values:
            assert isinstance(type_value, TypeAttribute)

        attributes = dict[str, Attribute]()
        properties = dict[str, Attribute]()

        # If the op is an IRDL-defined operation, get the property names.
        if issubclass(op_type, IRDLOperation):
            property_names = op_type.get_irdl_definition().properties.keys()
        else:
            property_names = []

        # Move the attributes to the attribute or property dictionary
        # depending on whether they are a properties or not.
        for attribute_name, attribute_value in zip(
            attribute_value_names, attribute_values
        ):
            if attribute_name in property_names:
                properties[attribute_name] = attribute_value
            else:
                attributes[attribute_name] = attribute_value

        result_op = op_type.create(
            operands=eclass_operand_values,
            result_types=type_values,
            attributes=attributes,
            properties=properties,
        )

        return (result_op,)

    @impl(pdl.ResultOp)
    def run_result(
        self, interpreter: Interpreter, op: pdl.ResultOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        (parent,) = args
        assert isinstance(parent, Operation)
        return (parent.results[op.index.value.data],)

    @impl(pdl.AttributeOp)
    def run_attribute(
        self, interpreter: Interpreter, op: pdl.AttributeOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.value, Attribute)
        return (op.value,)

    @impl(pdl.ReplaceOp)
    def run_replace(
        self, interpreter: Interpreter, op: pdl.ReplaceOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        rewriter = self.rewriter

        (old,) = interpreter.get_values((op.op_value,))
        assert isinstance(old, Operation)

        assert not op.repl_values

        if op.repl_operation is not None:
            (new_op,) = interpreter.get_values((op.repl_operation,))
            rewriter.insert_op(new_op, InsertPoint.before(old))
            # Add new op to eclass
            eclass_op = self.value_to_eclass[old.results[0]]
            eclass_op.operands = eclass_op.arguments + (new_op.res,)
        elif op.repl_values:
            assert False, "Not implemented"
        else:
            assert False, "Unexpected ReplaceOp"

        return ()
