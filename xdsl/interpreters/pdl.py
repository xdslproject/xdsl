from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import IO, Any

from xdsl.context import Context
from xdsl.dialects import pdl
from xdsl.dialects.builtin import IntegerAttr, IntegerType, ModuleOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.ir import Attribute, Operation, OpResult, SSAValue, TypeAttribute
from xdsl.irdl import IRDLOperation
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
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
        default_factory=dict[SSAValue, Operation | Attribute | SSAValue]
    )
    """
    For each SSAValue that is an OpResult of an operation in the PDL dialect,
    the corresponding xDSL object.
    """

    native_constraints: dict[str, Callable[..., bool]] = field(
        default_factory=lambda: {}
    )
    """
    The functions that can be used in `pdl.apply_native_constraint`. Note that we do
    not verify that the functions are used with the correct types.
    """

    def get_constant_or_matched_value(
        self, ssa_val: SSAValue
    ) -> Operation | Attribute | SSAValue:
        """
        Get the value that is already matched, or that is defined by a constant such as
        the result of a constant `pdl.attribute`, or of a `pdl.type`, or of a matched
        operation.
        """
        if ssa_val in self.matching_context:
            return self.matching_context[ssa_val]
        if isinstance(ssa_val.owner, pdl.AttributeOp):
            if ssa_val.owner.value is None:
                raise InterpretationError("expected constant `pdl.attribute`")
            return ssa_val.owner.value
        if isinstance(ssa_val.owner, pdl.TypeOp):
            if ssa_val.owner.constantType is None:
                raise InterpretationError("expected constant `pdl.type`")
            return ssa_val.owner.constantType
        raise InterpretationError("expected constant or matched value")

    def match_operand(
        self, ssa_val: SSAValue, pdl_op: pdl.OperandOp, xdsl_val: SSAValue
    ):
        if ssa_val in self.matching_context:
            return self.matching_context[ssa_val] == xdsl_val

        if pdl_op.value_type is not None:
            assert isinstance(pdl_op.value_type, OpResult)
            assert isinstance(pdl_op.value_type.op, pdl.TypeOp)

            if not self.match_type(
                pdl_op.value_type, pdl_op.value_type.op, xdsl_val.type
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

        if original_op.type_values and len(original_op.type_values) <= index:
            return False

        self.matching_context[ssa_val] = xdsl_op.results[index]

        return True

    def match_type(self, ssa_val: SSAValue, pdl_op: pdl.TypeOp, xdsl_attr: Attribute):
        if ssa_val in self.matching_context:
            return self.matching_context[ssa_val] == xdsl_attr

        if pdl_op.constantType is None or pdl_op.constantType == xdsl_attr:
            self.matching_context[ssa_val] = xdsl_attr
            return True
        else:
            return False

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

            assert isa(xdsl_attr, IntegerAttr[IntegerType]), (
                "Only handle integer types for now"
            )

            if not self.match_type(
                pdl_op.value_type, pdl_op.value_type.op, xdsl_attr.type
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
            if (attr := xdsl_op.get_attr_or_prop(avn)) is None:
                return False

            if not self.match_attribute(av, av.op, avn, attr):
                return False

        pdl_operands = pdl_op.operand_values
        xdsl_operands = xdsl_op.operands

        if len(pdl_operands) != len(xdsl_operands):
            return False

        for pdl_operand, xdsl_operand in zip(pdl_operands, xdsl_operands):
            assert isinstance(pdl_operand, OpResult)
            assert isinstance(pdl_operand.op, pdl.OperandOp | pdl.ResultOp)
            match pdl_operand.op:
                case pdl.OperandOp():
                    if not self.match_operand(
                        pdl_operand, pdl_operand.op, xdsl_operand
                    ):
                        return False
                case pdl.ResultOp():
                    if not self.match_result(pdl_operand, pdl_operand.op, xdsl_operand):
                        return False

        pdl_results = pdl_op.type_values
        xdsl_results = xdsl_op.results

        if len(pdl_results) != len(xdsl_results):
            return False

        for pdl_result, xdsl_result in zip(pdl_results, xdsl_results):
            assert isinstance(pdl_result, OpResult)
            assert isinstance(pdl_result.op, pdl.TypeOp)
            if not self.match_type(pdl_result, pdl_result.op, xdsl_result.type):
                return False

        self.matching_context[ssa_val] = xdsl_op

        return True

    def check_native_constraints(self, pdl_op: pdl.ApplyNativeConstraintOp) -> bool:
        args = [
            self.get_constant_or_matched_value(operand) for operand in pdl_op.operands
        ]
        name = pdl_op.constraint_name.data
        if name not in self.native_constraints:
            raise InterpretationError(f"{name} PDL native constraint is not registered")
        return self.native_constraints[name](*args)


@dataclass
class PDLRewritePattern(RewritePattern):
    functions: PDLRewriteFunctions
    pdl_rewrite_op: pdl.RewriteOp
    interpreter: Interpreter
    native_constraints: dict[str, Callable[..., bool]]

    def __init__(
        self,
        pdl_rewrite_op: pdl.RewriteOp,
        ctx: Context,
        file: IO[str] | None = None,
        native_constraints: dict[str, Callable[..., bool]] | None = None,
    ):
        pdl_pattern = pdl_rewrite_op.parent_op()
        assert isinstance(pdl_pattern, pdl.PatternOp)
        pdl_module = pdl_pattern.parent_op()
        assert isinstance(pdl_module, ModuleOp)
        self.functions = PDLRewriteFunctions(ctx)
        self.interpreter = Interpreter(pdl_module, file=file)
        self.interpreter.register_implementations(self.functions)
        self.pdl_rewrite_op = pdl_rewrite_op
        if native_constraints is None:
            native_constraints = {}
        self.native_constraints = native_constraints

    def match_and_rewrite(self, xdsl_op: Operation, rewriter: PatternRewriter) -> None:
        pdl_op_val = self.pdl_rewrite_op.root
        assert pdl_op_val is not None, "TODO: handle None root op in pdl.RewriteOp"
        assert self.pdl_rewrite_op.body is not None, (
            "TODO: handle None body op in pdl.RewriteOp"
        )

        assert isinstance(pdl_op_val, OpResult)
        pdl_op = pdl_op_val.op

        assert isinstance(pdl_op, pdl.OperationOp)
        matcher = PDLMatcher(native_constraints=self.native_constraints)
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

        self.interpreter.run_ssacfg_region(self.pdl_rewrite_op.body, ())

        self.interpreter.pop_scope()


@register_impls
@dataclass
class PDLRewriteFunctions(InterpreterFunctions):
    """
    The implementations in this class are for the RHS of the rewrite. The SSA values
    referenced within the rewrite block are guaranteed to have been matched with the
    corresponding IR elements. The interpreter context stores the IR elements by SSA
    values.
    """

    ctx: Context
    _rewriter: PatternRewriter | None = field(default=None)

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
            operands=operand_values,
            result_types=type_values,
            attributes=attributes,
            properties=properties,
        )

        # Insert the new operation before the root operation
        self.rewriter.insert_op(result_op)

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

        if op.repl_operation is not None:
            (new_op,) = interpreter.get_values((op.repl_operation,))
            rewriter.replace_op(old, new_ops=[], new_results=new_op.results)
        elif len(op.repl_values):
            new_vals = interpreter.get_values(op.repl_values)
            rewriter.replace_op(old, new_ops=[], new_results=list(new_vals))
        else:
            raise ValueError(
                "Either a replacing operatoin or values must be provided with replace op"
            )

        return ()

    @impl(pdl.EraseOp)
    def run_erase(
        self, interpreter: Interpreter, op: pdl.EraseOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        (old,) = interpreter.get_values((op.op_value,))
        self.rewriter.erase_op(old)
        return ()

    @impl(pdl.TypeOp)
    def run_type(
        self, interpreter: Interpreter, op: pdl.TypeOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.constantType, Attribute)
        return (op.constantType,)
