from dataclasses import dataclass, field
from typing import Any, cast

from xdsl.context import Context
from xdsl.dialects import pdl_interp
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.pdl import ValueType
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    Successor,
    impl,
    impl_terminator,
    register_impls,
)
from xdsl.ir import Attribute, Operation, OpResult, SSAValue, TypeAttribute
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import InterpretationError


@register_impls
@dataclass
class PDLInterpFunctions(InterpreterFunctions):
    """
    Interpreter functions for the pdl_interp dialect.
    All operations that get a value from the IR will return None if the requested value cannot be determined.
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

    def clear_rewriter(self):
        self._rewriter = None

    @impl(pdl_interp.GetOperandOp)
    def run_getoperand(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetOperandOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        assert isinstance(args[0], Operation)
        if op.index.value.data >= len(args[0].operands):
            return (None,)
        else:
            return (args[0].operands[op.index.value.data],)

    @impl(pdl_interp.GetResultOp)
    def run_getresult(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetResultOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        assert isinstance(args[0], Operation)
        if len(args[0].results) <= op.index.value.data:
            return (None,)
        return (args[0].results[op.index.value.data],)

    @impl(pdl_interp.GetResultsOp)
    def run_getresults(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetResultsOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        assert isinstance(args[0], Operation)
        src_op = args[0]
        assert op.index is None, (
            "TODO: No support yet for getting a specific result group"
        )
        if isinstance(op.result_types[0], ValueType) and len(src_op.results) != 1:
            return (None,)
        return (src_op.results,)

    @impl(pdl_interp.GetAttributeOp)
    def run_getattribute(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetAttributeOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        assert isinstance(args[0], Operation)
        attrname = op.constraint_name.data
        if attrname in args[0].attributes:
            return (args[0].attributes[attrname],)
        elif attrname in args[0].properties:
            return (args[0].properties[attrname],)
        else:
            return (None,)

    @impl(pdl_interp.GetValueTypeOp)
    def run_getvaluetype(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetValueTypeOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        assert isinstance(args[0], SSAValue)
        value = cast(SSAValue, args[0])
        return (value.type,)

    @impl(pdl_interp.GetDefiningOpOp)
    def run_getdefiningop(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetDefiningOpOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        if args[0] is None:
            return (None,)
        assert isinstance(args[0], SSAValue)
        if not isinstance(args[0], OpResult):
            return (None,)
        assert isinstance(args[0].owner, Operation), (
            "Cannot get defining op of a Block argument"
        )
        return (args[0].owner,)

    @impl_terminator(pdl_interp.CheckOperationNameOp)
    def run_check_operation_name(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CheckOperationNameOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        assert isinstance(args[0], Operation)
        cond = args[0].name == op.operation_name.data
        successor = op.true_dest if cond else op.false_dest
        return Successor(successor, ()), ()

    @impl_terminator(pdl_interp.CheckOperandCountOp)
    def run_check_operand_count(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CheckOperandCountOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        assert isinstance(args[0], Operation)

        operand_count = len(args[0].operands)
        expected_count = op.count.value.data

        # If compareAtLeast is set, check if operand count is >= expected
        # Otherwise check for exact match
        if "compareAtLeast" in op.properties:
            cond = operand_count >= expected_count
        else:
            cond = operand_count == expected_count

        successor = op.true_dest if cond else op.false_dest
        return Successor(successor, ()), ()

    @impl_terminator(pdl_interp.CheckResultCountOp)
    def run_check_result_count(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CheckResultCountOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        assert isinstance(args[0], Operation)

        result_count = len(args[0].results)
        expected_count = op.count.value.data

        # If compareAtLeast is set, check if result count is >= expected
        # Otherwise check for exact match
        if "compareAtLeast" in op.properties:
            cond = result_count >= expected_count
        else:
            cond = result_count == expected_count

        successor = op.true_dest if cond else op.false_dest
        return Successor(successor, ()), ()

    @impl_terminator(pdl_interp.CheckAttributeOp)
    def run_checkattribute(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CheckAttributeOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        # args[0] should be the attribute value to check
        attribute = args[0]
        # Compare with the constant value from properties
        cond = attribute == op.constantValue

        successor = op.true_dest if cond else op.false_dest
        return Successor(successor, ()), ()

    @impl_terminator(pdl_interp.IsNotNullOp)
    def run_isnotnull(
        self,
        interpreter: Interpreter,
        op: pdl_interp.IsNotNullOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        # Check if the value is not None
        cond = args[0] is not None
        successor = op.true_dest if cond else op.false_dest
        return Successor(successor, ()), ()

    @impl_terminator(pdl_interp.AreEqualOp)
    def run_areequal(
        self,
        interpreter: Interpreter,
        op: pdl_interp.AreEqualOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 2
        # Compare the two values for equality
        cond = args[0] == args[1]
        successor = op.true_dest if cond else op.false_dest
        return Successor(successor, ()), ()

    @impl(pdl_interp.CreateAttributeOp)
    def run_createattribute(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CreateAttributeOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        # Simply return the attribute value
        return (op.value,)

    @impl(pdl_interp.CreateOperationOp)
    def run_createoperation(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CreateOperationOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        # Get operation name
        op_name = op.constraint_name.data
        op_type = self.ctx.get_optional_op(op_name)
        if op_type is None:
            raise InterpretationError(
                f"Could not find op type for name {op_name} in context"
            )

        # Split args into operands, attributes and result types based on operand segments
        operands = list(args[0 : len(op.input_operands)])
        attributes = list(
            args[
                len(op.input_operands) : len(op.input_operands)
                + len(op.input_attributes)
            ]
        )
        result_types = list(args[len(op.input_operands) + len(op.input_attributes) :])

        # Verify all arguments have correct types
        for operand in operands:
            assert isinstance(operand, SSAValue)
        for attr in attributes:
            assert isinstance(attr, Attribute)
        for res_type in result_types:
            assert isinstance(res_type, TypeAttribute)

        # Create attribute dictionary using input_attribute_names
        attr_names: list[str] = [
            cast(StringAttr, name).data for name in op.input_attribute_names.data
        ]
        attr_dict = dict(zip(attr_names, attributes))

        # Create the new operation
        result_op = op_type.create(
            operands=operands,
            result_types=result_types,
            attributes=attr_dict,
        )

        self.rewriter.insert_op_before_matched_op(result_op)

        # Return the created operation
        return (result_op,)
