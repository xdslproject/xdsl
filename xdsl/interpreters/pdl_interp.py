from dataclasses import dataclass, field
from typing import Any, cast

from xdsl.context import Context
from xdsl.dialects import pdl_interp
from xdsl.dialects.builtin import StringAttr
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    ReturnedValues,
    Successor,
    impl,
    impl_callable,
    impl_terminator,
    register_impls,
)
from xdsl.ir import Attribute, Operation, OpResult, SSAValue, TypeAttribute
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import InterpretationError


@register_impls
@dataclass
class PDLInterpFunctions(InterpreterFunctions):
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
        assert len(args) > 0
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
        assert len(args) > 0
        assert isinstance(args[0], Operation)
        return (args[0].results[op.index.value.data],)

    @impl(pdl_interp.GetAttributeOp)
    def run_getattribute(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetAttributeOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) > 0
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
        assert len(args) > 0
        assert isinstance(args[0], SSAValue)
        assert len(args) == 1, "TODO: Implement this"
        return (args[0].type,)

    @impl(pdl_interp.GetDefiningOpOp)
    def run_getdefiningop(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetDefiningOpOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) > 0
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
    def run_checkoperationname(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CheckOperationNameOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) > 0
        assert isinstance(args[0], Operation)
        cond = args[0].name == op.operation_name.data
        successor = op.true_dest if cond else op.false_dest
        return Successor(successor, ()), ()

    @impl_terminator(pdl_interp.CheckOperandCountOp)
    def run_checkoperandcount(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CheckOperandCountOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) > 0
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
    def run_checkresultcount(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CheckResultCountOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) > 0
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
        assert len(args) > 0
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
        assert len(args) > 0
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
        assert len(args) >= 2
        # Compare the two values for equality
        cond = args[0] == args[1]
        successor = op.true_dest if cond else op.false_dest
        return Successor(successor, ()), ()

    @impl(pdl_interp.ReplaceOp)
    def run_replace(
        self,
        interpreter: Interpreter,
        op: pdl_interp.ReplaceOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) >= 1
        input_op = args[0]
        assert isinstance(input_op, Operation)

        # Get replacement values (if any)
        repl_values: list[SSAValue] = list(args[1:]) if len(args) > 1 else []
        for val in repl_values:
            assert isinstance(val, SSAValue)

        assert len(input_op.results) == len(repl_values), (
            "Number of replacement values should match number of results"
        )

        # Replace the operation with the replacement values
        self.rewriter.replace_op(input_op, new_ops=[], new_results=repl_values)
        return ()

    @impl(pdl_interp.CreateAttributeOp)
    def run_createattribute(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CreateAttributeOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        value = args[0]
        assert isinstance(value, Attribute)
        # Simply return the attribute value
        return (value,)

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

    @impl_callable(pdl_interp.FuncOp)
    def call_func(
        self, interpreter: Interpreter, op: pdl_interp.FuncOp, args: tuple[Any, ...]
    ):
        if op.sym_name.data == "matcher":
            assert self._rewriter is None
            assert len(args) == 1
            root_op = args[0]
            assert isinstance(root_op, Operation)
            self.rewriter = PatternRewriter(root_op)
        else:
            assert self.rewriter is not None

        return interpreter.run_ssacfg_region(op.body, args, op.sym_name.data)

    @impl_terminator(pdl_interp.RecordMatchOp)
    def run_recordmatch(
        self,
        interpreter: Interpreter,
        op: pdl_interp.RecordMatchOp,
        args: tuple[Any, ...],
    ):
        assert self.rewriter is not None
        # TODO properly fix nested symbolcallref lookup
        interpreter.call_op(op.rewriter.nested_references.data[-1].data, args)
        return Successor(op.dest, ()), ()

    @impl_terminator(pdl_interp.FinalizeOp)
    def run_finalize(
        self, interpreter: Interpreter, op: pdl_interp.FinalizeOp, args: tuple[Any, ...]
    ):
        return ReturnedValues(()), ()
