from dataclasses import dataclass, field
from typing import Any, cast

from xdsl.context import Context
from xdsl.dialects import pdl_interp
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.pdl import RangeType, ValueType
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
from xdsl.utils.hints import isa


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
    def run_get_result(
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
    def run_get_results(
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
    def run_get_attribute(
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
    def run_get_value_type(
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
    def run_get_defining_op(
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
    def run_check_attribute(
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
    def run_is_not_null(
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
    def run_are_equal(
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

    @impl(pdl_interp.ReplaceOp)
    def run_replace(
        self,
        interpreter: Interpreter,
        op: pdl_interp.ReplaceOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert args
        input_op = args[0]
        assert isinstance(input_op, Operation)

        # Get replacement values (if any)
        repl_values: list[SSAValue] = []
        for i in range(0, len(args) - 1):
            if isa(op.repl_values.types[i], ValueType):
                repl_values.append(args[i + 1])
            elif isa(op.repl_values.types[i], RangeType[ValueType]):
                repl_values.extend(args[i + 1])

        if len(input_op.results) != len(repl_values):
            raise InterpretationError(
                "Number of replacement values should match number of results"
            )
        # Replace the operation with the replacement values
        self.rewriter.replace_op(input_op, new_ops=[], new_results=repl_values)
        return ()

    @impl(pdl_interp.CreateAttributeOp)
    def run_create_attribute(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CreateAttributeOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        # Simply return the attribute value
        return (op.value,)

    @impl(pdl_interp.CreateOperationOp)
    def run_create_operation(
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
        if self._rewriter is None:
            raise InterpretationError(
                "Expected an active rewriter when calling a pdl_interp function."
            )
        if op.sym_name.data == "matcher":
            assert len(args) == 1
            root_op = args[0]
            assert isinstance(root_op, Operation)
            self.rewriter.current_operation = root_op

        return interpreter.run_ssacfg_region(op.body, args, op.sym_name.data)

    @impl_terminator(pdl_interp.RecordMatchOp)
    def run_recordmatch(
        self,
        interpreter: Interpreter,
        op: pdl_interp.RecordMatchOp,
        args: tuple[Any, ...],
    ):
        interpreter.call_op(op.rewriter, args)
        return Successor(op.dest, ()), ()

    @impl_terminator(pdl_interp.FinalizeOp)
    def run_finalize(
        self, interpreter: Interpreter, op: pdl_interp.FinalizeOp, args: tuple[Any, ...]
    ):
        self._rewriter = None
        return ReturnedValues(()), ()
