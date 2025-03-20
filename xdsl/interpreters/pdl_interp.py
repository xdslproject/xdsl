from dataclasses import dataclass
from typing import Any

from xdsl.context import Context
from xdsl.dialects import pdl_interp
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    Successor,
    impl,
    impl_terminator,
    register_impls,
)
from xdsl.ir import Operation, SSAValue


@register_impls
@dataclass
class PDLInterpFunctions(InterpreterFunctions):
    ctx: Context

    @impl(pdl_interp.GetOperandOp)
    def run_getoperand(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetOperandOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) > 0
        assert isinstance(args[0], Operation)
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
        return (args[0].attributes[op.constraint_name.data],)

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
        assert isinstance(args[0], SSAValue)
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
