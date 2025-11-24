from typing import Any, cast

from xdsl.context import Context
from xdsl.dialects import pdl_interp
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.pdl import RangeType, ValueType
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    Successor,
    impl,
    impl_callable,
    impl_terminator,
    register_impls,
)
from xdsl.ir import Attribute, Operation, OpResult, SSAValue
from xdsl.irdl import IRDLOperation
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.hints import isa


@register_impls
class PDLInterpFunctions(InterpreterFunctions):
    """
    Interpreter functions for the pdl_interp dialect.
    All operations that get a value from the IR will return None if the requested value
    cannot be determined.

    Native constraints are implemented via `impl_external`. Clients that want to provide
    extra constraint declarations must do so with a similar extra `InterpreterFunctions`
    subclass registered with the interpreter:

    ```py
    @register_impls
    class TestImplFunctions(InterpreterFunctions):
        @impl_external("test_constraint")
        def run_test_constraint(
            self, interp: Interpreter, op: Operation, args: PythonValues
        ) -> tuple[bool, PythonValues]:
            assert isinstance(args[0], int)
            (x,) = args
            return True, (x + 42,)
    ```

    Note that the return type of a native constraint must be `tuple[bool, PythonValues]`.
    """

    @staticmethod
    def get_ctx(interpreter: Interpreter) -> Context:
        return interpreter.get_data(
            PDLInterpFunctions,
            "ctx",
            lambda: Context(),
        )

    @staticmethod
    def set_ctx(interpreter: Interpreter, ctx: Context) -> None:
        interpreter.set_data(
            PDLInterpFunctions,
            "ctx",
            ctx,
        )

    @staticmethod
    def get_rewriter(interpreter: Interpreter) -> PatternRewriter:
        rewriter: PatternRewriter | None = interpreter.get_data(
            PDLInterpFunctions, "rewriter", lambda: None
        )
        if rewriter is None:
            raise InterpretationError(
                "Expected an active rewriter when calling a pdl_interp function."
            )
        return rewriter

    @staticmethod
    def set_rewriter(
        interpreter: Interpreter, rewriter: PatternRewriter | None
    ) -> None:
        interpreter.set_data(PDLInterpFunctions, "rewriter", rewriter)

    @impl(pdl_interp.GetOperandOp)
    def run_get_operand(
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
        assert isinstance(src_op, IRDLOperation)
        if op.index is not None:
            # get the field name of the result group:
            if op.index.value.data >= len(src_op.get_irdl_definition().results):
                return (None,)
            field = src_op.get_irdl_definition().results[op.index.value.data][0]
            results = getattr(src_op, field)
            if isa(results, OpResult):
                results = (results,)
        else:
            results = src_op.results

        if isinstance(op.result_types[0], ValueType):
            if len(results) != 1:
                return (None,)
            return (results[0],)
        return (results,)

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

    @impl_terminator(pdl_interp.SwitchOperationNameOp)
    def run_switch_operation_name(
        self,
        interpreter: Interpreter,
        op: pdl_interp.SwitchOperationNameOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        input_op: Operation = args[0]
        op_name = input_op.name
        for name, block in zip(op.case_values, op.cases):
            if name.data == op_name:
                return Successor(block, ()), ()
        return Successor(op.default_dest, ()), ()

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

    @impl_terminator(pdl_interp.CheckTypeOp)
    def run_check_type(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CheckTypeOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        value = args[0]
        cond = value == op.type

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
        self.get_rewriter(interpreter).replace_op(
            input_op, new_ops=[], new_results=repl_values
        )
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

    @impl(pdl_interp.CreateTypeOp)
    def run_create_type(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CreateTypeOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        # Simply return the type value
        return (op.value,)

    @impl(pdl_interp.CreateTypesOp)
    def run_create_types(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CreateTypesOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        # Return the list of types from the array attribute
        types = list(op.value.data)
        return (types,)

    @impl_terminator(pdl_interp.SwitchAttributeOp)
    def run_switch_attribute(
        self,
        interpreter: Interpreter,
        op: pdl_interp.SwitchAttributeOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        input_attr = args[0]
        for case_value, block in zip(op.caseValues.data, op.cases):
            if input_attr == case_value:
                return Successor(block, ()), ()
        return Successor(op.defaultDest, ()), ()

    @impl(pdl_interp.CreateOperationOp)
    def run_create_operation(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CreateOperationOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        # Get operation name
        op_name = op.constraint_name.data
        ctx = self.get_ctx(interpreter)
        op_type = ctx.get_optional_op(op_name)
        if op_type is None:
            raise InterpretationError(
                f"Could not find op type for name {op_name} in context"
            )

        # Split args into operands, attributes and result types based on operand segments
        operands = list(args[0 : len(op.input_operands)])

        attr_names: list[str] = [
            cast(StringAttr, name).data for name in op.input_attribute_names.data
        ]

        assert issubclass(op_type, IRDLOperation)
        existing_properties = op_type.get_irdl_definition().properties.keys()

        attributes: dict[str, Attribute] = {}
        properties: dict[str, Attribute] = {}
        for name, prop_or_attr in zip(
            attr_names,
            args[
                len(op.input_operands) : len(op.input_operands)
                + len(op.input_attributes)
            ],
        ):
            if name in existing_properties:
                properties[name] = prop_or_attr
            else:
                attributes[name] = prop_or_attr
        result_types = list(args[len(op.input_operands) + len(op.input_attributes) :])

        # Create the new operation
        result_op = op_type.create(
            operands=operands,
            result_types=result_types,
            attributes=attributes,
            properties=properties,
        )

        rewriter = self.get_rewriter(interpreter)
        rewriter.insert_op(result_op)

        return (result_op,)

    @impl_callable(pdl_interp.FuncOp)
    def call_func(
        self, interpreter: Interpreter, op: pdl_interp.FuncOp, args: tuple[Any, ...]
    ):
        if op.sym_name.data == "matcher":
            assert len(args) == 1
            root_op = args[0]
            assert isinstance(root_op, Operation)
            rewriter = self.get_rewriter(interpreter)
            rewriter.current_operation = root_op
            rewriter.insertion_point = InsertPoint.before(root_op)

        return interpreter.run_ssacfg_region(op.body, args, op.sym_name.data)

    @impl_terminator(pdl_interp.ApplyConstraintOp)
    def run_apply_constraint(
        self,
        interpreter: Interpreter,
        op: pdl_interp.ApplyConstraintOp,
        args: tuple[Any, ...],
    ) -> tuple[Successor, PythonValues]:
        assert len(args) == 1
        constraint_name = op.constraint_name.data

        passed, results = interpreter.call_external(constraint_name, op, args)
        assert isinstance(passed, bool)
        assert isinstance(results, tuple)
        any_results = cast(tuple[Any], results)

        terminator = op.true_dest if passed != bool(op.is_negated) else op.false_dest

        return Successor(terminator, ()), any_results

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
        PDLInterpFunctions.set_rewriter(interpreter, None)
        return ReturnedValues(()), ()
