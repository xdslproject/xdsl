from collections.abc import Sequence
from typing import Any, cast

import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, eqsat, eqsat_pdl_interp, func, pdl, pdl_interp, test
from xdsl.dialects.builtin import ModuleOp, i32, i64
from xdsl.interpreter import Interpreter
from xdsl.interpreters.eqsat_pdl_interp import (
    EqsatPDLInterpFunctions,
    NonPropagatingDataFlowSolver,
)
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
from xdsl.irdl import (
    AttrSizedResultSegments,
    IRDLOperation,
    irdl_op_definition,
    result_def,
    var_result_def,
)
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.test_value import create_ssa_value


def test_populate_known_ops():
    """Test that populate_known_ops correctly categorizes operations."""
    # Create test operations
    regular_op = test.TestOp(result_types=[i32])
    eclass_op = eqsat.EClassOp(create_ssa_value(i32), res_type=i32)

    # Create module containing both types of operations
    module = ModuleOp([regular_op, eclass_op])

    # Create interpreter functions instance
    interp_functions = EqsatPDLInterpFunctions()

    # Call the method under test
    interp_functions.populate_known_ops(module)

    # Assert that regular operations are in known_ops
    assert regular_op in interp_functions.known_ops
    assert interp_functions.known_ops[regular_op] is regular_op

    # Assert that EClassOp is not in known_ops but is in eclass_union_find
    assert eclass_op not in interp_functions.known_ops
    assert eclass_op in interp_functions.eclass_union_find._index_by_value  # pyright: ignore[reportPrivateUsage]


def test_run_get_result():
    """Test that run_get_result handles EClass operations correctly."""
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EqsatPDLInterpFunctions())

    # Create a test operation with results
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32, i64))

    # Create EClass operations that use both results
    eclass_op1 = eqsat.EClassOp(test_op.results[0], res_type=i32)
    eclass_op2 = eqsat.EClassOp(test_op.results[1], res_type=i64)

    # Test GetResultOp for first result wrapped in EClass - should return EClass result
    result = interpreter.run_op(
        pdl_interp.GetResultOp(0, create_ssa_value(pdl.OperationType())), (test_op,)
    )
    assert result == (eclass_op1.results[0],)

    # Test GetResultOp for second result wrapped in EClass - should return EClass result
    result = interpreter.run_op(
        pdl_interp.GetResultOp(1, create_ssa_value(pdl.OperationType())), (test_op,)
    )
    assert result == (eclass_op2.results[0],)


def test_run_get_result_error_case():
    """Test that run_get_result raises error when result is not used by EClass."""
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EqsatPDLInterpFunctions())

    # Create a test operation with result that is not used by EClass
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32,))
    _eclass_user = eqsat.EClassOp(test_op.results[0])
    _extra_user = test.TestOp((test_op.results[0],))

    # Don't create any EClass operations, so result is not used by EClass

    # Test GetResultOp should raise InterpretationError
    with pytest.raises(
        InterpretationError,
        match="pdl_interp.get_result currently only supports operations with results that are used by a single eclass each.",
    ):
        interpreter.run_op(
            pdl_interp.GetResultOp(0, create_ssa_value(pdl.OperationType())), (test_op,)
        )


def test_run_get_results():
    @irdl_op_definition
    class MultiResultGroupsOp(IRDLOperation):
        name = "test.multi_result_group"
        res1 = var_result_def()
        res2 = result_def()
        res3 = var_result_def()

        irdl_options = [AttrSizedResultSegments()]

        def __init__(
            self,
            res1: Sequence[Attribute],
            res2: Attribute,
            res3: Sequence[Attribute],
        ) -> None:
            super().__init__(result_types=[res1, res2, res3])

    # Create an op with 6 results total: 2 in group 0, 1 in group 1, 3 in group 2
    op = MultiResultGroupsOp(
        res1=[i32, i64],
        res2=i32,
        res3=[i64, i32, i64],
    )

    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(
        eqsat_pdl_interp_functions := EqsatPDLInterpFunctions()
    )

    # Create EClass operations that use all results of the op.
    # The GetResultsOp implementation for EqSat requires that the results
    # of the matched operation are consumed by EClass operations.
    eclass_ops = [eqsat.EClassOp(res, res_type=res.type) for res in op.results]

    # Helper to run the op
    def run_get_results(
        index: int | None, type_val: pdl.ValueType | pdl.RangeType[pdl.ValueType]
    ):
        return eqsat_pdl_interp_functions.run_get_results(
            interpreter,
            pdl_interp.GetResultsOp(
                index,
                create_ssa_value(pdl.OperationType()),
                type_val,
            ),
            (op,),
        )[0]

    # Case 1: Get a Variadic Group (index 0) as a Range
    # Corresponds to 'res1' which has 2 results. Returns the EClass results.
    res = run_get_results(0, pdl.RangeType(pdl.ValueType()))
    assert res == (tuple(op.result for op in eclass_ops[0:2]),)

    # Case 2: Get a Single Group (index 1) as a Range
    # Corresponds to 'res2' which has 1 result.
    res = run_get_results(1, pdl.RangeType(pdl.ValueType()))
    assert res == (tuple(op.result for op in eclass_ops[2:3]),)

    # Case 3: Get a Single Group (index 1) as a Value
    # Corresponds to 'res2'. Requesting ValueType extracts the single item.
    res = run_get_results(1, pdl.ValueType())
    assert res == (eclass_ops[2].result,)

    # Case 4: Get a Variadic Group (index 2) as a Range
    # Corresponds to 'res3' which has 3 results.
    res = run_get_results(2, pdl.RangeType(pdl.ValueType()))
    assert res == (tuple(op.result for op in eclass_ops[3:6]),)

    # Case 5: Mismatch - Get Variadic Group expecting a Single Value
    # 'res1' has 2 items, but we ask for ValueType (single). Should return None.
    res = run_get_results(0, pdl.ValueType())
    assert res == (None,)

    # Case 6: Index Out of Bounds
    # Op has groups 0, 1, 2. Requesting 3 should return None.
    res = run_get_results(3, pdl.RangeType(pdl.ValueType()))
    assert res == (None,)

    # Case 7: No Index (Get All Results) as Range
    # Should return all 6 results flattened (as EClasses)
    res = run_get_results(None, pdl.RangeType(pdl.ValueType()))
    assert res == (tuple(op.result for op in eclass_ops),)

    # Case 8: No Index (Get All Results) as Value
    # All results > 1, but we ask for single ValueType. Should return None.
    res = run_get_results(None, pdl.ValueType())
    assert res == (None,)


def test_run_get_results_error_case_multiple_uses():
    """Test that run_get_results raises error when result is not used by a single EClass."""
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EqsatPDLInterpFunctions())

    # Create a test operation with results
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32, i64))

    # Result 0 is used by an EClassOp AND another op.
    _eclass_user = eqsat.EClassOp(test_op.results[0])
    _extra_user = test.TestOp((test_op.results[0],))

    # Test GetResultsOp should raise InterpretationError due to multiple uses
    with pytest.raises(
        InterpretationError,
        match="pdl_interp.get_results only supports results that are used by a single eclass each.",
    ):
        interpreter.run_op(
            pdl_interp.GetResultsOp(
                None,
                create_ssa_value(pdl.OperationType()),
                pdl.RangeType(pdl.ValueType()),
            ),
            (test_op,),
        )


def test_run_get_results_error_case_non_eclass_use():
    """Test that run_get_results raises error when result is not used by a single EClass."""
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EqsatPDLInterpFunctions())

    # Create a test operation with results
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32, i64))

    _non_eclass_user = test.TestOp((test_op.results[0],))

    # Test GetResultsOp should raise InterpretationError due to multiple uses
    with pytest.raises(
        InterpretationError,
        match="pdl_interp.get_results only supports results that are used by a single eclass each.",
    ):
        interpreter.run_op(
            pdl_interp.GetResultsOp(
                None,
                create_ssa_value(pdl.OperationType()),
                pdl.RangeType(pdl.ValueType()),
            ),
            (test_op,),
        )


def test_run_get_result_none_case():
    """Test that run_get_result returns None when result index doesn't exist."""
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EqsatPDLInterpFunctions())

    # Create a test operation with only one result
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32,))

    # Try to get result at index 1 (doesn't exist, only index 0 exists)
    result = interpreter.run_op(
        pdl_interp.GetResultOp(1, create_ssa_value(pdl.OperationType())), (test_op,)
    )
    assert result == (None,)


def test_run_get_results_valuetype_multi_results():
    """Test that run_get_results returns None for ValueType with multiple results."""
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EqsatPDLInterpFunctions())

    # Create a test operation with multiple results
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32, i64))

    # Test GetResultsOp with ValueType (not RangeType) - should return None since op has != 1 result
    result = interpreter.run_op(
        pdl_interp.GetResultsOp(
            None,
            create_ssa_value(pdl.OperationType()),
            pdl.ValueType(),
        ),
        (test_op,),
    )
    assert result == (None,)


def test_run_get_results_valuetype_no_results():
    """Test that run_get_results returns None for ValueType with no results."""
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EqsatPDLInterpFunctions())

    # Create a test operation with no results
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), ())

    # Test GetResultsOp with ValueType (not RangeType) - should return None since op has != 1 result
    result = interpreter.run_op(
        pdl_interp.GetResultsOp(
            None,
            create_ssa_value(pdl.OperationType()),
            pdl.ValueType(),
        ),
        (test_op,),
    )
    assert result == (None,)


def test_run_get_defining_op():
    """Test that run_get_defining_op handles regular operations correctly."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    interpreter.register_implementations(interp_functions)

    # Create a test operation and its result
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32,))

    # Test GetDefiningOpOp with regular operation result
    result = interpreter.run_op(
        pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType())),
        (test_op.results[0],),
    )

    # Should return the defining operation
    assert result == (test_op,)

    # Should not have set up any backtracking for regular operations
    assert len(interp_functions.backtrack_stack) == 0


def test_run_get_defining_op_eclass_not_visited():
    """Test that run_get_defining_op handles EClassOp when not visited."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    interpreter.register_implementations(interp_functions)

    # Create test operations
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32,))
    eclass_op = eqsat.EClassOp(test_op.results[0], res_type=i32)

    # Create GetDefiningOpOp
    gdo_op = pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType()))

    # Set up backtrack stack and visited state
    from xdsl.interpreters.eqsat_pdl_interp import BacktrackPoint
    from xdsl.ir import Block
    from xdsl.utils.scoped_dict import ScopedDict

    block = Block()
    scope = ScopedDict[Any, Any]()
    backtrack_point = BacktrackPoint(block, (), scope, gdo_op, 0, 0)
    interp_functions.backtrack_stack.append(backtrack_point)
    interp_functions.visited = False

    # Test with EClassOp result
    result = interpreter.run_op(gdo_op, (eclass_op.results[0],))

    # Should use index from backtrack stack and set visited to True
    assert interp_functions.visited
    assert result == (test_op,)  # Should return the operand at index 0


def test_run_get_defining_op_eclass_visited():
    """Test that run_get_defining_op handles EClassOp when visited."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    interpreter.register_implementations(interp_functions)

    # Create test operations with multiple operands
    c0 = create_ssa_value(i32)
    c1 = create_ssa_value(i32)
    test_op1 = test.TestOp((c0,), (i32,))
    test_op2 = test.TestOp((c1,), (i32,))
    eclass_op = eqsat.EClassOp(test_op1.results[0], test_op2.results[0], res_type=i32)

    # Create a block and add the GetDefiningOpOp to it
    from xdsl.ir import Block

    block = Block()
    gdo_op = pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType()))
    block.add_op(gdo_op)

    # Set visited to True and create a parent scope for the interpreter context
    interp_functions.visited = True

    # Create a child scope to give the current context a parent
    from xdsl.utils.scoped_dict import ScopedDict

    child_scope = ScopedDict(parent=interpreter._ctx)  # pyright: ignore[reportPrivateUsage]
    interpreter._ctx = child_scope  # pyright: ignore[reportPrivateUsage]

    # Test with EClassOp result
    result = interpreter.run_op(gdo_op, (eclass_op.results[0],))

    # Should create new backtrack point and use index 0
    assert len(interp_functions.backtrack_stack) == 1
    assert interp_functions.backtrack_stack[0].index == 0
    assert interp_functions.backtrack_stack[0].max_index == 1  # len(operands) - 1
    assert result == (test_op1,)  # Should return the operand at index 0


def test_run_get_defining_op_eclass_error_multiple_gdo():
    """Test that run_get_defining_op raises error for multiple get_defining_op in block."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    interpreter.register_implementations(interp_functions)

    # Create test operations
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32,))
    eclass_op = eqsat.EClassOp(test_op.results[0], res_type=i32)

    # Create two different GetDefiningOpOp operations
    gdo_op1 = pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType()))
    gdo_op2 = pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType()))

    # Set up backtrack stack with different gdo_op and visited=False
    from xdsl.interpreters.eqsat_pdl_interp import BacktrackPoint
    from xdsl.ir import Block
    from xdsl.utils.scoped_dict import ScopedDict

    block = Block()
    scope = ScopedDict[Any, Any]()
    backtrack_point = BacktrackPoint(block, (), scope, gdo_op1, 0, 1)  # Different op
    interp_functions.backtrack_stack.append(backtrack_point)
    interp_functions.visited = False

    # Test should raise InterpretationError when using different gdo_op
    with pytest.raises(
        InterpretationError,
        match="Case where a block contains multiple pdl_interp.get_defining_op is currently not supported",
    ):
        interpreter.run_op(gdo_op2, (eclass_op.results[0],))


def test_run_create_operation_new_operation():
    """Test that run_create_operation creates new operation and EClass when no existing match."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    ctx = interp_functions.get_ctx(interpreter)
    ctx.register_dialect("test", lambda: test.Test)
    interpreter.register_implementations(interp_functions)

    # Set up a mock rewriter
    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block, Region
    from xdsl.pattern_rewriter import PatternRewriter

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        root = test.TestOp()
        operand = eqsat.EClassOp(create_ssa_value(i32), res_type=i32).result
    rewriter = PatternRewriter(root)
    interp_functions.set_rewriter(interpreter, rewriter)

    # Create operands and types for the operation
    result_type = i32

    # Create CreateOperationOp
    create_op = pdl_interp.CreateOperationOp(
        name="test.op",
        input_operands=[operand],
        input_attributes=[],
        input_result_types=[create_ssa_value(pdl.TypeType())],
    )
    interp_functions.populate_known_ops(testmodule)

    # Run the create operation
    result = interp_functions.run_create_operation(
        interpreter, create_op, (operand, result_type)
    )

    # Should return the created operation
    assert len(result.values) == 1
    created_op = result.values[0]
    assert isinstance(created_op, Operation)
    assert created_op.name == "test.op"

    assert created_op in interp_functions.known_ops
    assert interp_functions.known_ops[created_op] is created_op

    # Should create an EClass operation and add it to eclass_union_find
    # The EClass should be created and inserted after the operation
    assert len(interp_functions.eclass_union_find._values) == 2  # pyright: ignore[reportPrivateUsage]
    eclass_op = interp_functions.eclass_union_find._values[1]  # pyright: ignore[reportPrivateUsage]
    assert isinstance(eclass_op, eqsat.EClassOp)


def test_run_create_operation_existing_operation_in_use_by_eclass():
    """Test that run_create_operation returns existing operation when it's still in use."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    ctx = interp_functions.get_ctx(interpreter)
    ctx.register_dialect("test", lambda: test.Test)
    interpreter.register_implementations(interp_functions)

    # Set up a mock rewriter
    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block, Region
    from xdsl.pattern_rewriter import PatternRewriter

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        root = test.TestOp()
        operand = eqsat.EClassOp(create_ssa_value(i32), res_type=i32).result
        # Create an existing operation that's identical to what we'll create
        existing_op = test.TestOp((operand,), (i32,))
        _user_op = test.TestOp((existing_op.results[0],), (i32,))
        _eclass_user = eqsat.EClassOp(existing_op.results[0])

    rewriter = PatternRewriter(root)
    interp_functions.set_rewriter(interpreter, rewriter)

    # Create a user for the existing operation to ensure it's "in use"

    interp_functions.populate_known_ops(testmodule)

    # Create CreateOperationOp that will create an identical operation
    create_op = pdl_interp.CreateOperationOp(
        name="test.op",
        input_operands=[operand],
        input_attributes=[],
        input_result_types=[create_ssa_value(pdl.TypeType())],
    )

    # Track initial rewriter state
    initial_has_done_action = rewriter.has_done_action

    # Run the create operation
    result = interp_functions.run_create_operation(
        interpreter, create_op, (operand, i32)
    )

    # Should return the existing operation, not create a new one
    assert len(result.values) == 1
    returned_op = result.values[0]
    assert returned_op is existing_op

    assert rewriter.has_done_action == initial_has_done_action


def test_run_create_operation_existing_operation_in_use():
    """Test that run_create_operation returns existing operation when it's still in use but not by an eclass."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    ctx = interp_functions.get_ctx(interpreter)
    ctx.register_dialect("test", lambda: test.Test)
    interpreter.register_implementations(interp_functions)

    # Set up a mock rewriter
    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block, Region
    from xdsl.pattern_rewriter import PatternRewriter

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        root = test.TestOp()
        operand = eqsat.EClassOp(create_ssa_value(i32), res_type=i32).result
        # Create an existing operation that's identical to what we'll create
        existing_op = test.TestOp((operand,), (i32,))
        _user_op = test.TestOp((existing_op.results[0],), (i32,))

    rewriter = PatternRewriter(root)
    interp_functions.set_rewriter(interpreter, rewriter)

    # Create a user for the existing operation to ensure it's "in use"

    interp_functions.populate_known_ops(testmodule)

    # Create CreateOperationOp that will create an identical operation
    create_op = pdl_interp.CreateOperationOp(
        name="test.op",
        input_operands=[operand],
        input_attributes=[],
        input_result_types=[create_ssa_value(pdl.TypeType())],
    )

    # Run the create operation
    result = interp_functions.run_create_operation(
        interpreter, create_op, (operand, i32)
    )

    # Should return the existing operation, not create a new one
    assert len(result.values) == 1
    returned_op = result.values[0]
    assert returned_op is existing_op

    assert any(
        isinstance(use.operation, eqsat.EClassOp) for use in returned_op.results[0].uses
    )

    assert rewriter.has_done_action  # create a new EClassOp


def test_run_create_operation_existing_operation_not_in_use():
    """Test that run_create_operation reuses an operation not currently
    in use by wrapping it in a new eclass."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    ctx = interp_functions.get_ctx(interpreter)
    ctx.register_dialect("test", lambda: test.Test)
    interpreter.register_implementations(interp_functions)

    # Set up a mock rewriter
    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block, Region
    from xdsl.pattern_rewriter import PatternRewriter

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        root = test.TestOp()
        operand = eqsat.EClassOp(create_ssa_value(i32), res_type=i32).result
        # Create an existing operation with no result uses
        existing_op = test.TestOp((operand,), (i32,))
    rewriter = PatternRewriter(root)
    interp_functions.set_rewriter(interpreter, rewriter)

    # Verify the existing operation has no uses
    assert len(existing_op.results) > 0, "Existing operation must have results"
    assert not existing_op.results[0].uses, (
        "Existing operation result should have no uses"
    )

    interp_functions.populate_known_ops(testmodule)

    # Create CreateOperationOp that will create an identical operation
    create_op = pdl_interp.CreateOperationOp(
        name="test.op",
        input_operands=[operand],
        input_attributes=[],
        input_result_types=[create_ssa_value(pdl.TypeType())],
    )

    # Run the create operation
    result = interp_functions.run_create_operation(
        interpreter, create_op, (operand, i32)
    )

    # Should reuse the existing operation
    assert len(result.values) == 1
    created_op = result.values[0]
    assert created_op is existing_op

    assert existing_op.results[0].first_use, "Existing operation result have a use now"
    assert isinstance(existing_op.results[0].first_use.operation, eqsat.EClassOp)


def test_run_finalize_empty_stack():
    """Test that run_finalize handles empty backtrack stack correctly."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    interpreter.register_implementations(interp_functions)

    # Test finalize with empty backtrack stack - should return empty values
    result = interpreter.run_op(pdl_interp.FinalizeOp(), ())
    assert result == ()


def test_backtrack_stack_manipulation():
    """Test that backtrack stack operations work correctly."""
    interp_functions = EqsatPDLInterpFunctions()

    # Verify initial state
    assert len(interp_functions.backtrack_stack) == 0
    assert interp_functions.visited

    # Test adding to backtrack stack
    from xdsl.interpreters.eqsat_pdl_interp import BacktrackPoint
    from xdsl.ir import Block
    from xdsl.utils.scoped_dict import ScopedDict

    block = Block()
    scope = ScopedDict[Any, Any]()
    gdo_op = pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType()))
    backtrack_point = BacktrackPoint(block, (), scope, gdo_op, 0, 2)

    interp_functions.backtrack_stack.append(backtrack_point)
    assert len(interp_functions.backtrack_stack) == 1
    assert interp_functions.backtrack_stack[0].index == 0
    assert interp_functions.backtrack_stack[0].max_index == 2


def test_run_finalize_with_backtrack_stack():
    """Test that run_finalize handles non-empty backtrack stack correctly."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    interpreter.register_implementations(interp_functions)

    # Import necessary classes
    from xdsl.interpreter import Successor
    from xdsl.interpreters.eqsat_pdl_interp import BacktrackPoint
    from xdsl.ir import Block
    from xdsl.utils.scoped_dict import ScopedDict

    # Create a backtrack point that hasn't reached its max index
    block = Block()
    scope = ScopedDict[Any, Any]()
    gdo_op = pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType()))
    backtrack_point = BacktrackPoint(
        block, (), scope, gdo_op, 0, 2
    )  # index < max_index

    interp_functions.backtrack_stack.append(backtrack_point)

    # Store original interpreter scope
    original_scope = interpreter._ctx  # pyright: ignore[reportPrivateUsage]

    # Test finalize with backtrack stack that can continue
    result = interp_functions.run_finalize(interpreter, pdl_interp.FinalizeOp(), ())

    # Should return a Successor to continue backtracking
    assert isinstance(result.terminator_value, Successor)
    assert result.terminator_value.block is block
    assert result.terminator_value.args == ()
    assert result.values == ()

    # Should increment the index and set visited to False
    assert interp_functions.backtrack_stack[0].index == 1
    assert not interp_functions.visited

    # Should restore the interpreter scope from the backtrack point
    assert interpreter._ctx is scope  # pyright: ignore[reportPrivateUsage]
    assert interpreter._ctx is not original_scope  # pyright: ignore[reportPrivateUsage]

    # Test finalize when backtrack point reaches max index
    interp_functions.backtrack_stack[0].index = 2  # Set to max_index
    from xdsl.interpreter import ReturnedValues

    result = interp_functions.run_finalize(interpreter, pdl_interp.FinalizeOp(), ())

    # Should pop the backtrack point and return empty values
    assert isinstance(result.terminator_value, ReturnedValues)
    assert result.terminator_value.values == ()
    assert result.values == ()
    assert len(interp_functions.backtrack_stack) == 0


def test_run_replace():
    """Test that run_replace correctly merges EClass operations."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()

    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block, Region

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        # Create test operations
        c0 = create_ssa_value(i32)
        original_op = test.TestOp((c0,), (i32,))
        replacement_op = test.TestOp((c0,), (i32,))

        # Create EClass operations for both
        original_eclass = eqsat.EClassOp(original_op.results[0], res_type=i32)
        replacement_eclass = eqsat.EClassOp(replacement_op.results[0], res_type=i32)

    # Add both EClass operations to union-find
    interp_functions.eclass_union_find.add(original_eclass)
    interp_functions.eclass_union_find.add(replacement_eclass)

    # Create a ReplaceOp for testing
    input_op_value = create_ssa_value(pdl.OperationType())
    repl_value = create_ssa_value(pdl.ValueType())
    replace_op = pdl_interp.ReplaceOp(input_op_value, [repl_value])

    rewriter = PatternRewriter(original_op)
    interp_functions.set_rewriter(interpreter, rewriter)

    # Call run_replace directly
    result = interp_functions.run_replace(
        interpreter, replace_op, (original_op, replacement_eclass.results[0])
    )

    # Should return empty tuple
    assert result.values == ()

    # Should have merged the EClass operations in union-find
    assert interp_functions.eclass_union_find.connected(
        original_eclass, replacement_eclass
    )

    # Should have added a merge todo
    assert len(interp_functions.worklist) == 1
    merge_todo = interp_functions.worklist[0]
    # One of them should be the canonical representative
    canonical = interp_functions.eclass_union_find.find(original_eclass)
    assert interp_functions.eclass_union_find.find(merge_todo) == canonical


def test_run_replace_same_eclass():
    """Test that run_replace handles replacing with same EClass correctly."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()

    # Create test operation
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32,))

    # Create EClass operation
    eclass_op = eqsat.EClassOp(test_op.results[0], res_type=i32)

    # Add EClass operation to union-find
    interp_functions.eclass_union_find.add(eclass_op)

    # Create a ReplaceOp for testing
    input_op_value = create_ssa_value(pdl.OperationType())
    repl_value = create_ssa_value(pdl.ValueType())
    replace_op = pdl_interp.ReplaceOp(input_op_value, [repl_value])

    # Call run_replace directly
    result = interp_functions.run_replace(
        interpreter, replace_op, (test_op, eclass_op.results[0])
    )

    # Should return empty tuple
    assert not result.values

    # Should not add any merge todos since it's the same EClass
    assert not interp_functions.worklist


def test_run_replace_error_not_eclass_original():
    """Test that run_replace raises error when original operation result is not used by EClass."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()

    # Create test operation without EClass usage
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32,))

    # Create another operation that uses test_op's result (not an EClass)
    _user_op = test.TestOp((test_op.results[0],), (i32,))

    # Create replacement EClass
    replacement_op = test.TestOp((c0,), (i32,))
    replacement_eclass = eqsat.EClassOp(replacement_op.results[0], res_type=i32)

    # Create a ReplaceOp for testing
    input_op_value = create_ssa_value(pdl.OperationType())
    repl_value = create_ssa_value(pdl.ValueType())
    replace_op = pdl_interp.ReplaceOp(input_op_value, [repl_value])

    # Should raise InterpretationError
    with pytest.raises(
        InterpretationError,
        match="Replaced operation result must be used by an eclass",
    ):
        interp_functions.run_replace(
            interpreter, replace_op, (test_op, replacement_eclass.results[0])
        )


def test_run_replace_error_not_eclass_replacement():
    """Test that run_replace raises error when replacement value is not from EClass."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()

    # Create test operation with EClass usage
    c0 = create_ssa_value(i32)
    original_op = test.TestOp((c0,), (i32,))
    _original_eclass = eqsat.EClassOp(original_op.results[0], res_type=i32)

    # Create replacement operation without EClass
    replacement_op = test.TestOp((c0,), (i32,))

    # Create a ReplaceOp for testing
    input_op_value = create_ssa_value(pdl.OperationType())
    repl_value = create_ssa_value(pdl.ValueType())
    replace_op = pdl_interp.ReplaceOp(input_op_value, [repl_value])

    # Should raise InterpretationError
    with pytest.raises(
        InterpretationError,
        match="Replacement value must be the result of an eclass",
    ):
        interp_functions.run_replace(
            interpreter, replace_op, (original_op, replacement_op.results[0])
        )


def test_rebuilding():
    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block, Region

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        a = create_ssa_value(i32)
        b = create_ssa_value(i32)
        x = create_ssa_value(i32)

        c_a = eqsat.EClassOp(a, res_type=i32).result
        c_b = eqsat.EClassOp(b, res_type=i32).result
        c_x = eqsat.EClassOp(x, res_type=i32).result

        c = arith.MuliOp(c_a, c_a).result
        c_c = eqsat.EClassOp(c, res_type=i32).result

        d = arith.MuliOp(c_b, c_b).result
        c_d = eqsat.EClassOp(d, res_type=i32).result
    a.name_hint = "a"
    b.name_hint = "b"
    x.name_hint = "x"
    c_a.name_hint = "c_a"
    c_b.name_hint = "c_b"
    c_x.name_hint = "c_x"
    c.name_hint = "c"
    c_c.name_hint = "c_c"
    d.name_hint = "d"
    c_d.name_hint = "c_d"

    ctx = Context()
    ctx.register_dialect("func", lambda: func.Func)
    ctx.register_dialect("eqsat", lambda: eqsat.EqSat)
    ctx.register_dialect("arith", lambda: arith.Arith)
    rewriter = PatternRewriter(c_b.owner)

    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    ctx = interp_functions.get_ctx(interpreter)
    interp_functions.set_rewriter(interpreter, rewriter)

    interp_functions.populate_known_ops(testmodule)

    assert isinstance(c_x.owner, eqsat.EClassOp)
    assert isinstance(c_a.owner, eqsat.EClassOp)
    assert isinstance(c_b.owner, eqsat.EClassOp)
    assert isinstance(c_d.owner, eqsat.EClassOp)

    interp_functions.eclass_union(interpreter, c_x.owner, c_d.owner)
    interp_functions.worklist.append(c_x.owner)

    interp_functions.eclass_union(interpreter, c_b.owner, c_a.owner)
    interp_functions.worklist.append(c_b.owner)

    interp_functions.rebuild(interpreter)

    assert (
        str(testmodule)
        == """builtin.module {
  %a = "test.op"() : () -> i32
  %b = "test.op"() : () -> i32
  %x = "test.op"() : () -> i32
  %c_b = eqsat.eclass %b, %a : i32
  %c_x = eqsat.eclass %x, %c : i32
  %c = arith.muli %c_b, %c_b : i32
}"""
    )


def test_rebuilding_parents_already_equivalent():
    """
    Take for example:
    ```
    %c_x = eqsat.eclass(%x)
    %c_y = eqsat.eclass(%y)
    %a = f(%c_x)
    %b = f(%c_y)
    eqsat.eclass(%a, %b)
    ```
    When `%x` and `%y` become equivalent, this becomes:
    ```
    %c_xy = eqsat.eclass(%x, %y)
    %a = f(%x)
    %b = f(%x)
    eqsat.eclass(%a, %b)
    ```
    The rebuilding procedure has to deduplicate `%a` and `%b`, and the eclass should only contain `%c_xy`.
    """
    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block, Region

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        x = create_ssa_value(i32)
        y = create_ssa_value(i32)

        c_x = eqsat.EClassOp(x, res_type=i32).result
        c_y = eqsat.EClassOp(y, res_type=i32).result

        a = test.TestOp((c_x,), result_types=(i32,)).results[0]
        b = test.TestOp((c_y,), result_types=(i32,)).results[0]

        c_ab = eqsat.EClassOp(a, b, res_type=i32).result
    x.name_hint = "x"
    y.name_hint = "y"
    c_x.name_hint = "c_x"
    c_y.name_hint = "c_y"
    a.name_hint = "a"
    b.name_hint = "b"
    c_ab.name_hint = "c_ab"

    rewriter = PatternRewriter(c_ab.owner)

    interpreter = Interpreter(ModuleOp(()))
    interp_functions = EqsatPDLInterpFunctions()
    ctx = EqsatPDLInterpFunctions.get_ctx(interpreter)
    ctx.register_dialect("func", lambda: func.Func)
    ctx.register_dialect("eqsat", lambda: eqsat.EqSat)
    ctx.register_dialect("test", lambda: test.Test)

    interp_functions.set_rewriter(interpreter, rewriter)

    interp_functions.populate_known_ops(testmodule)

    assert isinstance(c_x.owner, eqsat.EClassOp)
    assert isinstance(c_y.owner, eqsat.EClassOp)

    interp_functions.eclass_union(interpreter, c_x.owner, c_y.owner)
    interp_functions.worklist.append(c_x.owner)

    interp_functions.rebuild(interpreter)

    assert (
        str(testmodule)
        == """builtin.module {
  %x = "test.op"() : () -> i32
  %y = "test.op"() : () -> i32
  %c_x = eqsat.eclass %x, %y : i32
  %b = "test.op"(%c_x) : (i32) -> i32
  %c_ab = eqsat.eclass %b : i32
}"""
    )


def test_run_get_defining_op_block_argument():
    """Test that run_get_defining_op returns None for block arguments."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    interpreter.register_implementations(interp_functions)

    # Create a block argument
    from xdsl.ir import Block

    block = Block((), arg_types=(i32,))
    block_arg = block.args[0]

    # Test GetDefiningOpOp with block argument
    result = interpreter.run_op(
        pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType())),
        (block_arg,),
    )

    # Should return None for block arguments
    assert result == (None,)

    # Should not have set up any backtracking for block arguments
    assert len(interp_functions.backtrack_stack) == 0

    # test the case where the value is used in an EClassOp:
    block.add_op(
        gdo := pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType()))
    )

    eclass_result = eqsat.EClassOp(block_arg, res_type=i32).result

    # set dummy value
    interpreter.push_scope()
    interpreter._ctx[block_arg] = None  # pyright: ignore[reportPrivateUsage]

    result = interpreter.run_op(
        gdo,
        (eclass_result,),
    )
    assert result == (None,)
    assert len(interp_functions.backtrack_stack) == 1


def test_run_choose_not_visited():
    """Test that run_choose handles ChooseOp when not visited (coming from run_finalize)."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    interpreter.register_implementations(interp_functions)

    # Create blocks for choices and default
    from xdsl.ir import Block

    choice1_block = Block()
    choice2_block = Block()
    default_block = Block()

    # Create ChooseOp with two choices
    choose_op = eqsat_pdl_interp.ChooseOp([choice1_block, choice2_block], default_block)

    # Set up backtrack stack with this ChooseOp and visited=False
    from xdsl.interpreters.eqsat_pdl_interp import BacktrackPoint
    from xdsl.utils.scoped_dict import ScopedDict

    block = Block()
    scope = ScopedDict[Any, Any]()
    backtrack_point = BacktrackPoint(block, (), scope, choose_op, 1, 2)  # Index 1
    interp_functions.backtrack_stack.append(backtrack_point)
    interp_functions.visited = False

    # Test ChooseOp execution
    from xdsl.interpreter import Successor

    result = interp_functions.run_choose(interpreter, choose_op, ())

    # Should use index from backtrack stack (1) and set visited to True
    assert interp_functions.visited
    assert isinstance(result.terminator_value, Successor)
    assert (
        result.terminator_value.block == choice2_block
    )  # Should go to choice at index 1
    assert result.terminator_value.args == ()
    assert result.values == ()


def test_run_choose_visited():
    """Test that run_choose handles ChooseOp when visited (creating new backtrack point)."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    interpreter.register_implementations(interp_functions)

    # Create blocks for choices and default
    from xdsl.ir import Block

    choice1_block = Block()
    choice2_block = Block()
    default_block = Block()

    # Create a parent block and add the ChooseOp to it
    parent_block = Block()
    choose_op = eqsat_pdl_interp.ChooseOp([choice1_block, choice2_block], default_block)
    parent_block.add_op(choose_op)

    # Set visited to True and create a parent scope for the interpreter context
    interp_functions.visited = True

    # Create a child scope to give the current context a parent
    from xdsl.utils.scoped_dict import ScopedDict

    child_scope = ScopedDict(parent=interpreter._ctx)  # pyright: ignore[reportPrivateUsage]
    interpreter._ctx = child_scope  # pyright: ignore[reportPrivateUsage]

    # Test ChooseOp execution
    from xdsl.interpreter import Successor

    result = interp_functions.run_choose(interpreter, choose_op, ())

    # Should create new backtrack point and use index 0
    assert len(interp_functions.backtrack_stack) == 1
    assert interp_functions.backtrack_stack[0].index == 0
    assert interp_functions.backtrack_stack[0].max_index == 2  # len(choices)
    assert interp_functions.backtrack_stack[0].cause == choose_op

    # Should return first choice
    assert isinstance(result.terminator_value, Successor)
    assert (
        result.terminator_value.block == choice1_block
    )  # Should go to choice at index 0
    assert result.terminator_value.args == ()
    assert result.values == ()


def test_run_choose_default_dest():
    """Test that run_choose goes to default destination when index equals len(choices)."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    interpreter.register_implementations(interp_functions)

    # Create blocks for choices and default
    from xdsl.ir import Block

    choice1_block = Block()
    choice2_block = Block()
    default_block = Block()

    # Create ChooseOp with two choices
    choose_op = eqsat_pdl_interp.ChooseOp([choice1_block, choice2_block], default_block)

    # Set up backtrack stack with index equal to number of choices
    from xdsl.interpreters.eqsat_pdl_interp import BacktrackPoint
    from xdsl.utils.scoped_dict import ScopedDict

    block = Block()
    scope = ScopedDict[Any, Any]()
    backtrack_point = BacktrackPoint(
        block, (), scope, choose_op, 2, 2
    )  # Index 2 = len(choices)
    interp_functions.backtrack_stack.append(backtrack_point)
    interp_functions.visited = False

    # Test ChooseOp execution
    from xdsl.interpreter import Successor

    result = interp_functions.run_choose(interpreter, choose_op, ())

    # Should go to default destination
    assert interp_functions.visited
    assert isinstance(result.terminator_value, Successor)
    assert (
        result.terminator_value.block == default_block
    )  # Should go to default destination
    assert result.terminator_value.args == ()
    assert result.values == ()


def test_run_choose_error_wrong_op():
    """Test that run_choose raises error when expected ChooseOp is not at top of backtrack stack."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    interpreter.register_implementations(interp_functions)

    # Create blocks for choices and default
    from xdsl.ir import Block

    choice1_block = Block()
    default_block = Block()

    # Create two different ChooseOp operations
    choose_op1 = eqsat_pdl_interp.ChooseOp([choice1_block], default_block)
    choose_op2 = eqsat_pdl_interp.ChooseOp([choice1_block], default_block)

    # Set up backtrack stack with different choose_op and visited=False
    from xdsl.interpreters.eqsat_pdl_interp import BacktrackPoint
    from xdsl.utils.scoped_dict import ScopedDict

    block = Block()
    scope = ScopedDict[Any, Any]()
    backtrack_point = BacktrackPoint(block, (), scope, choose_op1, 0, 1)  # Different op
    interp_functions.backtrack_stack.append(backtrack_point)
    interp_functions.visited = False

    # Test should raise InterpretationError when using different choose_op
    with pytest.raises(
        InterpretationError,
        match="Expected this ChooseOp to be at the top of the backtrack stack.",
    ):
        interp_functions.run_choose(interpreter, choose_op2, ())


def test_eclass_union_different_constants_fails():
    """Test that eclass_union of two ConstantEClassOp with different constant values fails."""
    interpreter = Interpreter(ModuleOp(()))
    interp_functions = EqsatPDLInterpFunctions()

    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.ir import Block, Region

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        # Create two constant operations with different values
        const1 = arith.ConstantOp(IntegerAttr(1, i32))
        const2 = arith.ConstantOp(IntegerAttr(2, i32))

        # Create ConstantEClassOp for both
        const_eclass1 = eqsat.ConstantEClassOp(const1.result)
        const_eclass1.value = IntegerAttr(1, i32)
        const_eclass2 = eqsat.ConstantEClassOp(const2.result)
        const_eclass2.value = IntegerAttr(2, i32)

    # Add both to union-find
    interp_functions.eclass_union_find.add(const_eclass1)
    interp_functions.eclass_union_find.add(const_eclass2)

    # Should raise assertion error when trying to union different constant eclasses
    with pytest.raises(
        AssertionError, match="Trying to union two different constant eclasses."
    ):
        interp_functions.eclass_union(interpreter, const_eclass1, const_eclass2)


def test_eclass_union_constant_with_regular():
    """Test that eclass_union of ConstantEClassOp with regular EClassOp results in ConstantEClassOp containing both operands."""
    interp_functions = EqsatPDLInterpFunctions()

    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.ir import Block, Region

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        # Create a constant operation
        const_op = arith.ConstantOp(IntegerAttr(1, i32))

        # Create a regular operation
        regular_op = test.TestOp(result_types=[i32])

        # Create ConstantEClassOp
        const_eclass = eqsat.ConstantEClassOp(const_op.result)
        const_eclass.value = IntegerAttr(1, i32)

        # Create regular EClassOp with the regular operation's result
        regular_eclass = eqsat.EClassOp(regular_op.results[0], res_type=i32)

    rewriter = PatternRewriter(const_op)
    interpreter = Interpreter(ModuleOp(()))
    interp_functions.set_rewriter(interpreter, rewriter)

    # Add both to union-find
    interp_functions.eclass_union_find.add(const_eclass)
    interp_functions.eclass_union_find.add(regular_eclass)

    # Union the constant eclass with the regular eclass
    interp_functions.eclass_union(interpreter, const_eclass, regular_eclass)

    # Find the canonical representative
    canonical = interp_functions.eclass_union_find.find(const_eclass)

    # Verify that the canonical is a ConstantEClassOp
    assert isinstance(canonical, eqsat.ConstantEClassOp), (
        "Result should be a ConstantEClassOp"
    )

    # Verify that the canonical contains both operands (constant and regular operation results)
    assert len(canonical.operands) == 2, (
        "ConstantEClassOp should contain both operands after union"
    )
    operand_values = [op for op in canonical.operands]
    assert const_op.result in operand_values, (
        "Should contain the constant operation result"
    )
    assert regular_op.results[0] in operand_values, (
        "Should contain the regular operation result"
    )

    # Verify the constant value is preserved
    assert canonical.value == IntegerAttr(1, i32), "Constant value should be preserved"


def test_run_replace_no_uses_returns_empty():
    """Test that run_replace errors when input_op has no uses."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()

    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block, Region

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        # Create test operation with no uses
        c0 = create_ssa_value(i32)
        input_op = test.TestOp((c0,), (i32,))
        # Don't create any uses for input_op.results[0]

        # Create replacement eclass
        replacement_op = test.TestOp((c0,), (i32,))
        replacement_eclass = eqsat.EClassOp(replacement_op.results[0], res_type=i32)

    # Add replacement eclass to union-find
    interp_functions.eclass_union_find.add(replacement_eclass)

    # Create a ReplaceOp for testing
    input_op_value = create_ssa_value(pdl.OperationType())
    repl_value = create_ssa_value(pdl.ValueType())
    replace_op = pdl_interp.ReplaceOp(input_op_value, [repl_value])

    rewriter = PatternRewriter(input_op)
    interp_functions.set_rewriter(interpreter, rewriter)

    # Call run_replace - should raise InterpretationError since input_op has no uses
    with pytest.raises(
        InterpretationError,
        match="Operation's result can only be used once, by an eclass operation.",
    ):
        interp_functions.run_replace(
            interpreter, replace_op, (input_op, replacement_eclass.results[0])
        )


def test_run_replace_multi_results():
    """Test that run_replace works with multi-result operations."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()

    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block, Region

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        # Create test operation with multiple results
        c0 = create_ssa_value(i32)
        original_op = test.TestOp((c0,), (i32, i64))

        # Create EClass operations for the results
        eclass1 = eqsat.EClassOp(original_op.results[0], res_type=i32)
        eclass2 = eqsat.EClassOp(original_op.results[1], res_type=i64)

        # Create replacement operations/eclasses
        repl1_op = test.TestOp((c0,), (i32,))
        repl1_eclass = eqsat.EClassOp(repl1_op.results[0], res_type=i32)

        repl2_op = test.TestOp((c0,), (i64,))
        repl2_eclass = eqsat.EClassOp(repl2_op.results[0], res_type=i64)

    # Add to union find
    interp_functions.eclass_union_find.add(eclass1)
    interp_functions.eclass_union_find.add(eclass2)
    interp_functions.eclass_union_find.add(repl1_eclass)
    interp_functions.eclass_union_find.add(repl2_eclass)

    rewriter = PatternRewriter(original_op)
    interp_functions.set_rewriter(interpreter, rewriter)

    # Create ReplaceOp with RangeType to simulate multiple replacement values
    input_op_val = create_ssa_value(pdl.OperationType())
    repl_range_val = create_ssa_value(pdl.RangeType(pdl.ValueType()))
    replace_op = pdl_interp.ReplaceOp(input_op_val, [repl_range_val])

    # Prepare arguments: input op and a tuple of replacement values
    repl_values = (repl1_eclass.results[0], repl2_eclass.results[0])
    args = (original_op, repl_values)

    assert interp_functions.run_replace(interpreter, replace_op, args).values == ()

    assert interp_functions.eclass_union_find.connected(eclass1, repl1_eclass)
    assert interp_functions.eclass_union_find.connected(eclass2, repl2_eclass)


def test_run_replace_rangetype_mixed():
    """
    Test that run_replace correctly flattens a mix of ValueType and RangeType
    replacement arguments and aligns them with the operation's results.
    """

    @irdl_op_definition
    class MultiResultGroupsOp(IRDLOperation):
        name = "test.multi_result_group"
        res1 = var_result_def()
        res2 = result_def()
        res3 = var_result_def()

        irdl_options = [AttrSizedResultSegments()]

        def __init__(
            self,
            res1: Sequence[Attribute],
            res2: Attribute,
            res3: Sequence[Attribute],
        ) -> None:
            super().__init__(result_types=[res1, res2, res3])

    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    interpreter.register_implementations(interp_functions)

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        # Create the op being replaced.
        # Total 5 results: 2 in group 1, 1 in group 2, 2 in group 3
        original_op = MultiResultGroupsOp(
            res1=[i32, i32],
            res2=i64,
            res3=[i32, i64],
        )

        orig_eclasses = [
            eqsat.EClassOp(res, res_type=res.type) for res in original_op.results
        ]

        c0 = create_ssa_value(i32)

        # Generate 5 dummy replacements
        repl_ops = [test.TestOp((c0,), (r.type,)) for r in original_op.results]
        repl_eclasses = [
            eqsat.EClassOp(op.results[0], res_type=op.results[0].type)
            for op in repl_ops
        ]
        repl_values = [e.results[0] for e in repl_eclasses]

    # Register EClasses in UnionFind
    for e in orig_eclasses + repl_eclasses:
        interp_functions.eclass_union_find.add(e)

    rewriter = PatternRewriter(original_op)
    interp_functions.set_rewriter(interpreter, rewriter)

    # Configure ReplaceOp with mixed types:
    # Structure: [Range (2 items), Value (1 item), Range (2 items)]
    # This tests the flattening logic: (2 + 1 + 2) == 5 results
    input_op_val = create_ssa_value(pdl.OperationType())

    types = [
        pdl.RangeType(pdl.ValueType()),
        pdl.ValueType(),
        pdl.RangeType(pdl.ValueType()),
    ]

    # Create replacement args corresponding to types
    # Arg 0 (Range): items 0 and 1
    # Arg 1 (Value): item 2
    # Arg 2 (Range): items 3 and 4
    replace_args_values = (
        (repl_values[0], repl_values[1]),
        repl_values[2],
        (repl_values[3], repl_values[4]),
    )

    # Create SSA values for the ReplaceOp definition (used for type info)
    repl_ssa_vals: list[SSAValue] = [create_ssa_value(t) for t in types]
    replace_op = pdl_interp.ReplaceOp(input_op_val, repl_ssa_vals)

    args = (original_op, *replace_args_values)
    result = interp_functions.run_replace(interpreter, replace_op, args)

    assert result.values == ()

    # Verify that every original EClass is connected to the corresponding replacement EClass
    # This ensures the flattening of [Range, Value, Range] mapped correctly to [0, 1, 2, 3, 4]
    for orig_ec, repl_ec in zip(orig_eclasses, repl_eclasses):
        assert interp_functions.eclass_union_find.connected(orig_ec, repl_ec)


def test_run_replace_rangetype_full_coverage():
    """
    Test replacing an operation with multiple results using a single RangeType
    argument containing all replacement values.
    """
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()

    from xdsl.builder import ImplicitBuilder

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block

    with ImplicitBuilder(block):
        # Op with 3 results
        c0 = create_ssa_value(i32)
        original_op = test.TestOp((c0,), (i32, i32, i32))

        # Wrap originals in EClasses
        orig_eclasses = [
            eqsat.EClassOp(res, res_type=i32) for res in original_op.results
        ]

        # Create replacements
        repl_ops = [test.TestOp((c0,), (i32,)) for _ in range(3)]
        repl_eclasses = [eqsat.EClassOp(op.results[0], res_type=i32) for op in repl_ops]

    for e in orig_eclasses + repl_eclasses:
        interp_functions.eclass_union_find.add(e)

    rewriter = PatternRewriter(original_op)
    interp_functions.set_rewriter(interpreter, rewriter)

    # ReplaceOp configuration: Single RangeType covering all results
    input_op_val = create_ssa_value(pdl.OperationType())
    repl_val_range = create_ssa_value(pdl.RangeType(pdl.ValueType()))
    replace_op = pdl_interp.ReplaceOp(input_op_val, [repl_val_range])

    # Prepare args: Input Op, and a Tuple of all replacements
    repl_tuple = tuple(e.results[0] for e in repl_eclasses)
    args = (original_op, repl_tuple)

    assert interp_functions.run_replace(interpreter, replace_op, args).values == ()

    # Verify all connections
    for o, r in zip(orig_eclasses, repl_eclasses):
        assert interp_functions.eclass_union_find.connected(o, r)


def test_run_replace_length_mismatch():
    """
    Test that run_replace asserts when the flattened number of replacement values
    does not match the number of operation results.
    """
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()

    from xdsl.builder import ImplicitBuilder

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block

    with ImplicitBuilder(block):
        c0 = create_ssa_value(i32)
        # Op has 2 results
        original_op = test.TestOp((c0,), (i32, i32))

        # Create EClasses (required to pass early checks in run_replace)
        eqsat.EClassOp(original_op.results[0], res_type=i32)
        eqsat.EClassOp(original_op.results[1], res_type=i32)

        # Replacement (only 1 provided)
        repl_op = test.TestOp((c0,), (i32,))
        repl_eclass = eqsat.EClassOp(repl_op.results[0], res_type=i32)

    input_op_val = create_ssa_value(pdl.OperationType())
    repl_val = create_ssa_value(pdl.ValueType())
    replace_op = pdl_interp.ReplaceOp(input_op_val, [repl_val])

    args = (original_op, repl_eclass.results[0])

    # Should raise assertion error because len(results)=2 != len(repl)=1
    with pytest.raises(AssertionError):
        interp_functions.run_replace(interpreter, replace_op, args)


def test_run_create_operation_multiple_results():
    """Test that run_create_operation handles operations with multiple results correctly."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    ctx = EqsatPDLInterpFunctions.get_ctx(interpreter)
    ctx.register_dialect("test", lambda: test.Test)
    interpreter.register_implementations(interp_functions)

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        root = test.TestOp()
        c0 = create_ssa_value(i32)
        # Input operand eclass
        operand_op = test.TestOp((c0,), (i32,))
        operand_eclass = eqsat.EClassOp(operand_op.results[0], res_type=i32)

    rewriter = PatternRewriter(root)
    interp_functions.set_rewriter(interpreter, rewriter)
    interp_functions.populate_known_ops(testmodule)
    interp_functions.eclass_union_find.add(operand_eclass)

    # Create CreateOperationOp for "test.op" with 2 results
    create_op = pdl_interp.CreateOperationOp(
        name="test.op",
        input_operands=[operand_eclass.result],
        input_attributes=[],
        input_result_types=[
            create_ssa_value(pdl.TypeType()),
            create_ssa_value(pdl.TypeType()),
        ],
    )

    # Run the create operation
    result = interp_functions.run_create_operation(
        interpreter, create_op, (operand_eclass.result, i32, i64)
    )

    assert len(result.values) == 1
    created_op = result.values[0]
    assert created_op.name == "test.op"
    assert len(created_op.results) == 2
    assert created_op.results[0].type == i32
    assert created_op.results[1].type == i64

    # Check that both results are wrapped in distinct EClassOps
    for res in created_op.results:
        assert res.has_one_use()
        user = res.first_use.operation
        assert isinstance(user, eqsat.EClassOp)
        assert interp_functions.eclass_union_find.find(user) is user


def test_run_create_operation_runs_analysis():
    """Test that run_create_operation runs registered analyses on new operations."""
    from dataclasses import dataclass

    from typing_extensions import Self

    from xdsl.analysis.dataflow import DataFlowSolver
    from xdsl.analysis.sparse_analysis import (
        AbstractLatticeValue,
        Lattice,
        SparseForwardDataFlowAnalysis,
    )

    @dataclass(frozen=True)
    class TestLatticeValue(AbstractLatticeValue):
        value: int

        @classmethod
        def initial_value(cls) -> Self:
            return cls(0)

        def meet(self, other: "TestLatticeValue") -> "TestLatticeValue":
            return TestLatticeValue(min(self.value, other.value))

        def join(self, other: "TestLatticeValue") -> "TestLatticeValue":
            return TestLatticeValue(max(self.value, other.value))

    class TestLattice(Lattice[TestLatticeValue]):
        value_cls = TestLatticeValue

    class TestAnalysis(SparseForwardDataFlowAnalysis[TestLattice]):
        def __init__(self, solver: DataFlowSolver):
            super().__init__(solver, TestLattice)
            self.visited_ops: list[Operation] = []

        def visit_operation_impl(
            self,
            op: Operation,
            operands: list[TestLattice],
            results: list[TestLattice],
        ) -> None:
            self.visited_ops.append(op)
            # Simple transfer function: result = operand + 1
            if operands and results:
                val = operands[0].value.value
                # If op is EClassOp, it passes through value.
                # If op is TestOp (the created op), it increments.
                if isinstance(op, eqsat.EClassOp):
                    new_val = val
                else:
                    new_val = val + 1

                self.join(
                    results[0],
                    TestLattice(results[0].anchor, TestLatticeValue(new_val)),
                )

        def set_to_entry_state(self, lattice: TestLattice) -> None:
            pass

    # Setup interpreter
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions()
    ctx = EqsatPDLInterpFunctions.get_ctx(interpreter)
    ctx.register_dialect("test", lambda: test.Test)
    interpreter.register_implementations(interp_functions)

    # Register analysis
    solver = NonPropagatingDataFlowSolver(ctx)
    analysis = TestAnalysis(solver)
    interp_functions.analyses.append(
        cast(SparseForwardDataFlowAnalysis[Lattice[TestLatticeValue]], analysis)
    )

    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block, Region
    from xdsl.pattern_rewriter import PatternRewriter

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        root = test.TestOp()
        c0 = create_ssa_value(i32)
        # Operand EClass with some state
        operand_op = test.TestOp((c0,), (i32,))
        operand_eclass = eqsat.EClassOp(operand_op.results[0], res_type=i32)

    # Initialize lattice state for operand
    operand_lattice = analysis.get_lattice_element(operand_eclass.result)
    operand_lattice._value = TestLatticeValue(  # pyright: ignore[reportPrivateUsage]
        10
    )  # Set initial value

    rewriter = PatternRewriter(root)
    interp_functions.set_rewriter(interpreter, rewriter)
    interp_functions.populate_known_ops(testmodule)
    interp_functions.eclass_union_find.add(operand_eclass)

    # Create CreateOperationOp
    create_op = pdl_interp.CreateOperationOp(
        name="test.op",
        input_operands=[operand_eclass.result],
        input_attributes=[],
        input_result_types=[create_ssa_value(pdl.TypeType())],
    )

    # Run run_create_operation
    result = interp_functions.run_create_operation(
        interpreter, create_op, (operand_eclass.result, i32)
    )

    created_op = result.values[0]
    assert isinstance(created_op, Operation)

    # Verify that analysis visited the new operations
    assert created_op in analysis.visited_ops

    # Find the new EClassOp
    assert created_op.results[0].first_use
    new_eclass_op = created_op.results[0].first_use.operation
    assert isinstance(new_eclass_op, eqsat.EClassOp)
    assert new_eclass_op in analysis.visited_ops

    # Verify values
    # created_op result lattice should be operand + 1 = 11
    res_lattice = analysis.get_lattice_element(created_op.results[0])
    assert res_lattice.value.value == 11

    # new_eclass_op result lattice should be created_op result (passed through) = 11
    eclass_lattice = analysis.get_lattice_element(new_eclass_op.result)
    assert eclass_lattice.value.value == 11
