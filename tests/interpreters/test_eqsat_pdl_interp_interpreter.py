from typing import Any

import pytest

from xdsl.context import Context
from xdsl.dialects import eqsat, pdl, pdl_interp, test
from xdsl.dialects.builtin import ModuleOp, i32, i64
from xdsl.interpreter import Interpreter
from xdsl.interpreters.eqsat_pdl_interp import EqsatPDLInterpFunctions
from xdsl.ir import Operation
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
    interp_functions = EqsatPDLInterpFunctions(Context())

    # Call the method under test
    interp_functions.populate_known_ops(module)

    # Assert that regular operations are in known_ops
    assert regular_op in interp_functions.known_ops
    assert interp_functions.known_ops[regular_op] is regular_op

    # Assert that EClassOp is not in known_ops but is in eclass_union_find
    assert eclass_op not in interp_functions.known_ops
    assert eclass_op in interp_functions.eclass_union_find._index_by_value  # pyright: ignore[reportPrivateUsage]


def test_run_getresult():
    """Test that run_getresult handles EClass operations correctly."""
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EqsatPDLInterpFunctions(Context()))

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


def test_run_getresult_error_case():
    """Test that run_getresult raises error when result is not used by EClass."""
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EqsatPDLInterpFunctions(Context()))

    # Create a test operation with result that is not used by EClass
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32,))
    _eclass_user = eqsat.EClassOp(test_op.results[0])
    _extra_user = test.TestOp((test_op.results[0],))

    # Don't create any EClass operations, so result is not used by EClass

    # Test GetResultOp should raise InterpretationError
    with pytest.raises(
        InterpretationError,
        match="pdl_interp.get_result currently only supports operations with results that are used by a single EClassOp each.",
    ):
        interpreter.run_op(
            pdl_interp.GetResultOp(0, create_ssa_value(pdl.OperationType())), (test_op,)
        )


def test_run_getresults():
    """Test that run_getresults handles EClass operations correctly."""
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EqsatPDLInterpFunctions(Context()))

    # Create a test operation with multiple results
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32, i64))

    # Create EClass operations that use all results
    eclass_op1 = eqsat.EClassOp(test_op.results[0], res_type=i32)
    eclass_op2 = eqsat.EClassOp(test_op.results[1], res_type=i64)

    # Test GetResultsOp with all results wrapped in EClass
    result = interpreter.run_op(
        pdl_interp.GetResultsOp(
            None,
            create_ssa_value(pdl.OperationType()),
            pdl.RangeType(pdl.ValueType()),
        ),
        (test_op,),
    )
    expected_results = [eclass_op1.results[0], eclass_op2.results[0]]
    assert result == (expected_results,)


def test_run_getresults_error_case():
    """Test that run_getresults raises error when result is not used by EClass."""
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EqsatPDLInterpFunctions(Context()))

    # Create a test operation with results that are not used by EClass
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32, i64))
    _eclass_user = eqsat.EClassOp(test_op.results[0])
    _extra_user = test.TestOp((test_op.results[0],))

    # Don't create any EClass operations, so results are not used by EClass

    # Test GetResultsOp should raise InterpretationError
    with pytest.raises(
        InterpretationError,
        match="pdl_interp.get_results currently only supports operations with results that are used by a single EClassOp each.",
    ):
        interpreter.run_op(
            pdl_interp.GetResultsOp(
                None,
                create_ssa_value(pdl.OperationType()),
                pdl.RangeType(pdl.ValueType()),
            ),
            (test_op,),
        )


def test_run_getresult_none_case():
    """Test that run_getresult returns None when result index doesn't exist."""
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EqsatPDLInterpFunctions(Context()))

    # Create a test operation with only one result
    c0 = create_ssa_value(i32)
    test_op = test.TestOp((c0,), (i32,))

    # Try to get result at index 1 (doesn't exist, only index 0 exists)
    result = interpreter.run_op(
        pdl_interp.GetResultOp(1, create_ssa_value(pdl.OperationType())), (test_op,)
    )
    assert result == (None,)


def test_run_getresults_valuetype_multi_results():
    """Test that run_getresults returns None for ValueType with multiple results."""
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EqsatPDLInterpFunctions(Context()))

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


def test_run_getresults_valuetype_no_results():
    """Test that run_getresults returns None for ValueType with no results."""
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EqsatPDLInterpFunctions(Context()))

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


def test_run_getdefiningop():
    """Test that run_getdefiningop handles regular operations correctly."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions(Context())
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


def test_run_getdefiningop_eclass_not_visited():
    """Test that run_getdefiningop handles EClassOp when not visited."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions(Context())
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


def test_run_getdefiningop_eclass_visited():
    """Test that run_getdefiningop handles EClassOp when visited."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions(Context())
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


def test_run_getdefiningop_eclass_error_multiple_gdo():
    """Test that run_getdefiningop raises error for multiple get_defining_op in block."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions(Context())
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
    ctx = Context()
    ctx.register_dialect("test", lambda: test.Test)
    interp_functions = EqsatPDLInterpFunctions(ctx)
    interpreter.register_implementations(interp_functions)

    # Set up a mock rewriter
    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block, Region
    from xdsl.pattern_rewriter import PatternRewriter

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        root = test.TestOp()
    rewriter = PatternRewriter(root)
    interp_functions.rewriter = rewriter

    # Create operands and types for the operation
    operand = create_ssa_value(i32)
    result_type = i32

    # Create CreateOperationOp
    create_op = pdl_interp.CreateOperationOp(
        name="test.op",
        input_operands=[operand],
        input_attributes=[],
        input_result_types=[create_ssa_value(pdl.TypeType())],
    )

    # Run the create operation
    result = interp_functions.run_create_operation(
        interpreter, create_op, (operand, result_type)
    )

    # Should return the created operation
    assert len(result.values) == 1
    created_op = result.values[0]
    assert isinstance(created_op, Operation)
    assert created_op.name == "test.op"

    # Should add the operation to known_ops
    assert created_op in interp_functions.known_ops
    assert interp_functions.known_ops[created_op] is created_op

    # Should create an EClass operation and add it to eclass_union_find
    # The EClass should be created and inserted after the operation
    assert len(interp_functions.eclass_union_find._values) == 1  # pyright: ignore[reportPrivateUsage]
    eclass_op = interp_functions.eclass_union_find._values[0]  # pyright: ignore[reportPrivateUsage]
    assert isinstance(eclass_op, eqsat.EClassOp)


def test_run_create_operation_existing_operation_in_use():
    """Test that run_create_operation returns existing operation when it's still in use."""
    interpreter = Interpreter(ModuleOp([]))
    ctx = Context()
    ctx.register_dialect("test", lambda: test.Test)
    interp_functions = EqsatPDLInterpFunctions(ctx)
    interpreter.register_implementations(interp_functions)

    # Set up a mock rewriter
    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block, Region
    from xdsl.pattern_rewriter import PatternRewriter

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        root = test.TestOp()
    rewriter = PatternRewriter(root)
    interp_functions.rewriter = rewriter

    # Create an existing operation that's identical to what we'll create
    operand = create_ssa_value(i32)
    existing_op = test.TestOp((operand,), (i32,))

    # Create a user for the existing operation to ensure it's "in use"
    _user_op = test.TestOp((existing_op.results[0],), (i32,))

    # Add the existing operation to known_ops
    interp_functions.known_ops[existing_op] = existing_op

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

    # Should restore the rewriter's has_done_action state
    assert rewriter.has_done_action == initial_has_done_action


def test_run_create_operation_existing_operation_not_in_use():
    """Test that run_create_operation creates new operation when existing has no uses."""
    interpreter = Interpreter(ModuleOp([]))
    ctx = Context()
    ctx.register_dialect("test", lambda: test.Test)
    interp_functions = EqsatPDLInterpFunctions(ctx)
    interpreter.register_implementations(interp_functions)

    # Set up a mock rewriter
    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block, Region
    from xdsl.pattern_rewriter import PatternRewriter

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        root = test.TestOp()
    rewriter = PatternRewriter(root)
    interp_functions.rewriter = rewriter

    # Create an existing operation with no result uses
    operand = create_ssa_value(i32)
    existing_op = test.TestOp((operand,), (i32,))

    # Verify the existing operation has no uses
    assert len(existing_op.results) > 0, "Existing operation must have results"
    assert len(existing_op.results[0].uses) == 0, (
        "Existing operation result should have no uses"
    )

    # Add the existing operation to known_ops
    interp_functions.known_ops[existing_op] = existing_op

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

    # Should return a new operation (core behavior test)
    assert len(result.values) == 1
    created_op = result.values[0]
    assert isinstance(created_op, Operation)
    assert created_op.name == "test.op"

    # The key test: should get a new operation, not the existing unused one
    assert created_op is not existing_op

    # Should create an EClass operation
    assert interp_functions.eclass_union_find._values  # pyright: ignore[reportPrivateUsage]


def test_run_finalize_empty_stack():
    """Test that run_finalize handles empty backtrack stack correctly."""
    interpreter = Interpreter(ModuleOp([]))
    interp_functions = EqsatPDLInterpFunctions(Context())
    interpreter.register_implementations(interp_functions)

    # Test finalize with empty backtrack stack - should return empty values
    result = interpreter.run_op(pdl_interp.FinalizeOp(), ())
    assert result == ()


def test_backtrack_stack_manipulation():
    """Test that backtrack stack operations work correctly."""
    interp_functions = EqsatPDLInterpFunctions(Context())

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
    interp_functions = EqsatPDLInterpFunctions(Context())
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
