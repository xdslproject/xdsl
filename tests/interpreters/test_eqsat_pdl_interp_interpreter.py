from typing import Any

import pytest

from xdsl.context import Context
from xdsl.dialects import eqsat, pdl, pdl_interp, test
from xdsl.dialects.builtin import ModuleOp, i32, i64
from xdsl.interpreter import Interpreter
from xdsl.interpreters.eqsat_pdl_interp import EqsatPDLInterpFunctions
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
