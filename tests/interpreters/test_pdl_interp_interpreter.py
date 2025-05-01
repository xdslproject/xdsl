import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import pdl, pdl_interp, test
from xdsl.dialects.builtin import (
    FunctionType,
    ModuleOp,
    StringAttr,
    UnitAttr,
    i32,
    i64,
)
from xdsl.interpreter import Interpreter, Successor
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Block, Region
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.test_value import create_ssa_value


def test_getters():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    c0 = create_ssa_value(i32)
    c1 = create_ssa_value(i32)
    myattr = StringAttr("hello_attr")
    myprop = StringAttr("hello_prop")
    op = test.TestOp((c0, c1), (i32, i64), {"myattr": myattr}, {"myprop": myprop})
    op_res = op.results[0]

    assert interpreter.run_op(
        pdl_interp.GetOperandOp(1, create_ssa_value(pdl.OperationType())), (op,)
    ) == (c1,)

    assert interpreter.run_op(
        pdl_interp.GetResultOp(1, create_ssa_value(pdl.OperationType())), (op,)
    ) == (op.results[1],)

    assert (
        interpreter.run_op(
            pdl_interp.GetResultsOp(
                None,
                create_ssa_value(pdl.OperationType()),
                pdl.RangeType(pdl.ValueType()),
            ),
            (op,),
        )[0]
        == op.results
    )

    assert interpreter.run_op(
        pdl_interp.GetAttributeOp("myattr", create_ssa_value(pdl.OperationType())),
        (op,),
    ) == (myattr,)

    assert interpreter.run_op(
        pdl_interp.GetAttributeOp("myprop", create_ssa_value(pdl.OperationType())),
        (op,),
    ) == (myprop,)

    assert interpreter.run_op(
        pdl_interp.GetValueTypeOp(create_ssa_value(pdl.ValueType())), (c0,)
    ) == (i32,)

    assert interpreter.run_op(
        pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType())), (op_res,)
    ) == (op,)

    # Negative cases

    # Test GetOperandOp with out-of-bounds index
    assert interpreter.run_op(
        pdl_interp.GetOperandOp(5, create_ssa_value(pdl.OperationType())), (op,)
    ) == (None,)

    # Test GetResultOp with out-of-bounds index
    assert interpreter.run_op(
        pdl_interp.GetResultOp(5, create_ssa_value(pdl.OperationType())), (op,)
    ) == (None,)

    # Test GetResultsOp with single-SSA type but multiple results
    single_result_type_op = pdl_interp.GetResultsOp(
        None,
        create_ssa_value(pdl.OperationType()),
        pdl.ValueType(),
    )
    assert interpreter.run_op(single_result_type_op, (op,)) == (None,)

    # Test GetAttributeOp with non-existent attribute
    assert interpreter.run_op(
        pdl_interp.GetAttributeOp(
            "non_existent", create_ssa_value(pdl.OperationType())
        ),
        (op,),
    ) == (None,)

    # Test GetDefiningOpOp with non-OpResult value
    block_arg = Block((), arg_types=(i32,)).args[0]
    assert interpreter.run_op(
        pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType())), (block_arg,)
    ) == (None,)

    # Test GetDefiningOpOp with None input
    assert interpreter.run_op(
        pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType())), (None,)
    ) == (None,)


def test_check_operation_name():
    interpreter = Interpreter(ModuleOp([]))
    pdl_interp_functions = PDLInterpFunctions(Context())
    interpreter.register_implementations(pdl_interp_functions)

    truedest = Block()
    falsedest = Block()

    c0 = create_ssa_value(i32)
    c1 = create_ssa_value(i32)
    myattr = StringAttr("hello")
    op = test.TestOp((c0, c1), (i32, i64), {"myattr": myattr})

    check_name_op = pdl_interp.CheckOperationNameOp(
        "test.op", create_ssa_value(pdl.OperationType()), truedest, falsedest
    )

    trueresult = pdl_interp_functions.run_check_operation_name(
        interpreter, check_name_op, (op,)
    )

    assert isinstance(trueresult.terminator_value, Successor)
    assert isinstance(trueresult.terminator_value.block, Block)
    assert trueresult.terminator_value.block is truedest

    check_name_op_false = pdl_interp.CheckOperationNameOp(
        "test.other", create_ssa_value(pdl.OperationType()), truedest, falsedest
    )

    falseresult = pdl_interp_functions.run_check_operation_name(
        interpreter, check_name_op_false, (op,)
    )

    assert isinstance(falseresult.terminator_value, Successor)
    assert isinstance(falseresult.terminator_value.block, Block)
    assert falseresult.terminator_value.block is falsedest


def test_check_operand_count():
    interpreter = Interpreter(ModuleOp([]))
    pdl_interp_functions = PDLInterpFunctions(Context())
    interpreter.register_implementations(pdl_interp_functions)

    truedest = Block()
    falsedest = Block()

    c0 = create_ssa_value(i32)
    c1 = create_ssa_value(i32)
    myattr = StringAttr("hello")
    op = test.TestOp((c0, c1), (i32, i64), {"myattr": myattr})

    # Test exact operand count
    check_count_op = pdl_interp.CheckOperandCountOp(
        create_ssa_value(pdl.OperationType()),
        2,  # op has exactly 2 operands (c0, c1)
        truedest,
        falsedest,
        compareAtLeast=False,
    )

    exact_result = pdl_interp_functions.run_check_operand_count(
        interpreter, check_count_op, (op,)
    )

    assert isinstance(exact_result.terminator_value, Successor)
    assert exact_result.terminator_value.block is truedest

    # Test compareAtLeast=True
    check_count_op_at_least = pdl_interp.CheckOperandCountOp(
        create_ssa_value(pdl.OperationType()),
        1,  # op has 2 operands which is >= 1
        truedest,
        falsedest,
        compareAtLeast=True,
    )

    at_least_result = pdl_interp_functions.run_check_operand_count(
        interpreter, check_count_op_at_least, (op,)
    )

    assert isinstance(at_least_result.terminator_value, Successor)
    assert at_least_result.terminator_value.block is truedest

    # Test failing cases
    check_count_op_fail = pdl_interp.CheckOperandCountOp(
        create_ssa_value(pdl.OperationType()),
        3,  # op has only 2 operands
        truedest,
        falsedest,
        compareAtLeast=False,
    )

    fail_result = pdl_interp_functions.run_check_operand_count(
        interpreter, check_count_op_fail, (op,)
    )

    assert isinstance(fail_result.terminator_value, Successor)
    assert fail_result.terminator_value.block is falsedest


def test_check_result_count():
    interpreter = Interpreter(ModuleOp([]))
    pdl_interp_functions = PDLInterpFunctions(Context())
    interpreter.register_implementations(pdl_interp_functions)

    truedest = Block()
    falsedest = Block()

    c0 = create_ssa_value(i32)
    c1 = create_ssa_value(i32)
    myattr = StringAttr("hello")
    op = test.TestOp((c0, c1), (i32, i64), {"myattr": myattr})  # Has 2 results

    # Test exact result count
    check_result_op = pdl_interp.CheckResultCountOp(
        create_ssa_value(pdl.OperationType()),
        2,  # op has exactly 2 results
        truedest,
        falsedest,
        compareAtLeast=False,
    )

    exact_result = pdl_interp_functions.run_check_result_count(
        interpreter, check_result_op, (op,)
    )

    assert isinstance(exact_result.terminator_value, Successor)
    assert exact_result.terminator_value.block is truedest

    # Test compareAtLeast=True
    check_result_op_at_least = pdl_interp.CheckResultCountOp(
        create_ssa_value(pdl.OperationType()),
        1,  # op has 2 results which is >= 1
        truedest,
        falsedest,
        compareAtLeast=True,
    )

    at_least_result = pdl_interp_functions.run_check_result_count(
        interpreter, check_result_op_at_least, (op,)
    )

    assert isinstance(at_least_result.terminator_value, Successor)
    assert at_least_result.terminator_value.block is truedest

    # Test failing case
    check_result_op_fail = pdl_interp.CheckResultCountOp(
        create_ssa_value(pdl.OperationType()),
        3,  # op has only 2 results
        truedest,
        falsedest,
        compareAtLeast=False,
    )

    fail_result = pdl_interp_functions.run_check_result_count(
        interpreter, check_result_op_fail, (op,)
    )

    assert isinstance(fail_result.terminator_value, Successor)
    assert fail_result.terminator_value.block is falsedest


def test_check_attribute():
    interpreter = Interpreter(ModuleOp([]))
    pdl_interp_functions = PDLInterpFunctions(Context())
    interpreter.register_implementations(pdl_interp_functions)

    truedest = Block()
    falsedest = Block()

    # Test matching attribute
    check_attr_op = pdl_interp.CheckAttributeOp(
        StringAttr("hello"),  # Expected value
        create_ssa_value(pdl.AttributeType()),  # Input attribute
        truedest,
        falsedest,
    )

    match_result = pdl_interp_functions.run_check_attribute(
        interpreter, check_attr_op, (StringAttr("hello"),)
    )

    assert isinstance(match_result.terminator_value, Successor)
    assert match_result.terminator_value.block is truedest

    # Test non-matching attribute
    nomatch_result = pdl_interp_functions.run_check_attribute(
        interpreter, check_attr_op, (StringAttr("world"),)
    )

    assert isinstance(nomatch_result.terminator_value, Successor)
    assert nomatch_result.terminator_value.block is falsedest


def test_is_not_null():
    interpreter = Interpreter(ModuleOp([]))
    pdl_interp_functions = PDLInterpFunctions(Context())
    interpreter.register_implementations(pdl_interp_functions)

    truedest = Block()
    falsedest = Block()

    c0 = create_ssa_value(i32)

    # Test with non-null value
    is_not_null_op = pdl_interp.IsNotNullOp(
        create_ssa_value(pdl.ValueType()),
        truedest,
        falsedest,
    )

    notnull_result = pdl_interp_functions.run_is_not_null(
        interpreter, is_not_null_op, (c0,)
    )

    assert isinstance(notnull_result.terminator_value, Successor)
    assert notnull_result.terminator_value.block is truedest

    # Test with null value
    null_result = pdl_interp_functions.run_is_not_null(
        interpreter, is_not_null_op, (None,)
    )

    assert isinstance(null_result.terminator_value, Successor)
    assert null_result.terminator_value.block is falsedest


def test_are_equal():
    interpreter = Interpreter(ModuleOp([]))
    pdl_interp_functions = PDLInterpFunctions(Context())
    interpreter.register_implementations(pdl_interp_functions)

    truedest = Block()
    falsedest = Block()

    c0 = create_ssa_value(i32)
    c1 = create_ssa_value(i32)

    # Test with equal values
    are_equal_op = pdl_interp.AreEqualOp(
        create_ssa_value(pdl.ValueType()),
        create_ssa_value(pdl.ValueType()),
        truedest,
        falsedest,
    )

    equal_result = pdl_interp_functions.run_are_equal(
        interpreter, are_equal_op, (c0, c0)
    )

    assert isinstance(equal_result.terminator_value, Successor)
    assert equal_result.terminator_value.block is truedest

    # Test with unequal values
    unequal_result = pdl_interp_functions.run_are_equal(
        interpreter, are_equal_op, (c0, c1)
    )

    assert isinstance(unequal_result.terminator_value, Successor)
    assert unequal_result.terminator_value.block is falsedest


def test_create_attribute():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    # Create test attribute
    test_attr = StringAttr("test")

    # Test create attribute operation
    create_attr_op = pdl_interp.CreateAttributeOp(test_attr)
    result = interpreter.run_op(create_attr_op, ())

    assert len(result) == 1
    assert result[0] is test_attr


def test_create_operation():
    interpreter = Interpreter(ModuleOp([]))
    ctx = Context()
    ctx.register_dialect("test", lambda: test.Test)
    implementations = PDLInterpFunctions(ctx)
    interpreter.register_implementations(implementations)

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        root = test.TestOp()
        implementations.rewriter = PatternRewriter(root)

    # Create test values
    c0 = create_ssa_value(i32)
    c1 = create_ssa_value(i32)
    attr = StringAttr("test")

    # Test create operation
    create_op = pdl_interp.CreateOperationOp(
        name="test.op",
        inferred_result_types=UnitAttr(),
        input_attribute_names=[StringAttr("attr")],
        input_operands=[c0, c1],
        input_attributes=[create_ssa_value(pdl.AttributeType())],
        input_result_types=[create_ssa_value(pdl.TypeType())],
    )

    result = interpreter.run_op(create_op, (c0, c1, attr, i32))

    assert len(result) == 1
    assert isinstance(result[0], test.TestOp)
    created_op = result[0]
    assert len(created_op.operands) == 2
    assert created_op.ops == (c0, c1)
    assert created_op.attributes["attr"] is attr
    assert len(created_op.results) == 1
    assert created_op.results[0].type == i32
    # Verify that the operation was inserted:
    assert created_op.parent is block

    create_op_nonexistent = pdl_interp.CreateOperationOp(
        name="nonexistent.op",
        inferred_result_types=UnitAttr(),
        input_attribute_names=[StringAttr("attr")],
        input_operands=[c0, c1],
        input_attributes=[create_ssa_value(pdl.AttributeType())],
        input_result_types=[create_ssa_value(pdl.TypeType())],
    )
    with pytest.raises(InterpretationError):
        interpreter.run_op(create_op_nonexistent, (c0, c1, attr, i32))


def test_replace():
    interpreter = Interpreter(ModuleOp([]))
    ctx = Context()
    ctx.register_dialect("test", lambda: test.Test)
    pdl_interp_functions = PDLInterpFunctions(ctx)
    interpreter.register_implementations(pdl_interp_functions)

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    with ImplicitBuilder(block):
        target_val = create_ssa_value(i32)
        target_op = target_val.op
        _c1 = test.TestOp((), (i32,), {"replacement": UnitAttr()})
        # Create an operation that we'll replace
        user = test.TestOp((target_val,), (i32,))

    repl_owner = target_op.next_op
    assert repl_owner is not None
    repl_value = repl_owner.results[0]

    # Set up the rewriter for testing
    pdl_interp_functions.rewriter = PatternRewriter(target_op)

    # Before replacement, verify the target_op is in the block
    assert target_op.parent is block

    # Create the replace op
    replace_op = pdl_interp.ReplaceOp(
        create_ssa_value(pdl.OperationType()), [create_ssa_value(pdl.ValueType())]
    )

    # Execute the replace operation
    interpreter.run_op(replace_op, (target_op, repl_value))

    # Verify the replacement worked:
    # The target_op should be removed from the block
    assert target_op.parent is None
    assert user.operands[0] == repl_value


def test_replace_with_range():
    interpreter = Interpreter(ModuleOp([]))
    ctx = Context()
    ctx.register_dialect("test", lambda: test.Test)
    pdl_interp_functions = PDLInterpFunctions(ctx)
    interpreter.register_implementations(pdl_interp_functions)

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block

    with ImplicitBuilder(block):
        target_op = test.TestOp((), (i32, i32, i32, i32))
        c0 = create_ssa_value(i32)
        c1 = create_ssa_value(i32)
        c2 = create_ssa_value(i32)
        c3 = create_ssa_value(i32)
        # Create an operation that we'll replace
        user = test.TestOp(target_op.results)
    # Set up the rewriter for testing
    pdl_interp_functions.rewriter = PatternRewriter(target_op)

    # Create the replace op
    replace_op = pdl_interp.ReplaceOp(
        create_ssa_value(pdl.OperationType()),
        [
            create_ssa_value(pdl.ValueType()),
            create_ssa_value(pdl.RangeType(pdl.ValueType())),
            create_ssa_value(pdl.ValueType()),
        ],
    )

    interpreter.run_op(
        replace_op,
        (target_op, c0, (c1, c2), c3),
    )

    assert target_op.parent is None
    assert user.operands[0] == c0
    assert user.operands[1] == c1
    assert user.operands[2] == c2
    assert user.operands[3] == c3


def test_replace_with_range_invalid():
    interpreter = Interpreter(ModuleOp([]))
    ctx = Context()
    ctx.register_dialect("test", lambda: test.Test)
    pdl_interp_functions = PDLInterpFunctions(ctx)
    interpreter.register_implementations(pdl_interp_functions)

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block

    with ImplicitBuilder(block):
        target_op = test.TestOp((), (i32, i32, i32, i32))
        c0 = create_ssa_value(i32)
        c1 = create_ssa_value(i32)
        _c2 = create_ssa_value(i32)
        c3 = create_ssa_value(i32)
        # Create an operation that we'll replace
        test.TestOp(target_op.results)
    # Set up the rewriter for testing
    pdl_interp_functions.rewriter = PatternRewriter(target_op)

    # Create the replace op
    replace_op = pdl_interp.ReplaceOp(
        create_ssa_value(pdl.OperationType()),
        [
            create_ssa_value(pdl.ValueType()),
            create_ssa_value(pdl.RangeType(pdl.ValueType())),
            create_ssa_value(pdl.ValueType()),
        ],
    )

    with pytest.raises(InterpretationError):
        # include one result too little:
        interpreter.run_op(replace_op, (target_op, c0, (c1,), c3))


def test_func():
    # not really a unit test, just check if pdl_interp.func can be called.
    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    func_body = Region([Block()])
    with ImplicitBuilder(func_body.block):
        pdl_interp.FinalizeOp()
    func_body2 = func_body.clone()
    with ImplicitBuilder(block):
        op = test.TestOp()
        pdl_interp.FuncOp(
            "matcher",
            FunctionType.from_lists([pdl.OperationType()], []),
            None,
            None,
            func_body,
        )
        pdl_interp.FuncOp(
            "rewrite",
            FunctionType.from_lists([pdl.OperationType()], []),
            None,
            None,
            func_body2,
        )

    testmodule.verify()

    interpreter = Interpreter(testmodule)
    ctx = Context()
    ctx.register_dialect("test", lambda: test.Test)
    pdl_interp_functions = PDLInterpFunctions(ctx)
    interpreter.register_implementations(pdl_interp_functions)
    interpreter.call_op("matcher", (op,))

    with pytest.raises(InterpretationError):
        interpreter.call_op("rewrite", (op,))

    pdl_interp_functions.rewriter = PatternRewriter(op)

    with pytest.raises(InterpretationError):
        interpreter.call_op("matcher", (op,))

    interpreter.call_op("rewrite", (op,))
