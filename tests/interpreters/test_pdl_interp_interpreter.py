from xdsl.context import Context
from xdsl.dialects import pdl, pdl_interp, test
from xdsl.dialects.builtin import (
    ModuleOp,
    StringAttr,
    i32,
    i64,
)
from xdsl.interpreter import Interpreter, Successor
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Attribute, Block
from xdsl.utils.test_value import TestSSAValue


def test_getters():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    c0 = TestSSAValue(i32)
    c1 = TestSSAValue(i32)
    myattr = StringAttr("hello")
    op = test.TestOp((c0, c1), (i32, i64), {"myattr": myattr})
    op_res = op.results[0]

    assert interpreter.run_op(
        pdl_interp.GetOperandOp(1, TestSSAValue(pdl.OperationType())), (op,)
    ) == (c1,)

    assert interpreter.run_op(
        pdl_interp.GetResultOp(1, TestSSAValue(pdl.OperationType())), (op,)
    ) == (op.results[1],)

    assert interpreter.run_op(
        pdl_interp.GetAttributeOp("myattr", TestSSAValue(pdl.OperationType())), (op,)
    ) == (myattr,)

    assert interpreter.run_op(
        pdl_interp.GetValueTypeOp(TestSSAValue(pdl.ValueType())), (c0,)
    ) == (i32,)

    assert interpreter.run_op(
        pdl_interp.GetDefiningOpOp(TestSSAValue(pdl.OperationType())), (op_res,)
    ) == (op,)


def test_check_operation_name():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    truedest = Block()
    falsedest = Block()

    c0 = TestSSAValue(i32)
    c1 = TestSSAValue(i32)
    myattr = StringAttr("hello")
    op = test.TestOp((c0, c1), (i32, i64), {"myattr": myattr})

    trueresult = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckOperationNameOp(
            "test.op", TestSSAValue(pdl.OperationType()), truedest, falsedest
        ),
        (op,),
    )
    assert isinstance(trueresult.terminator_value, Successor)
    assert isinstance(trueresult.terminator_value.block, Block)
    assert trueresult.terminator_value.block == truedest

    falseresult = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckOperationNameOp(
            "test.other", TestSSAValue(pdl.OperationType()), truedest, falsedest
        ),
        (op,),
    )
    assert isinstance(falseresult.terminator_value, Successor)
    assert isinstance(falseresult.terminator_value.block, Block)
    assert falseresult.terminator_value.block == falsedest


def test_check_operand_count():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    truedest = Block()
    falsedest = Block()

    c0 = TestSSAValue(i32)
    c1 = TestSSAValue(i32)
    myattr = StringAttr("hello")
    op = test.TestOp((c0, c1), (i32, i64), {"myattr": myattr})

    # Test exact operand count
    exact_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckOperandCountOp(
            TestSSAValue(pdl.OperationType()),
            2,  # op has exactly 2 operands (c0, c1)
            truedest,
            falsedest,
            compareAtLeast=False,
        ),
        (op,),
    )
    assert isinstance(exact_result.terminator_value, Successor)
    assert exact_result.terminator_value.block == truedest

    # Test compareAtLeast=True
    at_least_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckOperandCountOp(
            TestSSAValue(pdl.OperationType()),
            1,  # op has 2 operands which is >= 1
            truedest,
            falsedest,
            compareAtLeast=True,
        ),
        (op,),
    )
    assert isinstance(at_least_result.terminator_value, Successor)
    assert at_least_result.terminator_value.block == truedest

    # Test failing cases
    fail_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckOperandCountOp(
            TestSSAValue(pdl.OperationType()),
            3,  # op has only 2 operands
            truedest,
            falsedest,
            compareAtLeast=False,
        ),
        (op,),
    )
    assert isinstance(fail_result.terminator_value, Successor)
    assert fail_result.terminator_value.block == falsedest


def test_check_result_count():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    truedest = Block()
    falsedest = Block()

    c0 = TestSSAValue(i32)
    c1 = TestSSAValue(i32)
    myattr = StringAttr("hello")
    op = test.TestOp((c0, c1), (i32, i64), {"myattr": myattr})  # Has 2 results

    # Test exact result count
    exact_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckResultCountOp(
            TestSSAValue(pdl.OperationType()),
            2,  # op has exactly 2 results
            truedest,
            falsedest,
            compareAtLeast=False,
        ),
        (op,),
    )
    assert isinstance(exact_result.terminator_value, Successor)
    assert exact_result.terminator_value.block == truedest

    # Test compareAtLeast=True
    at_least_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckResultCountOp(
            TestSSAValue(pdl.OperationType()),
            1,  # op has 2 results which is >= 1
            truedest,
            falsedest,
            compareAtLeast=True,
        ),
        (op,),
    )
    assert isinstance(at_least_result.terminator_value, Successor)
    assert at_least_result.terminator_value.block == truedest

    # Test failing case
    fail_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckResultCountOp(
            TestSSAValue(pdl.OperationType()),
            3,  # op has only 2 results
            truedest,
            falsedest,
            compareAtLeast=False,
        ),
        (op,),
    )
    assert isinstance(fail_result.terminator_value, Successor)
    assert fail_result.terminator_value.block == falsedest


def test_check_attribute():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    truedest = Block()
    falsedest = Block()

    # Test matching attribute
    match_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckAttributeOp(
            StringAttr("hello"),  # Expected value
            TestSSAValue(Attribute),  # Input attribute
            truedest,
            falsedest,
        ),
        (StringAttr("hello"),),  # Actual value that matches
    )
    assert isinstance(match_result.terminator_value, Successor)
    assert match_result.terminator_value.block == truedest

    # Test non-matching attribute
    nomatch_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckAttributeOp(
            StringAttr("hello"),  # Expected value
            TestSSAValue(Attribute),  # Input attribute
            truedest,
            falsedest,
        ),
        (StringAttr("world"),),  # Different value
    )
    assert isinstance(nomatch_result.terminator_value, Successor)
    assert nomatch_result.terminator_value.block == falsedest
