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
from xdsl.ir import Block
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


def test_checks():
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
